"""Pipeline de doublage vidéo : extraction audio -> transcription (mlx-whisper)
-> traduction (Google Translate) -> synthèse (edge-tts) -> remux (ffmpeg)."""

import asyncio
import os
import re
import subprocess
import tempfile
import wave

import edge_tts
import imageio_ffmpeg
import mlx_whisper
import requests
import sherpa_onnx

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 16000

# Dossier central des modèles IA (partageable entre plusieurs projets)
MODELS_DIR = os.environ.get("VIDEO_DUBBER_MODELS",
                            "/Volumes/Mustang/DEV/modelsAI/video")
WHISPER_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
SEG_MODEL_NAME = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB_MODEL_NAME = "wespeaker_en_voxceleb_resnet34_LM.onnx"


_WHISPER_PATH = None
WHISPER_REQUIRED_FILES = ("config.json", "weights.safetensors")


def whisper_model_path(progress=None):
    """Chemin local du modèle Whisper. Vérifié une fois par session ;
    téléchargé uniquement si des fichiers requis manquent."""
    global _WHISPER_PATH
    if _WHISPER_PATH:
        return _WHISPER_PATH
    local = os.path.join(MODELS_DIR, "whisper-large-v3-turbo")
    missing = [f for f in WHISPER_REQUIRED_FILES
               if not os.path.exists(os.path.join(local, f))]
    if not missing:
        if progress:
            progress(f"Modèle Whisper local : {local}", 0.06)
        _WHISPER_PATH = local
        return local
    if progress:
        progress(f"Téléchargement du modèle Whisper (~1,5 Go, "
                 f"manque : {', '.join(missing)})…", 0.06)
    from huggingface_hub import snapshot_download
    _WHISPER_PATH = snapshot_download(WHISPER_MODEL_REPO, local_dir=local)
    return _WHISPER_PATH


def _model_file(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modèle manquant : {path}\nCopie les modèles de diarisation "
            f"depuis le dossier models/ du projet.")
    return path


def check_environment():
    """Vérifie libs, ffmpeg, modèles et services réseau.
    Retourne une liste de (nom, ok, détail)."""
    checks = []

    for mod, pkg in [("mlx_whisper", "mlx-whisper"),
                     ("edge_tts", "edge-tts"),
                     ("sherpa_onnx", "sherpa-onnx"),
                     ("requests", "requests"),
                     ("gradio", "gradio"),
                     ("imageio_ffmpeg", "imageio-ffmpeg")]:
        try:
            __import__(mod)
            checks.append((f"Librairie {pkg}", True, "installée"))
        except ImportError:
            checks.append((f"Librairie {pkg}", False,
                           f"manquante → pip install {pkg}"))

    checks.append(("Binaire ffmpeg", os.path.exists(FFMPEG), FFMPEG))

    wdir = os.path.join(MODELS_DIR, "whisper-large-v3-turbo")
    w_missing = [f for f in WHISPER_REQUIRED_FILES
                 if not os.path.exists(os.path.join(wdir, f))]
    checks.append(("Modèle Whisper", not w_missing,
                   wdir if not w_missing
                   else f"manque : {', '.join(w_missing)}"))
    checks.append(("Modèle segmentation locuteurs",
                   os.path.exists(os.path.join(MODELS_DIR, SEG_MODEL_NAME)),
                   SEG_MODEL_NAME))
    checks.append(("Modèle embeddings locuteurs",
                   os.path.exists(os.path.join(MODELS_DIR, EMB_MODEL_NAME)),
                   EMB_MODEL_NAME))

    if _azure_creds():
        try:
            _azure_translate(["test"], "en", "fr")
            checks.append(("Traduction Azure", True,
                           "en ligne — identifiants OK"))
        except Exception as e:
            checks.append(("Traduction Azure", False,
                           f"en échec : {str(e)[:80]}"))
    else:
        try:
            requests.get("https://translate.googleapis.com", timeout=4)
            checks.append(("Traduction Google (repli)", True,
                           "en ligne — Azure non configuré"))
        except Exception:
            checks.append(("Traduction Google (repli)", False,
                           "injoignable (pas de connexion ?)"))
    try:
        ok = bool(asyncio.run(edge_tts.list_voices()))
    except Exception:
        ok = False
    checks.append(("Voix neurales Edge-TTS", ok,
                   "en ligne" if ok else "injoignables"))

    checks.append(("Clonage vocal Chatterbox (optionnel)", True,
                   "installé" if chatterbox_available()
                   else "non installé — option de clonage inactive"))

    return checks

# Voix par défaut par langue cible (locuteur féminin/masculin réalistes)
DEFAULT_VOICES = {
    "fr": "fr-FR-DeniseNeural",
    "en": "en-US-AriaNeural",
    "es": "es-ES-ElviraNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "nl": "nl-NL-ColetteNeural",
    "ar": "ar-SA-ZariyahNeural",
    "ja": "ja-JP-NanamiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ko": "ko-KR-SunHiNeural",
}


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{r.stderr[-2000:]}")
    return r


def media_duration(path):
    r = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        raise RuntimeError("Impossible de lire la durée du média")
    h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mn * 60 + s


def extract_audio(video, wav_path):
    run([FFMPEG, "-y", "-i", video, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-c:a", "pcm_s16le", wav_path])


def transcribe(wav_path, progress=None):
    import numpy as np
    with wave.open(wav_path, "rb") as w:
        audio = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                 .astype(np.float32) / 32768.0)
    result = mlx_whisper.transcribe(
        audio, path_or_hf_repo=whisper_model_path(progress),
        word_timestamps=True)
    dur = len(audio) / SAMPLE_RATE
    segments = []
    for s in result["segments"]:
        text = s["text"].strip()
        if not text:
            continue
        start, end = s["start"], s["end"]
        words = s.get("words") or []
        if words:        # frontières précises : 1er et dernier mot
            start, end = words[0]["start"], words[-1]["end"]
        # borne à la durée réelle (Whisper hallucine parfois la fin)
        start, end = max(0.0, start), min(end, dur)
        if end - start < 0.05:
            continue
        segments.append({"start": start, "end": end, "text": text})
    return segments, result.get("language", "auto")


_DIARIZER = None
_DIARIZER_KEY = None


def _get_diarizer(num_speakers=0, threshold=0.4):
    """Diarizer mis en cache par (num_speakers, threshold).
    num_speakers>0 force le nombre de clusters."""
    global _DIARIZER, _DIARIZER_KEY
    key = (num_speakers, threshold)
    if _DIARIZER is None or _DIARIZER_KEY != key:
        seg_model = _model_file(SEG_MODEL_NAME)
        emb_model = _model_file(EMB_MODEL_NAME)
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=seg_model),
                num_threads=4),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=emb_model, num_threads=4),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers if num_speakers > 0 else -1,
                threshold=threshold),
            min_duration_on=0.2,
            min_duration_off=0.3)
        _DIARIZER = sherpa_onnx.OfflineSpeakerDiarization(config)
        _DIARIZER_KEY = key
    return _DIARIZER


def diarize(wav_path, num_speakers=0, threshold=0.4, progress=None):
    """Retourne les segments locuteurs [{start, end, speaker}, ...]."""
    import numpy as np
    with wave.open(wav_path, "rb") as w:
        samples = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                   .astype(np.float32) / 32768.0)
    sd = _get_diarizer(num_speakers, threshold)

    def cb(done, total):
        if progress and total:
            progress(f"Diarisation {done}/{total}", 0.2 + 0.1 * done / total)
        return 0

    result = sd.process(samples, cb).sort_by_start_time()
    return [{"start": s.start, "end": s.end, "speaker": s.speaker}
            for s in result]


def assign_speakers(segments, dia_segments):
    """Attribue à chaque segment Whisper le locuteur dominant (max recouvrement)."""
    for seg in segments:
        best, best_overlap = 0, -1.0
        for d in dia_segments:
            overlap = min(seg["end"], d["end"]) - max(seg["start"], d["start"])
            if overlap > best_overlap:
                best_overlap, best = overlap, d["speaker"]
        seg["speaker"] = best
    return segments


def _estimate_f0(samples, sr=16000):
    """F0 médian (Hz) par autocorrelation, trames de 40ms / pas de 20ms."""
    import numpy as np
    frame, hop = int(0.04 * sr), int(0.02 * sr)
    lo, hi = int(sr / 400), int(sr / 70)  # plage 70-400 Hz
    f0s = []
    for off in range(0, len(samples) - frame, hop):
        x = samples[off:off + frame].astype(np.float64)
        if np.sqrt((x ** 2).mean()) < 0.005:  # trame non voisée
            continue
        x = x - x.mean()
        r = np.correlate(x, x, mode="full")[len(x) - 1:]
        if r[0] <= 0:
            continue
        peak = lo + int(np.argmax(r[lo:hi]))
        if r[peak] / r[0] > 0.3:
            f0s.append(sr / peak)
    return float(np.median(f0s)) if f0s else 0.0


def estimate_genders(wav_path, segments, speaker_ids):
    """Genre estimé par locuteur via la hauteur de voix (seuil ~150 Hz)."""
    import numpy as np
    with wave.open(wav_path, "rb") as w:
        audio = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                 .astype(np.float32) / 32768.0)
    genders = {}
    for sp in speaker_ids:
        chunks = [audio[int(s["start"] * SAMPLE_RATE):
                        int(s["end"] * SAMPLE_RATE)]
                  for s in segments if s.get("speaker") == sp]
        if not chunks:
            genders[sp] = "Unknown"
            continue
        f0 = _estimate_f0(np.concatenate(chunks))
        genders[sp] = ("Male" if f0 < 165 else "Female") if f0 else "Unknown"
    return genders


def auto_assign_voices(speaker_ids, lang, genders=None):
    """Répartit les locuteurs sur des voix distinctes en alternant F/M."""
    async def _list():
        return await edge_tts.list_voices()
    try:
        all_voices = asyncio.run(_list())
    except Exception:
        all_voices = []
    females = sorted(v["ShortName"] for v in all_voices
                     if v["Locale"].startswith(lang)
                     and v.get("Gender") == "Female")
    males = sorted(v["ShortName"] for v in all_voices
                   if v["Locale"].startswith(lang) and v.get("Gender") == "Male")
    if not females and not males:
        fallback = DEFAULT_VOICES.get(lang, "en-US-AriaNeural")
        return {sp: fallback for sp in speaker_ids}

    genders = genders or {}
    assign = {}
    counters = {"Female": 0, "Male": 0, "Unknown": 0}
    for i, sp in enumerate(speaker_ids):
        g = genders.get(sp, "Unknown")
        if g == "Unknown":          # pas de pitch fiable : alterner
            g = "Female" if i % 2 == 0 else "Male"
        pool = females if g == "Female" else males
        if not pool:
            pool = females or males
        c = counters[g]
        assign[sp] = pool[c % len(pool)]
        counters[g] = c + 1
    return assign


def _gtranslate(text, source, target):
    r = requests.get(
        "https://translate.googleapis.com/translate_a/single",
        params={"client": "gtx", "sl": source, "tl": target,
                "dt": "t", "q": text},
        timeout=20)
    r.raise_for_status()
    return "".join(chunk[0] for chunk in r.json()[0] if chunk[0])


AZURE_ENDPOINT = "https://api.cognitive.microsofttranslator.com"


def _azure_creds():
    """(key, region) Azure Translator : env vars, puis dub_secrets.py
    (gitignoré). None si non configuré."""
    key = os.environ.get("AZURE_TRANSLATOR_KEY")
    region = os.environ.get("AZURE_TRANSLATOR_REGION")
    if key and region:
        return key, region
    try:
        import dub_secrets
        k = getattr(dub_secrets, "AZURE_TRANSLATOR_KEY", None)
        r = getattr(dub_secrets, "AZURE_TRANSLATOR_REGION", None)
        if k and r:
            return k, r
    except ImportError:
        pass
    return None


def _azure_translate(texts, source, target):
    """Traduit une liste de textes via Azure Translator (une requête).
    source=None -> détection auto."""
    creds = _azure_creds()
    if not creds:
        raise RuntimeError("Identifiants Azure Translator absents")
    key, region = creds
    params = {"api-version": "3.0", "to": target}
    if source:
        params["from"] = source
    r = requests.post(f"{AZURE_ENDPOINT}/translate",
                      params=params,
                      headers={"Ocp-Apim-Subscription-Key": key,
                               "Ocp-Apim-Subscription-Region": region,
                               "Content-Type": "application/json"},
                      json=[{"Text": t} for t in texts],
                      timeout=30)
    r.raise_for_status()
    return [item["translations"][0]["text"] for item in r.json()]


def translate_text(text, source, target):
    """Un texte : Azure si configuré, sinon Google."""
    if _azure_creds():
        return _azure_translate(
            [text], None if source == "auto" else source, target)[0]
    return _gtranslate(text, "auto" if source == "auto" else source,
                       target)


def _gtranslate_segments(segments, sl, target):
    """Traduction segment par segment via l'endpoint Google gratuit."""
    import time
    for seg in segments:
        delay = 1.0
        for attempt in range(6):
            try:
                seg["translation"] = _gtranslate(seg["text"], sl, target)
                seg["trans_lang"] = target
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay *= 2
        time.sleep(0.3)  # rester sous la limite de requêtes/seconde


def translate_segments(segments, source, target):
    """Traduit les segments : Azure en batch (~50 par requête) si les
    identifiants sont configurés, sinon Google (appel par segment).
    En cas d'échec d'un lot Azure, ce lot bascule sur Google."""
    sl = "auto" if source == "auto" else source
    if _azure_creds():
        bsize = 50
        for start in range(0, len(segments), bsize):
            chunk = segments[start:start + bsize]
            try:
                trs = _azure_translate(
                    [s["text"] for s in chunk],
                    None if sl == "auto" else sl, target)
                for s, t in zip(chunk, trs):
                    s["translation"], s["trans_lang"] = t, target
            except Exception:
                _gtranslate_segments(chunk, sl, target)
    else:
        _gtranslate_segments(segments, sl, target)
    return segments


def apply_edited_translations(segments, edited_by_speaker, target_lang):
    """Applique les traductions corrigées par l'utilisateur.
    edited_by_speaker : {speaker_id: texte multiligne} — 1 ligne = 1 segment,
    dans l'ordre chronologique. Lignes en trop -> ajoutées au dernier
    segment ; lignes manquantes -> la traduction proposée est conservée."""
    segs_by_sp = {}
    for s in segments:
        segs_by_sp.setdefault(s.get("speaker", 0), []).append(s)
    for sp, text in edited_by_speaker.items():
        lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
        segs = segs_by_sp.get(sp, [])
        if not lines or not segs:
            continue
        for i, s in enumerate(segs):
            if i < len(lines):
                s["translation"] = lines[i]
        if len(lines) > len(segs):
            segs[-1]["translation"] = " ".join(lines[len(segs) - 1:])
        for s in segs:
            if s.get("translation"):
                s["trans_lang"] = target_lang
    return segments


async def _tts_one(text, voice, mp3_path):
    await edge_tts.Communicate(text, voice).save(mp3_path)


def synthesize(text, voice, mp3_path):
    """Synthèse edge-tts avec retries (réseau instable / rate-limit)."""
    import time
    delay = 1.0
    for attempt in range(3):
        try:
            asyncio.run(_tts_one(text, voice, mp3_path))
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(delay)
            delay *= 2


def synthesize_batch(items, progress=None):
    """Synthétise [(text, voice, path), ...] en parallèle (8 max),
    retries par segment. progress(done_count) appelé à chaque fin."""
    done = [0]

    async def _run():
        sem = asyncio.Semaphore(8)

        async def one(t, v, p):
            async with sem:
                last = None
                for attempt in range(3):
                    try:
                        await edge_tts.Communicate(t, v).save(p)
                        break
                    except Exception as e:
                        last = e
                        await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    raise last
            done[0] += 1
            if progress:
                progress(done[0])

        await asyncio.gather(*[one(*it) for it in items])

    asyncio.run(_run())


def fit_segments_parallel(jobs, progress=None, workers=4):
    """jobs : [(raw_path, slot_seconds, wav_out, avail_seconds)] —
    calage ffmpeg en parallèle. progress(done_count) à chaque fin.
    Retourne les facteurs d'accélération requis, alignés sur jobs."""
    from concurrent.futures import ThreadPoolExecutor
    done = [0]

    def _fit(job):
        speed = fit_segment(*job)
        done[0] += 1
        if progress:
            progress(done[0])
        return speed

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_fit, jobs))


_CHATTERBOX = None


def _get_chatterbox(progress=None):
    """Chatterbox Multilingual, chargé une fois (MPS si dispo, sinon CPU)."""
    global _CHATTERBOX
    if _CHATTERBOX is None:
        import torch
        import perth
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        # resemble-perth 1.0.1 : le watermarker réel est None sur cette
        # plateforme -> no-op, sinon le constructeur du modèle plante
        if perth.PerthImplicitWatermarker is None:
            perth.PerthImplicitWatermarker = perth.DummyWatermarker
        if progress:
            progress("Chargement du modèle de clonage Chatterbox…", 0.35)
        device = ("mps" if torch.backends.mps.is_available() else "cpu")
        try:
            _CHATTERBOX = ChatterboxMultilingualTTS.from_pretrained(
                device=device)          # str attendu : "cpu"/"mps"/"cuda"
        except Exception:
            _CHATTERBOX = ChatterboxMultilingualTTS.from_pretrained(
                device="cpu")
    return _CHATTERBOX


def chatterbox_available():
    try:
        import chatterbox  # noqa: F401
        return True
    except ImportError:
        return False


def synthesize_clone(text, ref_wav, lang_id, wav_out):
    """Synthèse par clonage : la voix de `ref_wav` prononce `text`.
    Écrit un wav au taux natif du modèle (le calage/resample est fait
    ensuite par fit_segment)."""
    import soundfile as sf
    model = _get_chatterbox()
    audio = model.generate(text, language_id=lang_id,
                           audio_prompt_path=ref_wav)
    sf.write(wav_out, audio.squeeze(0).cpu().numpy(), model.sr)


def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def read_wav_frames(path):
    with wave.open(path, "rb") as w:
        return w.readframes(w.getnframes())


def _atempo_chain(factor):
    """Décompose un facteur de vitesse en filtres atempo (0.5..2.0 chacun)."""
    parts = []
    f = factor
    while f > 2.0:
        parts.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        parts.append("atempo=0.5")
        f /= 0.5
    parts.append(f"atempo={f:.4f}")
    return ",".join(parts)


def fit_segment(mp3_path, slot_seconds, wav_out, avail_seconds=None):
    """Convertit un mp3 TTS en wav calé sur le créneau du segment.
    `avail_seconds` = temps réellement disponible en empruntant la pause
    qui suit le segment (jusqu'au prochain segment) : c'est elle qui
    décide de l'accélération, pas le créneau seul — ainsi la voix peut
    déborder dans le silence suivant plutôt que d'être survoltée.
    Retourne le facteur d'accélération requis (dur/avail) pour
    permettre de signaler les traductions trop longues."""
    dur = wav_duration_from_file(mp3_path)
    if slot_seconds <= 0:
        slot_seconds = dur
    avail = max(avail_seconds or slot_seconds, 0.05)
    speed = dur / avail                    # >1 : audio plus long que le temps libre
    filters = []
    if speed > 1.12:                       # trop long -> accélérer
        filters.append(_atempo_chain(min(speed, 4.0)))
    elif dur / slot_seconds < 0.95 and slot_seconds - dur <= 1.5:
        # léger décalage : étirer doucement (min x0.85). Au-delà on
        # préfère la voix naturelle + silence en fin de créneau.
        filters.append(_atempo_chain(
            max(dur / (slot_seconds * 0.95), 0.85)))
    run([FFMPEG, "-y", "-i", mp3_path, "-ac", "1", "-ar", str(SAMPLE_RATE)]
        + (["-af", ",".join(filters)] if filters else [])
        + ["-c:a", "pcm_s16le", wav_out])
    return speed


def wav_duration_from_file(path):
    r = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def build_dub_track(segments, tts_wavs, total_seconds, out_wav):
    """Assemble les segments TTS sur une piste silence de la durée de la vidéo."""
    n = int(total_seconds * SAMPLE_RATE) + SAMPLE_RATE
    buf = bytearray(n * 2)
    for i, seg in enumerate(segments):
        if tts_wavs[i] is None:               # segment sans texte -> silence
            continue
        frames = read_wav_frames(tts_wavs[i])
        start = int(seg["start"] * SAMPLE_RATE) * 2
        next_start = (int(segments[i + 1]["start"] * SAMPLE_RATE) * 2
                      if i + 1 < len(segments) else len(buf))
        end = min(start + len(frames), next_start, len(buf))
        if end > start:
            buf[start:end] = frames[: end - start]
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(buf))


def mix_duck(src_wav, segments, tts_wavs, dub_wav, out_wav,
             original_volume):
    """Piste finale : voix doublées + audio original atténué
    (`original_volume`) uniquement sous les segments de parole, avec
    rampes de 120 ms — le reste (jingle, musique, ambiance) reste à
    100 %. original_volume=0 -> original uniquement hors parole."""
    import numpy as np

    def read(path):
        with wave.open(path, "rb") as w:
            return (np.frombuffer(w.readframes(w.getnframes()),
                                  dtype=np.int16)
                    .astype(np.float32) / 32768.0)

    orig, dub = read(src_wav), read(dub_wav)
    n = max(len(orig), len(dub))
    orig = np.pad(orig, (0, n - len(orig)))
    dub = np.pad(dub, (0, n - len(dub)))

    env = np.ones(n, dtype=np.float32)
    ramp = int(0.12 * SAMPLE_RATE)
    for i, seg in enumerate(segments):
        if tts_wavs[i] is None:
            continue
        s = int(seg["start"] * SAMPLE_RATE)
        # le ducking couvre l'étendue RÉELLE de l'audio doublé (qui peut
        # déborder dans la pause suivante), pas seulement le créneau
        with wave.open(tts_wavs[i], "rb") as w:
            frames = w.getnframes()
        nxt = (int(segments[i + 1]["start"] * SAMPLE_RATE)
               if i + 1 < len(segments) else n)
        e = min(s + frames, nxt, n)
        a0, r1 = max(0, s - ramp), min(n, e + ramp)
        if a0 < s:
            env[a0:s] = np.minimum(
                env[a0:s],
                np.linspace(1.0, original_volume, s - a0))
        env[s:e] = original_volume
        if r1 > e:
            env[e:r1] = np.minimum(
                env[e:r1],
                np.linspace(original_volume, 1.0, r1 - e))

    mix = np.clip(orig * env + dub, -1.0, 1.0)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())


def mux(video, audio_wav, out_path, original_volume=0.1):
    """Mux simple : vidéo + piste audio déjà mixée."""
    run([FFMPEG, "-y", "-i", video, "-i", audio_wav,
         "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out_path])


def list_voices(lang_prefix):
    """Voix edge-tts disponibles pour une langue (nécessite le réseau)."""
    async def _list():
        return await edge_tts.list_voices()
    try:
        voices = asyncio.run(_list())
    except Exception:
        return [DEFAULT_VOICES.get(lang_prefix, "en-US-AriaNeural")]
    matches = [v["ShortName"] for v in voices
               if v["Locale"].startswith(lang_prefix)]
    return sorted(matches) or [DEFAULT_VOICES.get(lang_prefix,
                                                "en-US-AriaNeural")]


def analyze_video(video_path, diarize_speakers=True, num_speakers=0,
                  target_lang=None, progress=None):
    """Phase 1 : extraction audio + transcription Whisper (+ diarisation).
    Si target_lang est fourni, produit aussi une traduction proposée par
    segment (éditable dans l'UI avant le rendu).
    Retourne un dict d'état à passer à render_video()."""
    def p(msg, frac):
        if progress:
            progress(msg, frac)

    tmp = tempfile.mkdtemp(prefix="dub_")
    src_wav = os.path.join(tmp, "source.wav")

    p("Extraction de l'audio…", 0.02)
    extract_audio(video_path, src_wav)

    p("Transcription (Whisper)…", 0.05)
    segments, src_lang = transcribe(src_wav, progress=p)
    if not segments:
        raise RuntimeError("Aucune parole détectée dans la vidéo.")

    if diarize_speakers:
        p("Détection des locuteurs (diarisation)…", 0.15)
        dia_segments = diarize(src_wav, num_speakers=num_speakers,
                               progress=p)
        assign_speakers(segments, dia_segments)
    else:
        for seg in segments:
            seg["speaker"] = 0

    speakers = sorted({s["speaker"] for s in segments})
    genders = (estimate_genders(src_wav, segments, speakers)
               if len(speakers) > 1 else {})

    trans_lang = None
    if target_lang:
        p("Traduction proposée…", 0.27)
        try:
            translate_segments(segments, src_lang, target_lang)
            trans_lang = target_lang
        except Exception:
            pass  # le rendu réessaiera ; les zones restent éditables

    p("Analyse terminée.", 0.3)
    return {"tmp": tmp, "src_wav": src_wav, "src_path": video_path,
            "segments": segments, "src_lang": src_lang,
            "speakers": speakers, "genders": genders,
            "trans_lang": trans_lang, "total": media_duration(video_path)}


_TMP_MARKERS = ("/gradio", tempfile.gettempdir(), "/var/folders",
                "/private/tmp", "/private/var")


def is_temp_path(path):
    """Vrai si le chemin est un fichier temporaire (upload navigateur)."""
    return any(m in os.path.dirname(os.path.abspath(path))
               for m in _TMP_MARKERS)


def default_output_dir(video_path):
    """Dossier de travail : ~/Downloads/<nom de la vidéo>/ — regroupe
    l'analyse, les traductions sauvegardées et la vidéo doublée."""
    base = os.path.splitext(os.path.basename(video_path))[0] or "video"
    d = os.path.join(os.path.expanduser("~/Downloads"), base)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return os.path.expanduser("~/Downloads")
    return d


def translation_file(video_path, speaker_index, target_lang):
    """Fichier de sauvegarde de la traduction éditée d'un locuteur,
    dans le même dossier que la vidéo source (numéro séquentiel)."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(
        default_output_dir(video_path),
        f"{base}_traduction_{speaker_index + 1:02d}_{target_lang}.txt")


def analysis_file(video_path):
    """JSON de l'état d'analyse, à côté de la vidéo source."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(default_output_dir(video_path),
                        f"{base}_analyse.json")


def _analysis_wav(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(default_output_dir(video_path),
                        f"{base}_analyse.wav")


def save_analysis(state, video_path, diarize_speakers, num_speakers):
    """Persiste l'état d'analyse + le wav extrait, pour reprendre
    sans relancer Whisper/diarisation."""
    import json
    import shutil
    wav_dst = _analysis_wav(video_path)
    try:
        if os.path.abspath(state["src_wav"]) != os.path.abspath(wav_dst):
            shutil.copy2(state["src_wav"], wav_dst)
        data = {k: state[k] for k in
                ("src_path", "src_lang", "speakers", "genders",
                 "trans_lang", "total", "segments")}
        data.update({"src_wav": wav_dst,
                     "diarize": diarize_speakers,
                     "num_speakers": num_speakers,
                     "src_mtime": os.path.getmtime(video_path)})
        with open(analysis_file(video_path), "w",
                  encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


def load_analysis(video_path, diarize_speakers, num_speakers):
    """Recharge une analyse sauvegardée si elle correspond aux mêmes
    paramètres et à la même vidéo (mtime). None sinon."""
    import json
    path = analysis_file(video_path)
    if not os.path.exists(path):
        return None
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    wav = data.get("src_wav") or ""
    if not os.path.exists(wav):
        wav = _analysis_wav(video_path)
    if (data.get("diarize") != diarize_speakers
            or data.get("num_speakers") != num_speakers
            or data.get("src_mtime") != os.path.getmtime(video_path)
            or not data.get("segments")
            or not os.path.exists(wav)):
        return None
    data["src_wav"] = wav
    data["genders"] = {int(k): v for k, v in
                       data.get("genders", {}).items()}
    data["tmp"] = tempfile.mkdtemp(prefix="dub_")
    data["restored"] = True
    return data


def _avail_slots(segments, total_seconds, margin=0.08):
    """Temps libre réel par segment : jusqu'au début du segment suivant
    (moins une marge de 80 ms) ou jusqu'à la fin de la vidéo. Permet à
    une voix de déborder dans la pause qui suit au lieu d'être
    exagérément accélérée."""
    out = []
    for i, seg in enumerate(segments):
        nxt = (segments[i + 1]["start"] if i + 1 < len(segments)
               else total_seconds)
        out.append(max(nxt - margin, seg["start"] + 0.05) - seg["start"])
    return out


def estimate_speech_seconds(text, lang=None):
    """Durée parlée estimée avant TTS — ~13 car./s (langues latines,
    ~6 car./s pour le CJK), marge 5 %."""
    n = len(re.sub(r"\s+", "", text or ""))
    rate = 6.0 if lang in ("ja", "zh", "ko") else 13.0
    return n / rate * 1.05


def translation_warnings(segments, total_seconds, lang=None,
                         speeds=None, threshold=1.4):
    """Segments dont la traduction est trop longue pour le temps libre.
    `speeds` : facteurs réels mesurés post-TTS ; sinon estimation sur
    le texte. Retourne des dicts {index, speaker, start, end, speed,
    text, translation}."""
    avails = _avail_slots(segments, total_seconds)
    out = []
    for i, seg in enumerate(segments):
        trad = (seg.get("translation") or "").strip()
        if not trad:
            continue
        sp = (speeds[i] if speeds is not None
              else estimate_speech_seconds(trad, lang)
              / max(avails[i], 0.05))
        if sp > threshold:
            out.append({"index": i, "speaker": seg.get("speaker", 0),
                        "start": seg["start"], "end": seg["end"],
                        "speed": sp, "text": seg.get("text", ""),
                        "translation": trad})
    return out


def render_video(video_path, state, target_lang, speaker_voices,
                 original_volume=0.1, output_dir=None,
                 edited_translations=None, clone_voices=False,
                 progress=None):
    """Phase 2 : traduction (corrigée par l'utilisateur si fournie)
    + synthèse par locuteur + mixage.
    clone_voices=True : Chatterbox clone la voix de chaque locuteur depuis
    l'audio original au lieu des voix neurales (fichier *_CHTBOX_*)."""
    def p(msg, frac):
        if progress:
            progress(msg, frac)

    tmp, segments = state["tmp"], state["segments"]
    dub_wav = os.path.join(tmp, "dub.wav")
    default_voice = DEFAULT_VOICES.get(target_lang, "en-US-AriaNeural")

    if edited_translations:
        apply_edited_translations(segments, edited_translations,
                                  target_lang)
    stale = [s for s in segments
             if not s.get("translation")
             or s.get("trans_lang") != target_lang]
    if stale:
        p("Traduction…", 0.35)
        translate_segments(stale, state["src_lang"], target_lang)

    clone_refs = {}
    if clone_voices:
        _get_chatterbox(progress=p)
        for sp in state["speakers"]:
            clone_refs[sp] = original_excerpt(state, sp)

    tts_wavs = [None] * len(segments)
    avails = _avail_slots(segments, state["total"])
    speeds = [1.0] * len(segments)
    if clone_voices:
        # Clonage : séquentiel (le modèle MPS ne parallélise pas)
        for i, seg in enumerate(segments):
            text = (seg.get("translation") or "").strip()
            if not text:
                continue
            sp = seg.get("speaker", 0)
            raw = os.path.join(tmp, f"tts_{i:04d}.mp3")
            wav = os.path.join(tmp, f"seg_{i:04d}.wav")
            synthesize_clone(text, clone_refs[sp], target_lang, raw)
            speeds[i] = fit_segment(raw, seg["end"] - seg["start"], wav,
                                    avails[i])
            tts_wavs[i] = wav
            p(f"Clonage {i + 1}/{len(segments)} (locuteur {sp + 1})",
              0.4 + 0.4 * (i + 1) / len(segments))
    else:
        # Voix neurales : TTS en parallèle puis calage en parallèle
        jobs, fit_jobs = [], []
        for i, seg in enumerate(segments):
            text = (seg.get("translation") or "").strip()
            if not text:
                continue
            sp = seg.get("speaker", 0)
            voice = speaker_voices.get(sp, default_voice)
            raw = os.path.join(tmp, f"tts_{i:04d}.mp3")
            jobs.append((text, voice, raw))
            fit_jobs.append((i, raw, seg["end"] - seg["start"],
                             os.path.join(tmp, f"seg_{i:04d}.wav"),
                             avails[i]))
        n = max(len(jobs), 1)
        synthesize_batch(
            jobs,
            progress=lambda d: p(f"Synthèse vocale {d}/{len(jobs)}",
                                 0.4 + 0.3 * d / n))
        fit_speeds = fit_segments_parallel(
            [(r, slot, w, a) for _, r, slot, w, a in fit_jobs],
            progress=lambda d: p(f"Calage temporel {d}/{len(fit_jobs)}",
                                 0.7 + 0.1 * d / n))
        for (i, raw, slot, wav, _a), spd in zip(fit_jobs, fit_speeds):
            tts_wavs[i] = wav
            speeds[i] = spd

    p("Assemblage de la piste doublée…", 0.85)
    build_dub_track(segments, tts_wavs, state["total"], dub_wav)

    p("Mixage et encodage de la vidéo finale…", 0.92)
    mix_wav = os.path.join(tmp, "mix.wav")
    mix_duck(state["src_wav"], segments, tts_wavs, dub_wav, mix_wav,
             original_volume)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = output_dir or default_output_dir(video_path)
    tag = "CHTBOX_" if clone_voices else ""
    out_path = os.path.join(out_dir, f"{base}_{tag}{target_lang}.mp4")
    mux(video_path, mix_wav, out_path, original_volume)

    warnings = translation_warnings(segments, state["total"],
                                    speeds=speeds, threshold=1.5)
    p("Terminé.", 1.0)
    return out_path, warnings


def original_excerpt(state, speaker, max_seconds=20.0):
    """Extrait les premiers ~max_seconds de parole du locuteur `speaker`
    depuis l'audio original. Retourne le chemin du wav."""
    segs = [s for s in state["segments"] if s.get("speaker", 0) == speaker]
    with wave.open(state["src_wav"], "rb") as w:
        audio = w.readframes(w.getnframes())

    def frames(s, e):
        return audio[int(s * SAMPLE_RATE) * 2:int(e * SAMPLE_RATE) * 2]

    buf, acc = bytearray(), 0.0
    for s in segs:
        buf += frames(s["start"], s["end"])
        buf += b"\x00" * int(0.25 * SAMPLE_RATE) * 2   # petite pause
        acc += s["end"] - s["start"]
        if acc >= max_seconds:
            break

    buf = buf[:int(max_seconds * SAMPLE_RATE) * 2]
    out = os.path.join(state["tmp"], f"original_{speaker}.wav")
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(buf))
    return out


def preview_voice(state, speaker, voice, target_lang, max_seconds=30.0,
                  progress=None):
    """Synthétise un extrait (~max_seconds) de la voix `voice` sur le texte
    traduit du locuteur `speaker`. Retourne le chemin du wav."""
    tmp = state["tmp"]
    segs = [s for s in state["segments"] if s.get("speaker", 0) == speaker]
    chosen, acc = [], 0.0
    for s in segs:
        chosen.append(s)
        acc += s["end"] - s["start"]
        if acc >= max_seconds:
            break

    src = state["src_lang"] if state["src_lang"] != "auto" else "auto"
    need = [s for s in chosen
            if "translation" not in s or s.get("trans_lang") != target_lang]
    if need:
        translate_segments(need, src, target_lang)

    # TTS en parallèle puis conversion wav
    tts_jobs = [(s["translation"], voice,
                 os.path.join(tmp, f"prev_{speaker}_{i}.mp3"))
                for i, s in enumerate(chosen)]
    synthesize_batch(tts_jobs)
    buf = bytearray()
    for i, (t, v, mp3) in enumerate(tts_jobs):
        wav = os.path.join(tmp, f"prev_{speaker}_{i}.wav")
        run([FFMPEG, "-y", "-i", mp3, "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-c:a", "pcm_s16le", wav])
        buf += read_wav_frames(wav)
        buf += b"\x00" * int(0.25 * SAMPLE_RATE) * 2   # petite pause
        if progress:
            progress(f"Extrait locuteur {speaker + 1} : segment "
                     f"{i + 1}/{len(chosen)}", 0.5)
        if len(buf) >= int(max_seconds * SAMPLE_RATE) * 2:
            break

    buf = buf[:int(max_seconds * SAMPLE_RATE) * 2]
    out = os.path.join(tmp, f"preview_{speaker}.wav")
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(buf))
    return out


def dub_video(video_path, target_lang, voice=None, original_volume=0.1,
              diarize_speakers=True, num_speakers=0, progress=None):
    """Pipeline complet en un appel (usage programmatique / tests)."""
    state = analyze_video(video_path, diarize_speakers,
                          num_speakers=num_speakers, progress=progress)
    speakers = state["speakers"]
    if len(speakers) > 1:
        speaker_voices = auto_assign_voices(speakers, target_lang,
                                            state.get("genders"))
    else:
        speaker_voices = {speakers[0]:
                          voice or DEFAULT_VOICES.get(
                              target_lang, "en-US-AriaNeural")}
    out, _warns = render_video(video_path, state, target_lang,
                               speaker_voices,
                               original_volume=original_volume,
                               progress=progress)
    return out, state["src_lang"], state["segments"], speaker_voices
