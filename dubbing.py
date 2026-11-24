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

    try:
        requests.get("https://translate.googleapis.com", timeout=4)
        checks.append(("Traduction Google", True, "en ligne"))
    except Exception:
        checks.append(("Traduction Google", False,
                       "injoignable (pas de connexion ?)"))
    try:
        ok = bool(asyncio.run(edge_tts.list_voices()))
    except Exception:
        ok = False
    checks.append(("Voix neurales Edge-TTS", ok,
                   "en ligne" if ok else "injoignables"))

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
        audio, path_or_hf_repo=whisper_model_path(progress))
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result["segments"]
        if s["text"].strip()
    ]
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


def translate_segments(segments, source, target):
    import time
    sl = "auto" if source == "auto" else source
    for seg in segments:
        delay = 1.0
        for attempt in range(6):
            try:
                seg["translation"] = _gtranslate(seg["text"], sl, target)
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay *= 2
        time.sleep(0.3)  # rester sous la limite de requêtes/seconde
    return segments


async def _tts_one(text, voice, mp3_path):
    await edge_tts.Communicate(text, voice).save(mp3_path)


def synthesize(text, voice, mp3_path):
    asyncio.run(_tts_one(text, voice, mp3_path))


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


def fit_segment(mp3_path, slot_seconds, wav_out):
    """Convertit un mp3 TTS en wav calé sur la durée du segment."""
    dur = wav_duration_from_file(mp3_path)
    if slot_seconds <= 0:
        slot_seconds = dur
    factor = dur / slot_seconds
    filters = []
    if factor > 1.12:                      # trop long -> accélérer
        filters.append(_atempo_chain(min(factor, 4.0)))
    elif factor < 0.8:                     # beaucoup trop court -> ralentir un peu
        filters.append(_atempo_chain(max(dur / (slot_seconds * 0.9), 0.6)))
    run([FFMPEG, "-y", "-i", mp3_path, "-ac", "1", "-ar", str(SAMPLE_RATE)]
        + (["-af", ",".join(filters)] if filters else [])
        + ["-c:a", "pcm_s16le", wav_out])


def wav_duration_from_file(path):
    r = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def build_dub_track(segments, tts_wavs, total_seconds, out_wav):
    """Assemble les segments TTS sur une piste silence de la durée de la vidéo."""
    n = int(total_seconds * SAMPLE_RATE) + SAMPLE_RATE
    buf = bytearray(n * 2)
    for i, seg in enumerate(segments):
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


def mux(video, dub_wav, out_path, original_volume=0.1):
    if original_volume <= 0:
        run([FFMPEG, "-y", "-i", video, "-i", dub_wav,
             "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out_path])
    else:
        run([FFMPEG, "-y", "-i", video, "-i", dub_wav,
             "-filter_complex",
             f"[0:a]volume={original_volume}[bg];"
             f"[bg][1:a]amix=inputs=2:duration=first:normalize=0[aout]",
             "-map", "0:v", "-map", "[aout]",
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
                  progress=None):
    """Phase 1 : extraction audio + transcription Whisper (+ diarisation).
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
    p("Analyse terminée.", 0.3)
    return {"tmp": tmp, "src_wav": src_wav, "segments": segments,
            "src_lang": src_lang, "speakers": speakers, "genders": genders,
            "total": media_duration(video_path)}


def default_output_dir(video_path):
    """Dossier de sortie : à côté de la source si c'est un vrai chemin
    utilisateur ; sinon ~/Downloads (upload navigateur = temp Gradio)."""
    d = os.path.dirname(os.path.abspath(video_path))
    tmp_markers = ("/gradio", tempfile.gettempdir(), "/var/folders",
                   "/private/tmp", "/private/var")
    is_temp = any(m in d for m in tmp_markers)
    if not is_temp and os.access(d, os.W_OK):
        return d
    return os.path.expanduser("~/Downloads")


def render_video(video_path, state, target_lang, speaker_voices,
                 original_volume=0.1, output_dir=None, progress=None):
    """Phase 2 : traduction + synthèse par locuteur + mixage."""
    def p(msg, frac):
        if progress:
            progress(msg, frac)

    tmp, segments = state["tmp"], state["segments"]
    dub_wav = os.path.join(tmp, "dub.wav")
    default_voice = DEFAULT_VOICES.get(target_lang, "en-US-AriaNeural")

    p("Traduction…", 0.35)
    translate_segments(segments, state["src_lang"], target_lang)

    tts_wavs = []
    for i, seg in enumerate(segments):
        voice = speaker_voices.get(seg.get("speaker", 0), default_voice)
        mp3 = os.path.join(tmp, f"tts_{i:04d}.mp3")
        wav = os.path.join(tmp, f"seg_{i:04d}.wav")
        synthesize(seg["translation"], voice, mp3)
        fit_segment(mp3, seg["end"] - seg["start"], wav)
        tts_wavs.append(wav)
        p(f"Synthèse vocale {i + 1}/{len(segments)} (locuteur "
          f"{seg.get('speaker', 0) + 1} → {voice})",
          0.4 + 0.4 * (i + 1) / len(segments))

    p("Assemblage de la piste doublée…", 0.85)
    build_dub_track(segments, tts_wavs, state["total"], dub_wav)

    p("Mixage et encodage de la vidéo finale…", 0.92)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = output_dir or default_output_dir(video_path)
    out_path = os.path.join(out_dir, f"{base}_{target_lang}.mp4")
    mux(video_path, dub_wav, out_path, original_volume)

    p("Terminé.", 1.0)
    return out_path


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
    buf = bytearray()
    for i, s in enumerate(chosen):
        if "translation" not in s:
            s["translation"] = _gtranslate(s["text"], src, target_lang)
        mp3 = os.path.join(tmp, f"prev_{speaker}_{i}.mp3")
        wav = os.path.join(tmp, f"prev_{speaker}_{i}.wav")
        synthesize(s["translation"], voice, mp3)
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
    out = render_video(video_path, state, target_lang, speaker_voices,
                       original_volume, progress)
    return out, state["src_lang"], state["segments"], speaker_voices
