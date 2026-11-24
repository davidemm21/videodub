"""Interface web locale pour le doublage vidéo multi-locuteurs.
Lancement : .venv/bin/python app.py  ->  http://127.0.0.1:7860"""

import os
import subprocess
import threading
import time
from functools import partial

import gradio as gr

import dubbing

GIT_DIR = os.path.dirname(os.path.abspath(__file__))
GIT_REMOTE = "origin"
GIT_BRANCH = "main"


def _git(*args):
    return subprocess.run(["git", "-C", GIT_DIR, *args],
                          capture_output=True, text=True)


def _dirty_mtime():
    """mtime le plus récent parmi les fichiers modifiés/non suivis."""
    r = _git("status", "--porcelain")
    mtimes = []
    for line in r.stdout.splitlines():
        f = line[3:].strip().strip('"')
        p = os.path.join(GIT_DIR, f)
        if os.path.isfile(p):
            mtimes.append(os.path.getmtime(p))
    return max(mtimes) if mtimes else None


def _sync_once():
    """Commit + push si des changements sont en attente."""
    _git("add", "-A")
    msg = "sync: " + time.strftime("%Y-%m-%d %H:%M")
    r = _git("commit", "-m", msg)
    if r.returncode != 0:
        return
    push = _git("push", GIT_REMOTE, GIT_BRANCH)
    status = "ok" if push.returncode == 0 else push.stderr[-200:]
    print(f"[gitsync] {msg} -> {status}", flush=True)


def git_sync_watcher(poll=15, quiet_window=30):
    """Pousse sur GitHub dès que le dépôt est stable depuis `quiet_window` s."""
    while True:
        time.sleep(poll)
        try:
            mtime = _dirty_mtime()
        except Exception:
            continue
        if mtime and time.time() - mtime >= quiet_window:
            _sync_once()

LANGS = {
    "Français": "fr", "Anglais": "en", "Espagnol": "es", "Allemand": "de",
    "Italien": "it", "Portugais": "pt", "Néerlandais": "nl", "Arabe": "ar",
    "Japonais": "ja", "Chinois": "zh", "Russe": "ru", "Coréen": "ko",
}
MAX_SPEAKERS = 6


def health_report():
    """Vérifications au démarrage -> markdown affiché en haut de page."""
    lines = []
    all_ok = True
    for name, ok, detail in dubbing.check_environment():
        lines.append(f"{'✅' if ok else '❌'} {name} "
                     f"<sub>{detail}</sub>")
        all_ok &= ok
    header = "### Environnement : prêt" if all_ok \
        else "### ⚠️ Environnement : problèmes détectés"
    return header + "\n\n" + "  \n".join(lines)


def refresh_voices(lang_name):
    code = LANGS[lang_name]
    voices = dubbing.list_voices(code)
    default = dubbing.DEFAULT_VOICES.get(code, voices[0])
    return gr.update(choices=voices,
                     value=default if default in voices else voices[0])


def _run_with_status(work, progress):
    """Exécute work(report) dans un thread en relayant son statut.
    Générateur : yield ("status", msg) pendant, puis ("result", value)."""
    status, result, error = {"msg": "Démarrage…"}, {}, {}

    def report(msg, frac):
        status["msg"] = msg
        progress(frac, desc=msg)

    def runner():
        try:
            result["value"] = work(report)
        except Exception as e:
            error["e"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    last = None
    while t.is_alive():
        t.join(0.25)
        if status["msg"] != last:
            last = status["msg"]
            yield ("status", last)
    if error:
        raise error["e"]
    yield ("result", result["value"])


def _file_info(video):
    size = os.path.getsize(video) / 1e6
    dur = dubbing.media_duration(video)
    return (f"📹 **{os.path.basename(video)}** — {size:.1f} Mo — "
            f"{int(dur // 60)}:{int(dur % 60):02d}")


def _resolve_source(video, local_path):
    """Chemin local saisi (prioritaire) ou fichier uploadé."""
    src = (local_path or "").strip() or video
    if not src:
        raise gr.Error("Choisis d'abord une vidéo ou indique son chemin.")
    if not os.path.exists(src):
        raise gr.Error(f"Fichier introuvable : {src}")
    return src


def analyze(video, lang_name, multi, num_spk, local_path,
            progress=gr.Progress()):
    src = _resolve_source(video, local_path)
    lang = LANGS[lang_name]
    # par locuteur : row + dropdown ; + accordéon ; + 6 lecteurs audio
    hidden = ([gr.update(visible=False), gr.update()] * MAX_SPEAKERS
              + [gr.update(visible=False)]
              + [gr.update(visible=False)] * MAX_SPEAKERS)
    yield [_file_info(src) + "\n\n⏳ Analyse…", None] + hidden

    def work(report):
        return dubbing.analyze_video(src, diarize_speakers=multi,
                                     num_speakers=int(num_spk or 0),
                                     progress=lambda m, f:
                                     report(m, min(f, 0.33)))

    for kind, payload in _run_with_status(work, progress):
        if kind == "status":
            yield [_file_info(src) + f"\n\n⏳ {payload}", None] + hidden
            continue

        state = payload
        speakers, segments = state["speakers"], state["segments"]
        n = len(speakers)
        if n > 1:
            voices = dubbing.auto_assign_voices(speakers, lang,
                                                state.get("genders"))
        else:
            voices = {speakers[0]: dubbing.DEFAULT_VOICES.get(
                lang, "en-US-AriaNeural")}

        choices = dubbing.list_voices(lang)
        speaker_segs = {sp: [s for s in segments if s.get("speaker") == sp]
                        for sp in speakers}
        lines = [_file_info(src), "",
                 f"**Langue détectée : {state['src_lang']}** — "
                 f"{len(segments)} segments, "
                 f"**{n} locuteur{'s' if n > 1 else ''}** détecté"
                 f"{'s' if n > 1 else ''}."]
        updates = []
        for i in range(MAX_SPEAKERS):
            if i < n:
                sp = speakers[i]
                segs = speaker_segs[sp]
                t = sum(s["end"] - s["start"] for s in segs)
                sample = segs[0]["text"][:80]
                g = state.get("genders", {}).get(sp)
                gtxt = {"Male": "voix grave (H)",
                        "Female": "voix aiguë (F)",
                        "Unknown": "genre ?"}.get(g, "")
                updates += [
                    gr.update(visible=True),   # la ligne complète
                    gr.update(visible=True, choices=choices, value=voices[sp],
                              label=f"Locuteur {i + 1}"
                                    f"{f' — {gtxt}' if gtxt else ''} — "
                                    f"{len(segs)} segments, {t:.0f}s — "
                                    f"« {sample} »"),
                ]
            else:
                updates += [gr.update(visible=False), gr.update()]
        if n > MAX_SPEAKERS:
            lines.append(f"⚠️ {n - MAX_SPEAKERS} locuteurs supplémentaires "
                         f"utiliseront la voix du locuteur {MAX_SPEAKERS}.")
        yield (["\n".join(lines), state] + updates
               + [gr.update(visible=True)]                     # accordéon
               + [gr.update(visible=False)] * MAX_SPEAKERS)    # audios


def render(video, lang_name, keep, local_path, state, *voice_dd,
           progress=gr.Progress()):
    if not state:
        raise gr.Error("Lance d'abord l'analyse (étape 1).")
    src = _resolve_source(video, local_path)
    lang = LANGS[lang_name]
    speakers = state["speakers"]
    speaker_voices = {}
    for i, sp in enumerate(speakers):
        if i < MAX_SPEAKERS and voice_dd[i]:
            speaker_voices[sp] = voice_dd[i]
        else:
            speaker_voices[sp] = speaker_voices.get(
                speakers[min(i, MAX_SPEAKERS - 1)],
                dubbing.DEFAULT_VOICES.get(lang, "en-US-AriaNeural"))

    yield None, "⏳ Génération en cours…"
    out_dir = dubbing.default_output_dir(src)

    def work(report):
        return dubbing.render_video(src, state, lang, speaker_voices,
                                    original_volume=keep, output_dir=out_dir,
                                    progress=lambda m, f:
                                    report(m, 0.33 + 0.67 * f))

    for kind, payload in _run_with_status(work, progress):
        if kind == "status":
            yield None, f"⏳ {payload}"
        else:
            yield payload, f"✅ Sauvée dans : {payload}"


with gr.Blocks(title="Video Dubber") as demo:
    gr.Markdown("# Doublage vidéo local — multi-locuteurs\n"
                "Transcription Whisper (local) + diarisation (local) → "
                "traduction Google → voix neurales Edge-TTS (une par "
                "locuteur) → nouvelle piste audio.")

    health = gr.Markdown("⏳ Vérification de l'environnement…")

    with gr.Row():
        with gr.Column():
            in_video = gr.Video(label="Vidéo source", sources=["upload"])
            local_path = gr.Textbox(
                label="…ou chemin local du fichier",
                placeholder="/Volumes/Mustang/Videos/ma_video.mp4",
                info="Si renseigné : utilisé à la place de l'upload et la "
                     "vidéo doublée sera enregistrée dans le même dossier.")
            lang = gr.Dropdown(list(LANGS), value="Français",
                               label="Langue cible")
            multi = gr.Checkbox(
                value=True, label="Détecter plusieurs locuteurs",
                info="Décoche pour forcer une seule voix (plus rapide)")
            num_spk = gr.Number(
                value=0, minimum=0, maximum=MAX_SPEAKERS, step=1,
                label="Nombre de locuteurs (0 = auto)",
                info="Force le découpage si l'auto-détection en rate — "
                     "ex. : 2 pour un dialogue à deux")
            keep = gr.Slider(0, 0.5, value=0.1, step=0.05,
                             label="Voix originales en fond (0 = supprimées)")
            btn_an = gr.Button("Analyser la vidéo", variant="secondary")
            gr.Markdown("*L'analyse démarre automatiquement après "
                        "l'upload.*")
            summary = gr.Markdown("En attente d'une vidéo…")

            speaker_rows, voice_dds, orig_btns = [], [], []
            for i in range(MAX_SPEAKERS):
                with gr.Row(visible=False) as row:
                    dd = gr.Dropdown(allow_custom_value=True,
                                     interactive=True, scale=4,
                                     label=f"Voix neurale locuteur {i + 1}")
                    btn_o = gr.Button("🔊 Original", scale=1, min_width=90,
                                      size="sm")
                speaker_rows.append(row)
                voice_dds.append(dd)
                orig_btns.append(btn_o)

            with gr.Accordion("Écouter les voix (~30 s de chaque)",
                              open=True, visible=False) as acc_listen:
                btn_listen = gr.Button("🎧 Écouter les voix neurales "
                                       "proposées")
                previews = [gr.Audio(visible=False, interactive=False,
                                     label=f"Locuteur {i + 1}")
                            for i in range(MAX_SPEAKERS)]
        with gr.Column():
            btn_go = gr.Button("Générer la vidéo doublée",
                               variant="primary")
            out_video = gr.Video(label="Vidéo doublée")
            info = gr.Textbox(label="Info", interactive=False)

    def listen(state, lang_name, *voice_dd, progress=gr.Progress()):
        if not state:
            raise gr.Error("Lance d'abord l'analyse (étape 1).")
        lang = LANGS[lang_name]
        outs = []
        for i in range(MAX_SPEAKERS):
            sp = state["speakers"][i] if i < len(state["speakers"]) else None
            if sp is None:
                outs.append(gr.update(visible=False))
                continue
            voice = voice_dd[i] or dubbing.DEFAULT_VOICES.get(
                lang, "en-US-AriaNeural")
            progress(i / MAX_SPEAKERS,
                     desc=f"Extrait locuteur {i + 1} ({voice})…")
            path = dubbing.preview_voice(
                state, sp, voice, lang,
                progress=lambda m, f: progress(f, desc=m))
            outs.append(gr.update(visible=True, value=path,
                                  label=f"Locuteur {i + 1} — {voice}"))
        return outs

    def play_original(i, state):
        """Joue un extrait de la voix originale du locuteur i."""
        if not state or i >= len(state["speakers"]):
            return gr.update(visible=False)
        path = dubbing.original_excerpt(state, state["speakers"][i])
        return gr.update(visible=True, value=path,
                         label=f"Locuteur {i + 1} — voix ORIGINALE")

    state = gr.State(None)

    def reset():
        return ([gr.update(value="En attente d'une vidéo…"), None]
                + [gr.update(visible=False), gr.update()] * MAX_SPEAKERS
                + [gr.update(visible=False)]            # accordéon
                + [gr.update(visible=False)] * MAX_SPEAKERS)  # audios

    demo.load(health_report, None, health)

    analyze_outputs = ([summary, state]
                       + [c for pair in zip(speaker_rows, voice_dds)
                          for c in pair]
                       + [acc_listen] + previews)

    # L'analyse démarre automatiquement dès que la vidéo est chargée
    in_video.upload(analyze, [in_video, lang, multi, num_spk, local_path],
                    analyze_outputs)
    in_video.clear(reset, None, analyze_outputs)
    btn_an.click(analyze, [in_video, lang, multi, num_spk, local_path],
                 analyze_outputs)
    btn_listen.click(listen, [state, lang] + voice_dds, previews)
    for i, btn_o in enumerate(orig_btns):
        btn_o.click(partial(play_original, i), [state], previews[i])
    btn_go.click(render, [in_video, lang, keep, local_path, state]
                 + voice_dds, [out_video, info])

if __name__ == "__main__":
    # Synchro GitHub : pousse chaque modification après 30 s de stabilité
    threading.Thread(target=git_sync_watcher, daemon=True).start()
    demo.launch(inbrowser=True, max_file_size="20gb")
