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

CSS = """
@keyframes vd-blink { 50% { opacity: 0.25; } }
.vd-blink { animation: vd-blink 1s ease-in-out infinite;
            color: #d97706; font-weight: 600; }
.vd-ready { color: #16a34a; font-weight: 600; }
"""
MSG_WAIT = "En attente d'une vidéo et d'une langue cible…"
MSG_READY = "✅ Vidéo prête à être analysée"
MSG_BUSY = "⏳ Vidéo en cours d'analyse…"


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


def gate(video, lang_name, local_path, *voice_vals):
    """Verrouille le bouton Analyser + message de statut + voix compatibles."""
    src = (local_path or "").strip() or video
    has_src = bool(src) and os.path.exists(src)

    if not lang_name:
        msg = "🈯 Choisis la langue cible pour continuer" if has_src \
            else MSG_WAIT
        return ([gr.update(value=msg, visible=True, elem_classes=""),
                 gr.update(interactive=False)]
                + [gr.update()] * MAX_SPEAKERS)

    if not has_src:
        return ([gr.update(value="Charge une vidéo ou indique son chemin",
                          visible=True, elem_classes=""),
                 gr.update(interactive=False)]
                + [gr.update()] * MAX_SPEAKERS)

    # Vidéo + langue OK -> prêt ; voix = uniquement celles de la langue
    code = LANGS[lang_name]
    voices = dubbing.list_voices(code)
    fallback = dubbing.DEFAULT_VOICES.get(code, voices[0])
    dd_updates = [gr.update(
        choices=voices,
        value=v if v in voices else fallback) for v in voice_vals]
    return ([gr.update(value=MSG_READY, visible=True,
                      elem_classes="vd-ready"),
             gr.update(interactive=True)] + dd_updates)


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
    if not lang_name:
        raise gr.Error("Choisis d'abord la langue cible.")
    src = _resolve_source(video, local_path)
    lang = LANGS[lang_name]
    busy = gr.update(value=MSG_BUSY, visible=True,
                     elem_classes="vd-blink")
    # par locuteur : col + dropdown + texte orig + texte trad ;
    # + accordéon ; + 6 lecteurs ; + statut
    hidden = ([gr.update(visible=False), gr.update(), gr.update(value=""),
               gr.update(value="")] * MAX_SPEAKERS
              + [gr.update(visible=False)]
              + [gr.update(visible=False)] * MAX_SPEAKERS
              + [busy])
    yield [_file_info(src) + "\n\n⏳ Analyse…", None] + hidden

    def work(report):
        return dubbing.analyze_video(src, diarize_speakers=multi,
                                     num_speakers=int(num_spk or 0),
                                     target_lang=lang,
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
                    gr.update(visible=True),   # le bloc complet
                    gr.update(visible=True, choices=choices, value=voices[sp],
                              label=f"Locuteur {i + 1}"
                                    f"{f' — {gtxt}' if gtxt else ''} — "
                                    f"{len(segs)} segments, {t:.0f}s — "
                                    f"« {sample} »"),
                    gr.update(value="\n".join(s["text"] for s in segs)),
                    gr.update(value="\n".join(s.get("translation", "")
                                             for s in segs)),
                ]
            else:
                updates += [gr.update(visible=False), gr.update(),
                            gr.update(), gr.update()]
        lines.append("✏️ **Corrige les traductions** dans les zones de "
                     "droite (1 ligne = 1 segment) puis clique "
                     "« Générer la vidéo doublée ».")
        if n > MAX_SPEAKERS:
            lines.append(f"⚠️ {n - MAX_SPEAKERS} locuteurs supplémentaires "
                         f"utiliseront la voix du locuteur {MAX_SPEAKERS}.")
        yield (["\n".join(lines), state] + updates
               + [gr.update(visible=True)]                     # accordéon
               + [gr.update(visible=False)] * MAX_SPEAKERS      # audios
               + [gr.update(visible=False)])                   # statut


def _collect_edits(speakers, trad_vals):
    """Textes de traduction édités par locuteur (zones non vides)."""
    return {sp: trad_vals[i] for i, sp in enumerate(speakers)
            if i < MAX_SPEAKERS
            and trad_vals[i] and trad_vals[i].strip()}


def render(video, lang_name, keep, local_path, state, *vals,
           progress=gr.Progress()):
    if not lang_name:
        raise gr.Error("Choisis d'abord la langue cible.")
    if not state:
        raise gr.Error("Lance d'abord l'analyse (étape 1).")
    src = _resolve_source(video, local_path)
    lang = LANGS[lang_name]
    speakers = state["speakers"]
    voice_dd, trad_vals = vals[:MAX_SPEAKERS], vals[MAX_SPEAKERS:]
    edited = _collect_edits(speakers, trad_vals)
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
                                    edited_translations=edited,
                                    progress=lambda m, f:
                                    report(m, 0.33 + 0.67 * f))

    for kind, payload in _run_with_status(work, progress):
        if kind == "status":
            yield None, f"⏳ {payload}"
        else:
            yield payload, f"✅ Sauvée dans : {payload}"


with gr.Blocks(title="Video Dubber", css=CSS) as demo:
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
            lang = gr.Dropdown(
                list(LANGS), value=None,
                label="Langue cible *",
                info="Obligatoire — débloque le bouton Analyser")
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
            status_msg = gr.Markdown(MSG_WAIT)
            btn_an = gr.Button("Analyser la vidéo", variant="primary",
                               interactive=False)
            summary = gr.Markdown("En attente d'une vidéo…")

            speaker_rows, voice_dds, orig_btns = [], [], []
            orig_txts, trad_txts = [], []
            for i in range(MAX_SPEAKERS):
                with gr.Column(visible=False) as col:
                    with gr.Row():
                        dd = gr.Dropdown(
                            allow_custom_value=True,
                            interactive=True, scale=4,
                            label=f"Voix neurale locuteur {i + 1}")
                        btn_o = gr.Button("🔊 Original", scale=1,
                                          min_width=90, size="sm")
                    with gr.Row():
                        t_orig = gr.Textbox(
                            label=f"Texte original — locuteur {i + 1}",
                            interactive=False, lines=6, max_lines=12)
                        t_trad = gr.Textbox(
                            label=f"Traduction — locuteur {i + 1} "
                                  f"(éditable · 1 ligne = 1 segment)",
                            interactive=True, lines=6, max_lines=12,
                            info="Corrige librement ; la vidéo utilisera "
                                 "ce texte pour la voix.")
                speaker_rows.append(col)
                voice_dds.append(dd)
                orig_btns.append(btn_o)
                orig_txts.append(t_orig)
                trad_txts.append(t_trad)

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

    def listen(state, lang_name, *vals, progress=gr.Progress()):
        if not state:
            raise gr.Error("Lance d'abord l'analyse (étape 1).")
        lang = LANGS[lang_name]
        voice_dd, trad_vals = vals[:MAX_SPEAKERS], vals[MAX_SPEAKERS:]
        edited = _collect_edits(state["speakers"], trad_vals)
        if edited:
            dubbing.apply_edited_translations(
                state["segments"], edited, lang)
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

    def retranslate(state, lang_name):
        """Re-traduit les zones éditables si la langue a changé après
        l'analyse (les corrections obsolètes sont remplacées)."""
        noop = [gr.update()] * MAX_SPEAKERS
        if not state or not lang_name:
            return noop
        code = LANGS[lang_name]
        if state.get("trans_lang") == code:
            return noop
        dubbing.translate_segments(state["segments"], state["src_lang"],
                                   code)
        state["trans_lang"] = code
        outs = []
        for i in range(MAX_SPEAKERS):
            if i < len(state["speakers"]):
                sp = state["speakers"][i]
                segs = [s for s in state["segments"]
                        if s.get("speaker") == sp]
                outs.append(gr.update(
                    value="\n".join(s.get("translation", "")
                                    for s in segs)))
            else:
                outs.append(gr.update())
        return outs

    state = gr.State(None)

    def reset():
        return ([gr.update(value="En attente d'une vidéo…"), None]
                + [gr.update(visible=False), gr.update(),
                   gr.update(value=""), gr.update(value="")] * MAX_SPEAKERS
                + [gr.update(visible=False)]            # accordéon
                + [gr.update(visible=False)] * MAX_SPEAKERS  # audios
                + [gr.update(value=MSG_WAIT, visible=True,
                            elem_classes="")])          # statut

    demo.load(health_report, None, health)

    analyze_outputs = ([summary, state]
                       + [c for quad in zip(speaker_rows, voice_dds,
                                            orig_txts, trad_txts)
                          for c in quad]
                       + [acc_listen] + previews + [status_msg])
    gate_inputs = [in_video, lang, local_path] + voice_dds
    gate_outputs = [status_msg, btn_an] + voice_dds

    in_video.upload(reset, None, analyze_outputs).then(
        gate, gate_inputs, gate_outputs)
    in_video.clear(reset, None, analyze_outputs).then(
        gate, gate_inputs, gate_outputs)
    local_path.change(gate, gate_inputs, gate_outputs)
    lang.change(gate, gate_inputs, gate_outputs).then(
        retranslate, [state, lang], trad_txts)
    btn_an.click(analyze, [in_video, lang, multi, num_spk, local_path],
                 analyze_outputs)
    btn_listen.click(listen, [state, lang] + voice_dds + trad_txts,
                     previews)
    for i, btn_o in enumerate(orig_btns):
        btn_o.click(partial(play_original, i), [state], previews[i])
    btn_go.click(render, [in_video, lang, keep, local_path, state]
                 + voice_dds + trad_txts, [out_video, info])

if __name__ == "__main__":
    # Synchro GitHub : pousse chaque modification après 30 s de stabilité
    threading.Thread(target=git_sync_watcher, daemon=True).start()
    demo.launch(inbrowser=True, max_file_size="20gb")
