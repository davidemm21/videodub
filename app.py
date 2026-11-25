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
/* ---- look moderne, tons bleus ---- */
body, .gradio-container {
  font-family: "SF Pro Display", -apple-system, "Segoe UI", Inter,
               sans-serif !important;
}
.vd-header {
  display: flex; align-items: center; gap: 20px;
  padding: 20px 26px; margin-bottom: 14px;
  border-radius: 18px;
  background: linear-gradient(120deg, #0a2a6b 0%, #1155c9 55%,
                               #1e88e5 100%);
  box-shadow: 0 8px 24px rgba(17, 85, 201, .25);
}
.vd-header img {
  height: 72px; object-fit: contain;
  background: #ffffff; padding: 8px 14px; border-radius: 12px;
}
.vd-header h1 { margin: 0; color: #fff; font-size: 1.55rem;
                font-weight: 700; letter-spacing: .2px; }
.vd-header p  { margin: 4px 0 0; color: #dbe9ff; font-size: .92rem; }
.gradio-container .prose h3 { color: #1155c9; }
/* ---- statuts analyse ---- */
@keyframes vd-blink { 50% { opacity: 0.25; } }
.vd-blink { animation: vd-blink 1s ease-in-out infinite;
            color: #d97706; font-weight: 600; }
.vd-ready { color: #16a34a; font-weight: 600; }
/* ---- alerte traductions trop longues ---- */
.vd-warnbox {
  border: 1px solid #f59e0b; background: #fffbeb;
  border-radius: 12px; padding: 12px 16px; margin: 10px 0;
}
.vd-warnbox > b { color: #92400e; }
.vd-warnbox .vd-sub { font-size: .85rem; color: #a16207; margin: 4px 0 8px; }
.vd-dictbox { border-color: #2563eb !important; background: #eff6ff !important; }
.vd-dictbox > b { color: #1e40af !important; }
.vd-dictbox .vd-sub { color: #1d4ed8 !important; }
.vd-warn-seg {
  margin: 6px 0; padding: 8px 10px;
  background: #fff; border: 1px solid #fde68a; border-radius: 8px;
}
.vd-warn-seg .vd-time { font-size: 12px; color: #b45309; font-weight: 600; }
.vd-warn-seg .vd-orig { color: #64748b; font-size: 13px; margin: 4px 0 2px; }
.vd-warn-seg .vd-trad { font-size: 14px; color: #0f172a; }
.vd-warn-seg mark { background: #fde68a; padding: 1px 4px;
                    border-radius: 4px; }
/* ---- pied de page ---- */
body { padding-bottom: 46px; }
.vd-footer {
  position: fixed; bottom: 0; left: 0; right: 0; z-index: 50;
  text-align: center; padding: 9px 12px;
  background: linear-gradient(90deg, #0a2a6b, #1155c9, #0a2a6b);
  color: #cfe1ff; font-size: .8rem; letter-spacing: .4px;
}
"""

LOGO_B64 = None


def _logo_b64():
    global LOGO_B64
    if LOGO_B64 is None:
        import base64
        p = os.path.join(GIT_DIR, "assets", "logo_ap.png")
        with open(p, "rb") as f:
            LOGO_B64 = base64.b64encode(f.read()).decode()
    return LOGO_B64


THEME = gr.themes.Ocean(primary_hue="blue", secondary_hue="sky",
                        neutral_hue="slate")
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
    Générateur : yield ("status", msg, frac) pendant,
    puis ("result", value, 1.0)."""
    status, result, error = {"msg": "Démarrage…", "frac": 0.0}, {}, {}

    def report(msg, frac):
        status["msg"], status["frac"] = msg, frac
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
        cur = (status["msg"], status["frac"])
        if cur != last:
            last = cur
            yield ("status", status["msg"], status["frac"])
    if error:
        raise error["e"]
    yield ("result", result["value"], 1.0)


def _prog_bar(frac, width=18):
    """Barre de progression ASCII : ██████░░░░."""
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


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
    # + accordéon ; + 6 lecteurs ; + statut ; + alerte trad. longues
    hidden = ([gr.update(visible=False), gr.update(), gr.update(value=""),
               gr.update(value="")] * MAX_SPEAKERS
              + [gr.update(visible=False)]
              + [gr.update(visible=False)] * MAX_SPEAKERS
              + [busy] + _warn_empty() + _dict_empty())
    yield [_file_info(src) + "\n\n⏳ Analyse…", None] + hidden

    nspk = int(num_spk or 0)

    def work(report):
        saved = dubbing.load_analysis(src, multi, nspk)
        if saved:
            report("Analyse chargée depuis la sauvegarde.", 0.3)
            return saved
        st = dubbing.analyze_video(src, diarize_speakers=multi,
                                   num_speakers=nspk, target_lang=lang,
                                   progress=lambda m, f:
                                   report(m, min(f, 0.33)))
        dubbing.save_analysis(st, src, multi, nspk)
        return st

    try:
        for kind, payload, frac in _run_with_status(work, progress):
            if kind == "status":
                yield ([_file_info(src) + f"\n\n⏳ {_prog_bar(frac)} "
                        f"{int(frac * 100):3d} % — {payload}", None]
                       + hidden)
                continue

            state = payload
            speakers, segments = state["speakers"], state["segments"]
            n = len(speakers)
            if n > 1:
                voices = dubbing.auto_assign_voices(
                    speakers, lang, state.get("genders"))
            else:
                voices = {speakers[0]: dubbing.DEFAULT_VOICES.get(
                    lang, "en-US-AriaNeural")}

            choices = dubbing.list_voices(lang)
            speaker_segs = {sp: [s for s in segments
                                 if s.get("speaker") == sp]
                            for sp in speakers}
            lines = [_file_info(src), "",
                     f"**Langue détectée : {state['src_lang']}** — "
                     f"{len(segments)} segments, "
                     f"**{n} locuteur{'s' if n > 1 else ''}** détecté"
                     f"{'s' if n > 1 else ''}."]
            updates = []
            restored = 0
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
                    proposed = "\n\n".join(s.get("translation", "")
                                           for s in segs)
                    trad, was_restored = _trad_value_or_saved(
                        src, i, lang, proposed)
                    restored += was_restored
                    updates += [
                        gr.update(visible=True),  # le bloc complet
                        gr.update(visible=True, choices=choices,
                                  value=voices[sp],
                                  label=f"Locuteur {i + 1}"
                                        f"{f' — {gtxt}' if gtxt else ''}"
                                        f" — {len(segs)} segments, "
                                        f"{t:.0f}s — « {sample} »"),
                        gr.update(value="\n\n".join(s["text"]
                                                   for s in segs)),
                        gr.update(value=trad),
                    ]
                else:
                    updates += [gr.update(visible=False), gr.update(),
                                gr.update(), gr.update()]
            if state.get("restored"):
                lines.append("💾 Analyse rechargée depuis la sauvegarde "
                             f"`{os.path.basename(dubbing.analysis_file(src))}`"
                             " — instantané, aucun recalcul.")
            save_dir = dubbing.default_output_dir(src)
            lines.append("✏️ **Corrige les traductions** dans les zones de "
                         "droite — sauvegarde automatique à chaque frappe "
                         "puis clique « Générer la vidéo doublée ».")
            lines.append(f"📁 Tous les fichiers dans : `{save_dir}` "
                         "(analyse, traductions, vidéo doublée)")
            if restored:
                lines.append(f"💾 {restored} traduction(s) restaurée(s) "
                             f"depuis les fichiers de sauvegarde.")
            if n > MAX_SPEAKERS:
                lines.append(f"⚠️ {n - MAX_SPEAKERS} locuteurs "
                             f"supplémentaires utiliseront la voix du "
                             f"locuteur {MAX_SPEAKERS}.")
            yield (["\n".join(lines), state] + updates
                   + [gr.update(visible=True)]                # accordéon
                   + [gr.update(visible=False)] * MAX_SPEAKERS  # audios
                   + [gr.update(visible=False)]                # statut
                   + _warn_outputs(state, lang)                # alerte
                   + _dict_outputs(state, lang))               # glossaire
    except Exception as e:
        yield ([f"❌ Erreur pendant l'analyse : {e}", None] + hidden)
        return


def _collect_edits(speakers, trad_vals):
    """Textes de traduction édités par locuteur (zones non vides)."""
    return {sp: trad_vals[i] for i, sp in enumerate(speakers)
            if i < MAX_SPEAKERS
            and trad_vals[i] and trad_vals[i].strip()}


MAX_WARNS = 8


def _warn_empty():
    """Sorties « panneau masqué » : en-tête + map + MAX_WARNS blocs."""
    return ([gr.update(visible=False), []]
            + [gr.update(visible=False), gr.update(), gr.update()]
            * MAX_WARNS)


def _warn_head(items):
    """En-tête du panneau d'alerte (les passages sont éditables juste
    en dessous, un bloc par segment trop long)."""
    if not items:
        return gr.update(visible=False)
    extra = (f" (+{len(items) - MAX_WARNS} autre(s) — corrige d'abord "
             "ceux-ci)" if len(items) > MAX_WARNS else "")
    return gr.update(
        visible=True,
        value=('<div class="vd-warnbox">'
               f'<b>⚠️ {len(items)} passage(s) trop long(s) — la voix '
               f'sera accélérée{extra}</b>'
               '<div class="vd-sub">Corrige directement les traductions '
               'ci-dessous : chaque zone est synchronisée avec la grande '
               'zone « Traduction — locuteur N ». Une fois le passage '
               'assez court, son alerte disparaît.</div></div>'))


def _warn_outputs(state, lang_code, items=None):
    """Sorties du panneau d'alerte : en-tête + jusqu'à MAX_WARNS blocs
    éditables. `items` : avertissements déjà calculés (mesure exacte
    post-génération) ; sinon estimation sur les traductions du state."""
    if not state or not lang_code:
        return _warn_empty()
    if items is None:
        items = dubbing.translation_warnings(
            state["segments"], state["total"], lang_code)
    if not items:
        return _warn_empty()
    out = [_warn_head(items),
           [it["index"] for it in items[:MAX_WARNS]]]
    for j in range(MAX_WARNS):
        if j < len(items):
            it = items[j]
            t0, t1 = it["start"], it["end"]
            md = (f"**Segment {it['index'] + 1}** · locuteur "
                  f"{it['speaker'] + 1} · "
                  f"{int(t0 // 60)}:{int(t0 % 60):02d} → "
                  f"{int(t1 // 60)}:{int(t1 % 60):02d} · "
                  f"**×{it['speed']:.1f} requis**"
                  f"  \nOriginal : _{it['text'][:140]}_")
            out += [gr.update(visible=True),
                    gr.update(value=md),
                    gr.update(value=it["translation"],
                              label="Raccourcis cette traduction")]
        else:
            out += [gr.update(visible=False), gr.update(), gr.update()]
    return out


MAX_SUGGEST = 8


def _dict_empty():
    """Sorties « glossaire masqué » : en-tête + map + MAX_SUGGEST lignes."""
    return ([gr.update(visible=False), []]
            + [gr.update(visible=False), gr.update()] * MAX_SUGGEST)


def _dict_head(items):
    if not items:
        return gr.update(visible=False)
    extra = (f" (+{len(items) - MAX_SUGGEST} autre(s) — traite d'abord "
             "ceux-ci)" if len(items) > MAX_SUGGEST else "")
    return gr.update(
        visible=True,
        value=('<div class="vd-warnbox vd-dictbox">'
               f'<b>📖 {len(items)} remplacement(s) proposé(s) par ton '
               f'dictionnaire personnel{extra}</b>'
               '<div class="vd-sub">Mémorisés depuis tes corrections '
               'précédentes — confirme chaque remplacement avec '
               '« Appliquer », ou « Ignorer » pour ce passage.</div>'
               '</div>'))


def _dict_outputs(state, lang_code):
    """Sorties du panneau glossaire : en-tête + jusqu'à MAX_SUGGEST
    lignes « terme → remplacement » avec boutons Appliquer/Ignorer."""
    if not state or not lang_code:
        return _dict_empty()
    items = dubbing.glossary_suggestions(state, lang_code)
    if not items:
        return _dict_empty()
    out = [_dict_head(items),
           [(it["index"], it["src"], it["dst"])
            for it in items[:MAX_SUGGEST]]]
    for j in range(MAX_SUGGEST):
        if j < len(items):
            it = items[j]
            md = (f"**Segment {it['index'] + 1}** · locuteur "
                  f"{it['speaker'] + 1} · "
                  f"« {it['src']} » → « **{it['dst']}** »"
                  f"  \n_{it['translation'][:140]}_")
            out += [gr.update(visible=True), gr.update(value=md)]
        else:
            out += [gr.update(visible=False), gr.update()]
    return out


def _trad_value_or_saved(src, i, lang_code, proposed):
    """Valeur affichée : fichier de sauvegarde existant (édition
    précédente) ou la proposition, qui est alors persistée."""
    path = dubbing.translation_file(src, i, lang_code)
    if os.path.exists(path):
        try:
            saved = open(path, encoding="utf-8").read()
            if saved.strip():
                return saved, True
        except OSError:
            pass
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(proposed)
    except OSError:
        pass
    return proposed, False


def save_trad(i, video, local_path, lang_name, text):
    """Persiste le texte de traduction du locuteur i à chaque
    modification (sécurité contre perte au rafraîchissement)."""
    if not lang_name:
        return
    src = (local_path or "").strip() or video
    if not src or not os.path.exists(src):
        return
    try:
        with open(dubbing.translation_file(src, i, LANGS[lang_name]),
                  "w", encoding="utf-8") as f:
            f.write(text or "")
    except OSError:
        pass


def render(video, lang_name, keep, local_path, state, clone, elabs,
           *vals, progress=gr.Progress()):
    if not lang_name:
        raise gr.Error("Choisis d'abord la langue cible.")
    if not state:
        raise gr.Error("Lance d'abord l'analyse (étape 1).")
    if clone and not dubbing.chatterbox_available():
        raise gr.Error("Chatterbox n'est pas installé "
                       "(pip install chatterbox-tts).")
    if elabs and not dubbing.elevenlabs_available():
        raise gr.Error("ElevenLabs : SDK ou clé API manquant "
                       "(pip install elevenlabs + .env.local).")
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

    yield [None, "⏳ Génération en cours…"] + _warn_empty()
    out_dir = dubbing.default_output_dir(src)

    def work(report):
        return dubbing.render_video(src, state, lang, speaker_voices,
                                    original_volume=keep, output_dir=out_dir,
                                    edited_translations=edited,
                                    clone_voices=clone, elabs=elabs,
                                    progress=lambda m, f:
                                    report(m, 0.33 + 0.67 * f))

    try:
        for kind, payload, frac in _run_with_status(work, progress):
            if kind == "status":
                yield ([None, f"⏳ {_prog_bar(frac)} "
                            f"{int(frac * 100):3d} % — {payload}"]
                       + _warn_empty())
            else:
                out_path, warns = payload
                msg = f"✅ Vidéo sauvée dans : {out_path}"
                if warns:
                    msg += (f"\n\n⚠️ {len(warns)} passage(s) parlé(s) "
                            "trop vite — corrige-les dans le panneau "
                            "d'alerte à droite puis régénère.")
                yield [out_path, msg] + _warn_outputs(state, lang, warns)
    except Exception as e:
        yield ([None, f"❌ Erreur pendant la génération : {e}"]
               + _warn_empty())


with gr.Blocks(title="Video Dubber", css=CSS, theme=THEME) as demo:
    gr.HTML(f"""
      <div class="vd-header">
        <img src="data:image/png;base64,{_logo_b64()}"
             alt="Aviation et Pilote">
        <div>
          <h1>Doublage vidéo local — multi-locuteurs</h1>
          <p>Transcription Whisper (local) + diarisation (local) →
             traduction Azure → voix neurales Edge-TTS ou clonage
             Chatterbox (une voix par locuteur) → nouvelle piste audio.</p>
        </div>
      </div>""")

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
                             label="Original sous les voix (0 = muet)",
                             info="Hors parole, l'audio original reste à "
                                  "100 % — jingle et musique conservés.")
            clone = gr.Checkbox(
                value=False,
                label="Cloner les voix originales (Chatterbox)",
                info="Chaque locuteur est doublé par un clone de sa propre "
                     "voix — bien plus fidèle mais plus lent. Les menus "
                     "de voix neurales sont ignorés. "
                     "Fichier : nom_CHTBOX_langue.mp4")
            elabs = gr.Checkbox(
                value=False,
                label="Cloner les voix (ElevenLabs — cloud, rapide)",
                info="Clonage instantané via l'API ElevenLabs : ~3 s par "
                     "segment, très fidèle. Consomme des crédits "
                     "(~1/caractère). Fichier : nom_EL_langue.mp4")
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
                                  f"(éditable · 1 paragraphe = 1 segment)",
                            interactive=True, lines=6, max_lines=12,
                            info="Corrige librement ; sauvegarde "
                                 "automatique (dossier indiqué après "
                                 "l'analyse). Lignes vides ignorées.")
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
            warn_box = gr.HTML(visible=False)
            warn_map = gr.State([])
            warn_rows = []
            for _ in range(MAX_WARNS):
                with gr.Column(visible=False) as wc:
                    wmd = gr.Markdown()
                    wtxt = gr.Textbox(interactive=True, lines=2,
                                      max_lines=8,
                                      label="Traduction à raccourcir")
                warn_rows.append((wc, wmd, wtxt))
            warn_components = ([warn_box, warn_map]
                               + [c for r in warn_rows for c in r])
            dict_head = gr.HTML(visible=False)
            dict_map = gr.State([])
            dict_rows = []
            for _ in range(MAX_SUGGEST):
                with gr.Row(visible=False) as drow:
                    dmd = gr.Markdown()
                    with gr.Column(scale=0, min_width=140):
                        dok = gr.Button("✓ Appliquer", size="sm",
                                        variant="primary")
                        dno = gr.Button("✗ Ignorer", size="sm")
                dict_rows.append((drow, dmd, dok, dno))
            dict_components = ([dict_head, dict_map]
                               + [c for r in dict_rows for c in r[:2]])
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
        noop = ([gr.update()] * MAX_SPEAKERS
                + [gr.update()] * len(warn_components)
                + [gr.update()] * len(dict_components))
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
                proposed = "\n\n".join(s.get("translation", "")
                                       for s in segs)
                if state.get("src_path"):
                    val, _ = _trad_value_or_saved(
                        state["src_path"], i, code, proposed)
                else:
                    val = proposed
                outs.append(gr.update(value=val))
            else:
                outs.append(gr.update())
        return (outs + _warn_outputs(state, code)
                + _dict_outputs(state, code))

    state = gr.State(None)

    def reset():
        return ([gr.update(value="En attente d'une vidéo…"), None]
                + [gr.update(visible=False), gr.update(),
                   gr.update(value=""), gr.update(value="")] * MAX_SPEAKERS
                + [gr.update(visible=False)]            # accordéon
                + [gr.update(visible=False)] * MAX_SPEAKERS  # audios
                + [gr.update(value=MSG_WAIT, visible=True,
                            elem_classes="")]           # statut
                + _warn_empty()                          # alerte
                + _dict_empty())                         # glossaire

    demo.load(health_report, None, health)

    analyze_outputs = ([summary, state]
                       + [c for quad in zip(speaker_rows, voice_dds,
                                            orig_txts, trad_txts)
                          for c in quad]
                       + [acc_listen] + previews + [status_msg]
                       + warn_components + dict_components)
    gate_inputs = [in_video, lang, local_path] + voice_dds
    gate_outputs = [status_msg, btn_an] + voice_dds

    in_video.upload(reset, None, analyze_outputs).then(
        gate, gate_inputs, gate_outputs)
    in_video.clear(reset, None, analyze_outputs).then(
        gate, gate_inputs, gate_outputs)
    local_path.change(gate, gate_inputs, gate_outputs)
    lang.change(gate, gate_inputs, gate_outputs).then(
        retranslate, [state, lang],
        trad_txts + warn_components + dict_components)
    btn_an.click(analyze, [in_video, lang, multi, num_spk, local_path],
                 analyze_outputs)
    btn_listen.click(listen, [state, lang] + voice_dds + trad_txts,
                     previews)
    for i, btn_o in enumerate(orig_btns):
        btn_o.click(partial(play_original, i), [state], previews[i])
    for i, t_trad in enumerate(trad_txts):
        t_trad.input(partial(save_trad, i),
                     [in_video, local_path, lang, t_trad], None)

    def check_trads(state, lang_name, *trad_vals):
        """Recalcule l'alerte « traduction trop longue » après édition."""
        if not state or not lang_name:
            return _warn_empty() + _dict_empty()
        edited = _collect_edits(state["speakers"], trad_vals)
        if edited:
            dubbing.apply_edited_translations(
                state["segments"], edited, LANGS[lang_name])
        code = LANGS[lang_name]
        return (_warn_outputs(state, code)
                + _dict_outputs(state, code))

    for t_trad in trad_txts:
        t_trad.change(check_trads, [state, lang] + trad_txts,
                      warn_components + dict_components)

    def edit_warn(row_i, state, wmap, video, local_path, lang_name,
                  new_text, *trad_vals):
        """Correction saisie dans le panneau d'alerte : met à jour le
        segment concerné, la grande zone du locuteur, le fichier de
        sauvegarde, puis recalcule les alertes."""
        trad_outs = [gr.update()] * MAX_SPEAKERS
        if state and lang_name and row_i < len(wmap):
            code = LANGS[lang_name]
            seg = state["segments"][wmap[row_i]]
            if (new_text or "").strip():
                dubbing.learn_replacements(
                    seg.get("translation", ""), new_text.strip(), code)
                seg["translation"] = new_text.strip()
                seg["trans_lang"] = code
            sp = seg.get("speaker", 0)
            if sp in state["speakers"]:
                pos = state["speakers"].index(sp)
                if pos < MAX_SPEAKERS:
                    segs = [s for s in state["segments"]
                            if s.get("speaker") == sp]
                    full = "\n\n".join(s.get("translation", "")
                                       for s in segs)
                    trad_outs[pos] = gr.update(value=full)
                    save_trad(pos, video, local_path, lang_name, full)
        code = LANGS.get(lang_name) if lang_name else None
        return (trad_outs + _warn_outputs(state, code)
                + _dict_outputs(state, code))

    for j, (_wc, _wmd, wtxt) in enumerate(warn_rows):
        wtxt.change(partial(edit_warn, j),
                    [state, warn_map, in_video, local_path, lang, wtxt]
                    + trad_txts,
                    trad_txts + warn_components + dict_components)

    def apply_sugg(row_i, state, dmap, video, local_path, lang_name):
        """Remplacement confirmé dans le panneau glossaire : met à jour
        le segment, la zone du locuteur, la sauvegarde, puis recalcule
        alertes + suggestions."""
        trad_outs = [gr.update()] * MAX_SPEAKERS
        if state and lang_name and row_i < len(dmap):
            seg_i, src_t, dst_t = dmap[row_i]
            seg = state["segments"][seg_i]
            dubbing.apply_glossary(seg, src_t, dst_t)
            seg["trans_lang"] = LANGS[lang_name]
            sp = seg.get("speaker", 0)
            if sp in state["speakers"]:
                pos = state["speakers"].index(sp)
                if pos < MAX_SPEAKERS:
                    segs = [s for s in state["segments"]
                            if s.get("speaker") == sp]
                    full = "\n\n".join(s.get("translation", "")
                                       for s in segs)
                    trad_outs[pos] = gr.update(value=full)
                    save_trad(pos, video, local_path, lang_name, full)
        code = LANGS.get(lang_name) if lang_name else None
        return (trad_outs + _warn_outputs(state, code)
                + _dict_outputs(state, code))

    def skip_sugg(row_i, state, dmap, lang_name):
        """Remplacement refusé : mémorisé comme ignoré pour cette
        analyse, la ligne disparaît du panneau."""
        if state and row_i < len(dmap):
            seg_i, src_t, _ = dmap[row_i]
            state.setdefault("dict_ignored", []).append([seg_i, src_t])
        code = LANGS.get(lang_name) if lang_name else None
        return _dict_outputs(state, code)

    for j, (_drow, _dmd, dok, dno) in enumerate(dict_rows):
        dok.click(partial(apply_sugg, j),
                  [state, dict_map, in_video, local_path, lang],
                  trad_txts + warn_components + dict_components)
        dno.click(partial(skip_sugg, j), [state, dict_map, lang],
                  dict_components)
    def _voice_inter(c, e):
        return [gr.update(interactive=not (c or e))] * MAX_SPEAKERS

    clone.change(_voice_inter, [clone, elabs], voice_dds)
    elabs.change(_voice_inter, [clone, elabs], voice_dds)
    clone.change(lambda c: gr.update(value=False) if c else gr.update(),
                 clone, elabs)
    elabs.change(lambda e: gr.update(value=False) if e else gr.update(),
                 elabs, clone)
    btn_go.click(render, [in_video, lang, keep, local_path, state, clone,
                          elabs]
                 + voice_dds + trad_txts,
                 [out_video, info] + warn_components)

    gr.HTML('<div class="vd-footer">— Copyright Emmanuel S. Davidson — '
            'Licensed to Aviation et Pilote 2026 —</div>')

if __name__ == "__main__":
    # Synchro GitHub : pousse chaque modification après 30 s de stabilité
    threading.Thread(target=git_sync_watcher, daemon=True).start()
    demo.launch(inbrowser=True, max_file_size="20gb",
                allowed_paths=[GIT_DIR,
                               os.path.expanduser("~"),
                               "/Volumes"])
