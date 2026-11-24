# Video Dubber

Application locale de doublage vidéo multi-locuteurs :
- Transcription : Whisper optimisé Apple Silicon (mlx-whisper), 100% local
- Diarisation : sherpa-onnx (segmentation pyannote + embeddings wespeaker),
  100% local
- Traduction : Google Translate (endpoint public)
- Voix : edge-tts (voix neurales Microsoft), une voix par locuteur avec
  estimation automatique du genre (hauteur F0)

## Lancer l'application

**Depuis le Finder / Launchpad** : `~/Applications/VideoDubber.app`
(démarre le serveur et ouvre http://127.0.0.1:7860)

**Arrêter** : `~/Applications/VideoDubber Stop.app`

**Depuis le terminal** :

    cd /Volumes/Mustang/DEV/video-dubber
    .venv/bin/python app.py

Logs : `/tmp/videodubber.log`

## Usage

1. Déposer une vidéo -> l'analyse démarre automatiquement
   (transcription + diarisation).
2. Une voix est proposée par locuteur détecté (modifiable).
3. « Générer » produit la vidéo doublée.

## Modèles

Stockés dans `/Volumes/Mustang/DEV/modelsAI/video` (surchargeable via la
variable d'environnement `VIDEO_DUBBER_MODELS`) :

- `whisper-large-v3-turbo/` : modèle Whisper MLX (~1,5 Go)
- `sherpa-onnx-pyannote-segmentation-3-0/` : segmentation locuteurs
- `wespeaker_en_voxceleb_resnet34_LM.onnx` : embeddings de locuteurs

## Notes

- Traduction et voix nécessitent une connexion ; transcription et
  diarisation sont hors-ligne.
- Fichiers temporaires dans /tmp/dub_*.
- Le projet `.app` est un lanceur : le dossier
  `/Volumes/Mustang/DEV/video-dubber` (venv inclus) doit rester en place.
