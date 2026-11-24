#!/bin/bash
# Synchronisation automatique video-dubber -> GitHub (davidemm21/videodub)
# Appelée par launchd toutes les 15 min ; ne fait rien s'il n'y a pas de changement.
cd "/Volumes/Mustang/DEV/video-dubber" || exit 1
LOG=/tmp/videodub_sync.log

git add -A
if git diff --cached --quiet; then
    exit 0   # rien à synchroniser
fi
{
    echo "=== $(date) ==="
    git commit -m "sync: $(date '+%Y-%m-%d %H:%M')"
    git push origin main
} >>"$LOG" 2>&1
