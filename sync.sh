#!/bin/bash
# Synchronisation video-dubber -> GitHub (davidemm21/videodub)
# Déclenchée par launchd à chaque modification (WatchPaths).
LOG=/tmp/videodub_sync.log
echo "--- $(date) sync.sh start ---" >>"$LOG"

# Petite fenêtre de stabilisation : regroupe les sauvegardes rapides
sleep 15

cd "/Volumes/Mustang/DEV/video-dubber" 2>>"$LOG" || exit 1

git add -A 2>>"$LOG"
if git diff --cached --quiet 2>>"$LOG"; then
    echo "rien à synchroniser" >>"$LOG"
    exit 0
fi
git commit -m "sync: $(date '+%Y-%m-%d %H:%M')" >>"$LOG" 2>&1
git push origin main >>"$LOG" 2>&1
echo "push exit: $?" >>"$LOG"
