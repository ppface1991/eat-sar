#!/bin/bash
# Run in background: watches best.pt and syncs to fs every 5 min
SRC=/root/autodl-tmp/eat-sar-work/runs/train/sarship/weights/best.pt
DST_DIR=/autodl-fs/data/eat-sar/runs_backup/sarship_wip
mkdir -p $DST_DIR
while true; do
    if [ -f "$SRC" ]; then
        cp "$SRC" "$DST_DIR/best.pt"
        cp /root/autodl-tmp/eat-sar-work/runs/train/sarship/results.csv "$DST_DIR/results.csv" 2>/dev/null
        cp /root/autodl-tmp/eat-sar-work/runs/train/sarship/weights/last.pt "$DST_DIR/last.pt" 2>/dev/null
        echo "[$(date +%H:%M:%S)] Synced best.pt to fs"
    fi
    sleep 300
done
