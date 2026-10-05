#!/bin/bash
# Full SAR-Ship recovery pipeline: train + backup + inference + figure
# Expected total time: ~3.5h training + ~15min inference/plotting
set -e

cd /autodl-fs/data/eat-sar
export PYTHONPATH=/autodl-fs/data/eat-sar/src

echo "===== Starting watchdog (backs up best.pt every 5min) ====="
nohup bash /autodl-fs/data/eat-sar/watchdog_backup.sh \
    > /autodl-fs/data/eat-sar/watchdog.log 2>&1 &
WATCHDOG_PID=$!
echo "Watchdog PID: $WATCHDOG_PID"
echo $WATCHDOG_PID > /autodl-fs/data/eat-sar/watchdog.pid

echo "===== Starting SAR-Ship training in background ====="
nohup bash /autodl-fs/data/eat-sar/scripts/03_train/prepare_and_train_sarship.sh \
    > /autodl-fs/data/eat-sar/sarship_train_run2.log 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
echo $TRAIN_PID > /autodl-fs/data/eat-sar/sarship_train.pid

echo ""
echo "===== Both running in background ====="
echo "Watch training progress:"
echo "  tail -f /autodl-fs/data/eat-sar/sarship_train_run2.log"
echo "Watch watchdog:"
echo "  tail -f /autodl-fs/data/eat-sar/watchdog.log"
echo ""
echo "Expected completion: ~3.5 hours"
echo "After training finishes, run:"
echo "  bash /autodl-fs/data/eat-sar/run_sarship_qualitative.sh"
