#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
RUNS_ROOT="$ROOT/runs"
DFINE_OUTPUT="$ROOT/D-FINE/output"

[[ "$(readlink -f "$RUNS_ROOT")" == /root/autodl-tmp/runs ]]
[[ "$(readlink -f "$DFINE_OUTPUT")" == /root/autodl-tmp/D-FINE/output ]]

if ps -eo args | grep -E '[t]rain\.py|[t]rain_.*\.py' >/dev/null; then
  echo "ACTIVE_TRAINING_PROCESS_FOUND"
  ps -eo pid,args | grep -E '[t]rain\.py|[t]rain_.*\.py'
  exit 2
fi

echo "Deleting disposable benchmark/smoke/probe run directories:"
mapfile -t disposable < <(
  find "$RUNS_ROOT/crop_cls" "$RUNS_ROOT/dinov2_rtdetr" \
    -mindepth 1 -maxdepth 1 -type d \
    \( -name 'batch_bench_*' -o -name '*smoke*' -o \
       -name '*probe*' -o -name 'overfit_*' \) | sort
)
printf '%s\n' "${disposable[@]}"
for directory in "${disposable[@]}"; do
  case "$directory" in
    /root/autodl-tmp/runs/crop_cls/*|/root/autodl-tmp/runs/dinov2_rtdetr/*)
      rm -rf -- "$directory"
      ;;
    *)
      echo "Refusing unexpected path: $directory" >&2
      exit 3
      ;;
  esac
done

echo "Deleting redundant epoch/last weights where a best weight exists:"
mapfile -t best_directories < <(
  find "$RUNS_ROOT" -type f \
    \( -name 'best.pt' -o -name 'best_loss.pt' \) \
    -printf '%h\n' | sort -u
)
for directory in "${best_directories[@]}"; do
  case "$directory" in
    /root/autodl-tmp/runs/*)
      find "$directory" -maxdepth 1 -type f \
        \( -name 'epoch_*.pt' -o -name 'last.pt' \) -delete
      ;;
    *)
      echo "Refusing unexpected path: $directory" >&2
      exit 4
      ;;
  esac
done

echo "Deleting D-FINE checkpoints/last where best_stg weights exist:"
while IFS= read -r directory; do
  if find "$directory" -maxdepth 1 -type f \
      -name 'best_stg*.pth' -print -quit | grep -q .; then
    case "$directory" in
      /root/autodl-tmp/D-FINE/output/*)
        find "$directory" -maxdepth 1 -type f \
          \( -name 'checkpoint*.pth' -o -name 'last.pth' \) -delete
        ;;
      *)
        echo "Refusing unexpected path: $directory" >&2
        exit 5
        ;;
    esac
  fi
done < <(find "$DFINE_OUTPUT" -mindepth 1 -maxdepth 1 -type d)

sync
df -h "$ROOT"
du -x -h --max-depth=1 "$ROOT" | sort -h | tail -n 12
