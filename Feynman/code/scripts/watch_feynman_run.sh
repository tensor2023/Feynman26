#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/Feynman26
log=outputs/feynman26.log
status=outputs/watchdog.status
checkpoint=outputs/lfm3-1.7b-grpo-b64-80g/latest_checkpointed_iteration.txt
offset=$(( $(stat -c %s "$log") + 1 ))

write_status() {
  printf '%s %s\n' "$(date '+%F %T %Z')" "$1" > "$status.tmp"
  mv "$status.tmp" "$status"
}

write_status "watching from byte $offset"
while true; do
  tracker=$(cat "$checkpoint" 2>/dev/null || echo 0)
  if [[ "$tracker" -ge 50 ]]; then
    write_status "complete tracker=$tracker"
    exit 0
  fi
  if ! tmux has-session -t feynman26 2>/dev/null; then
    write_status "failed training session missing tracker=$tracker"
    exit 2
  fi

  size=$(stat -c %s "$log")
  if [[ "$size" -ge "$offset" ]]; then
    new=$(tail -c +"$offset" "$log")
    offset=$((size + 1))
    if printf '%s' "$new" | grep -Eq 'CUDA out of memory|OutOfMemoryError|Error executing job|Insufficient Balance|Training failed'; then
      reason=$(printf '%s' "$new" | grep -E 'CUDA out of memory|OutOfMemoryError|Error executing job|Insufficient Balance|Training failed' | tail -n1)
      write_status "failed tracker=$tracker reason=$reason"
      tmux send-keys -t feynman26 C-c || true
      exit 3
    fi
  fi
  write_status "healthy tracker=$tracker log_bytes=$size"
  sleep 10
done
