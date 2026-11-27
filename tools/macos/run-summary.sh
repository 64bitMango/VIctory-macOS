#!/bin/sh
# Summarise a run log: frame rate over time, where the frame's time went,
# warnings, and how the run ended. Made for the 5-second lines the host logs
# (crates/victory-gpu/src/host.rs, copystat.rs, unwaited.rs and
# crates/victory-loader/src/amd_only.rs); see docs/benchmarking.md for what
# the numbers mean and how to compare runs fairly.
# Usage: run-summary.sh [log]   the newest victory.log under runs/ by default
#        run-summary.sh -         read the log from stdin
#        run-summary.sh --self-test
set -u
export LC_ALL=C

self_test() {
  local log out ok=1
  log=$(mktemp) || exit 1
  cat > "$log" <<'EOF'
2026-11-27T08:00:00.000000Z  INFO victory_gpu::host: 2.6 fps, 380.0 ms/frame (141.2 preparing: copies 30.1 ms 120.0 MB, textures 12.4 ms 80.0 MB; 158.0 on the GPU; 40.3 copying back 210.0 MB), 1200 draws/frame, dropped 0 without a host pipeline (0 being made) and 3 indirect; 8 root-view slots/frame
2026-11-27T08:00:05.000000Z  INFO victory_gpu::copystat:   preparing, all of it per frame: 141.2 ms = giving textures up 1.1 + heap syncs 31.4 + the sweep of idle copies 2.2 + heap texture uploads 30.9 (2048 viewed textures looked at, 64 asked for their host texture) + the lists' steps 40.2 (refresh 12.3, copy 4.4, work 8.1, GPU writes 3.3, textures 2.2, heaps' hulls 1.1, title memory 0.7; 5.0 of it in refresh batches) + state preludes 3.2 + the last refresh batch 2.1 + the rest 30.1
2026-11-27T08:00:05.000000Z  INFO victory_gpu::copystat:   heap sync time per frame: taking the heaps' changed pages 14.1 ms, asking whether pages are mapped 2.2 ms, making the slots 15.1 ms
2026-11-27T08:00:05.000000Z  WARN victory_loader::amd_only:   AMD-only instructions emulated per frame (an exception each): clzero 0.1 single lines and 0.0 whole loops, monitorx 1.2, mwaitx 1.2, rdpru 0.4, clwb 0.1 single and 0.0 whole loops, extrq/insertq 0.0, waits already replaced 0.1; 0.512 ms in their handler (the exception's own cost not counted); replaced so far: 12 clzero loops at load, 40 extrq/insertq at load, 3 wait instructions at their first exception
2026-11-27T08:00:10.000000Z  WARN victory_gpu::list: title shader refused; its work is dropped shader="compute aa" reason=translation: a thing
2026-11-27T08:00:10.000000Z  INFO victory_gpu::host: 2.8 fps, 360.0 ms/frame (140.0 preparing: copies 30.0 ms 120.0 MB, textures 12.0 ms 80.0 MB; 150.0 on the GPU; 40.0 copying back 200.0 MB), 1180 draws/frame, dropped 0 without a host pipeline (0 being made) and 3 indirect; 8 root-view slots/frame
wine: Unhandled exception 0xc0000005 in thread 41
EOF
  out=$(summarise "$log")
  rm -f "$log"
  chk() { printf '%s\n' "$out" | grep -q -- "$1" || { echo "self-test FAIL: missing '$1'"; ok=0; }; }
  chk '2.6 fps'; chk '2.8 fps'; chk '380.0 ms'
  chk 'median 2.6 fps'
  chk 'preparing, all of it per frame: 141.2 ms'
  chk 'heap sync time per frame'
  chk '12 clzero loops at load, 40 extrq/insertq'
  chk '1  title shader refused' 
  chk 'Unhandled exception 0xc0000005'
  chk 'giving textures up 1.1'
  [ "$ok" = 1 ] && echo "self-test ok" || { printf '%s\n' "$out"; exit 1; }
}

# The one summarising pass. The logger colours its lines even into a file, so
# the escapes are stripped first; everything after that matches on plain text,
# and nothing needs an interval expression (some awks read those differently).
summarise() {
  esc=$(printf '\033')
  sed "s/${esc}\\\\[[0-9;]*m//g" "$1" | awk -v file="$1" '
function norm(s) {
  gsub(/0x[0-9a-fA-F]+/, "H", s); gsub(/[0-9]+/, "N", s)
  return substr(s, 1, 160)
}
BEGIN {
  printf "== %s\n", file
}
{
  ts = $1; level = $2; target = $3
  line = $0
  sub(/^[^ ]+ +[A-Z]+ +[^ ]+: /, "", line)
  # The frame line, every 5 seconds.
  if (line ~ /^[0-9.]+ fps, [0-9.]+ ms\/frame \(/) {
    n++
    fps[n] = line + 0
    ms[n] = 0
    if (match(line, /[0-9.]+ ms\/frame/)) ms[n] = substr(line, RSTART, RLENGTH - 9) + 0
    draws[n] = 0
    if (match(line, /[0-9]+ draws\/frame/)) draws[n] = substr(line, RSTART, RLENGTH - 12) + 0
    printf "  %s  %5.1f fps %6.1f ms %6d draws\n", ts, fps[n], ms[n], draws[n]
  }
  # The periodic breakdowns: the last of each is the settled one.
  periodic = 0
  if (line ~ /preparing, all of it per frame/) { last["preparing"] = line; periodic = 1 }
  if (line ~ /heap sync time per frame/) { last["heap sync"] = line; periodic = 1 }
  if (line ~ /heap sync slots per frame/) { last["heap sync slots"] = line; periodic = 1 }
  if (line ~ /prepare waste per frame/) { last["prepare waste"] = line; periodic = 1 }
  if (line ~ /copying back per frame:/) { last["copying back"] = line; periodic = 1 }
  if (line ~ /lock waits per frame/) { last["lock waits"] = line; periodic = 1 }
  if (line ~ /slot locks per frame/) { last["slot locks"] = line; periodic = 1 }
  if (line ~ /AMD-only instructions emulated per frame/) { last["amd only"] = line; periodic = 1 }
  if (line ~ /host textures: /) { last["host textures"] = line; periodic = 1 }
  if (line ~ /video memory: /) { last["video memory"] = line; periodic = 1 }
  if (line ~ /^unwaited: /) { last["unwaited"] = line; periodic = 1 }
  # Warnings by template, the periodic lines left out: they are counted once
  # above, and a warning every 5 seconds is the log, not news.
  if (level == "WARN" && !periodic) warn[norm(line)]++
  if (level == "ERROR") err[norm(line)]++
  # What was still running at the end.
  tail[tailn++ % 5] = $0
  if ($0 ~ /[Ww]atchdog|[Kk]illed|[Tt]erminated|[Uu]nhandled exception|exit code|STATUS_[A-Z_]+/)
    ended[nlines++] = $0
  all[nlines_total++ % 2] = 0
}
END {
  if (n == 0) {
    print "no frame lines: not a run log?"
    exit
  }
  # Median fps over the frame lines.
  for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++)
    if (fps[j] < fps[i]) { t = fps[i]; fps[i] = fps[j]; fps[j] = t }
  printf "\n-- frame lines (every 5 s): %d of them, median %.1f fps, worst %.1f (%.1f ms), best %.1f\n",
    n, fps[int((n + 1) / 2)], fps[1], (fps[1] > 0 ? 1000 / fps[1] : 0), fps[n]

  print "\n-- the last of each periodic line (what a settled run logged)"
  m = split("preparing,heap sync,heap sync slots,prepare waste,copying back,lock waits,slot locks,amd only,host textures,video memory,unwaited", names, ",")
  for (k = 1; k <= m; k++) if (names[k] in last) print "  " last[names[k]]

  c = 0
  for (w in warn) c++
  if (c > 0) {
    print "\n-- warnings (by template)"
    for (w in warn) printf "%7d  %s\n", warn[w], w
  }
  c = 0
  for (e in err) c++
  if (c > 0) {
    print "\n-- errors (by template)"
    for (e in err) printf "%7d  %s\n", err[e], e
  }

  print "\n-- how it ended"
  if (nlines > 0) for (k = 0; k < nlines && k < 3; k++) print "  " ended[k]
  else for (k = (tailn - 3 < 0 ? 0 : tailn - 3); k < tailn; k++) print "  " tail[k % 5]
}' 2>/dev/null
}

case "${1:-}" in
  --self-test) self_test; exit ;;
  -h|--help) sed -n 2,8p "$0"; exit 0 ;;
esac
log=${1:-}
if [ -z "$log" ]; then
  root=$(cd "$(dirname "$0")/../.." && pwd)
  log=$(ls -t "$root"/runs/*/victory.log 2>/dev/null | head -1)
  [ -n "$log" ] || log=$(ls -t "$root"/*.log 2>/dev/null | head -1)
  [ -n "$log" ] || { echo "no run log found; pass one, or capture a run with: tools/macos/victory.sh run --execute <exe> 2>run.log" >&2; exit 2; }
fi
[ "$log" = "-" ] || [ -f "$log" ] || { echo "not a file: $log" >&2; exit 2; }
summarise "$log"
