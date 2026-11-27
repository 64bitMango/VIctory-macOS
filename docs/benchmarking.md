# Benchmarking: comparing runs fairly

`tools/macos/run-summary.sh [log]` prints what this page tells you to look
for from one run's log: the 5-second frame lines with their medians, the
last of each periodic breakdown (preparing, heap syncs, copying back, the
AMD-only instructions, video memory), warnings by template, and how the run
ended. Capture a run's log with `tools/macos/victory.sh run --execute <exe>
2>run.log`.

Comparing two runs by frame time alone misleads: after the opening, the camera's position can differ slightly between runs, and a heavier view can make a real improvement look like a regression (as happened in #20).

## 1. Match the scene and normalize by work

- **Check the last saved frames** of both runs to confirm they show the same view before trusting raw `fps` or `ms/frame` numbers.
- **Normalize by work** using the per-frame statistics lines (logged every 5 seconds from `Host` in `crates/victory-gpu/src/host.rs` and `crates/victory-gpu/src/copystat.rs`):
  - Compare `draws/frame` and `copies per frame` (`title copies`, `refreshes`, `submissions`) alongside `ms/frame`.
  - Compare costs **per draw** (e.g. `preparing` / `draws` or `GPU` / `draws`) and **per step** from `preparing, all of it per frame` (`refresh`, `copy`, `work`, `GPU writes`, `textures`, `heaps' hulls`, `title memory`) so a view with more draws or copies does not skew the comparison.

## 2. Alternate run order

Run comparisons in an **alternating ABBA order** (`off` / `on` / `on` / `off`, or `baseline` / `change` / `change` / `baseline`) to cancel out thermal throttling, background load, and cache warmth.

## 3. Diagnostics vs release builds and swap

- **Diagnostics vs release builds**: `tools/macos/victory.sh` builds `--release` by default, and builds with `--features diagnostics` when `VICTORY_DIAGNOSTICS=1`.
  - Release builds without `diagnostics` give true end-to-end frame times.
  - Diagnostics builds add extra instrumentation (per-list GPU timing, capture/dump paths, `video memory` / `views table` / `host textures` / `copying back` breakdowns) and force waited execution where diagnostics ask for it (see [docs/unwaited.md](unwaited.md)). Use diagnostics builds to attribute *where* time went, and release builds without `diagnostics` to measure final frame time.
- **Check swap and memory pressure (`memory.log`)**: Check `memory.log` (and, in diagnostics builds, the `video memory: ... process commit ... working set ...` and `hard faults` counters) to make sure neither run spilled into macOS swap. A run that hits swap or triggers heavy `host textures given up` evictions is not comparable to one that fits in memory.

## 4. Where each cost lives in the logs

The 5-second summary and `copystat` lines break down the frame into:

- **Frame & GPU time** (`crates/victory-gpu/src/host.rs`):
  - `<fps> fps, <ms> ms/frame (<cpu> preparing: copies <copy_ms> ms <copy_mb> MB, textures <texture_ms> ms <texture_mb> MB; <gpu> on the GPU; <back_ms> copying back <back_mb> MB), <draws> draws/frame, ...`
  - Under diagnostics, per-list GPU breakdown: `GPU timing: lists <ms> ms/frame of wall time, their commands <ms> ms/frame` followed by `GPU <ms> ms/frame in <n> commands/frame: ...`.
- **Preparing breakdown** (`crates/victory-gpu/src/copystat.rs`):
  - `preparing, all of it per frame: <total> ms = giving textures up <ms> + heap syncs <ms> + the sweep of idle copies <ms> + heap texture uploads <ms> (<n> viewed textures looked at, <m> asked for their host texture) + the lists' steps <ms> (refresh ..., copy ..., work ..., GPU writes ..., textures ..., heaps' hulls ..., title memory ...; ... of it in refresh batches) + state preludes <ms> + the last refresh batch <ms> + the rest <ms>`
- **Heap sync**:
  - Look at `heap syncs` in `preparing, all of it per frame` and `prepare waste per frame`, plus the dedicated lines `heap sync slots per frame: ...` and `heap sync time per frame: taking the heaps' changed pages <ms>, asking whether pages are mapped <ms>, making the slots <ms> ...`.
  - Lock contention during heap syncs and view writes appears in `lock waits per frame (...)` and `slot locks per frame (...)`.
- **Uploads**:
  - `heap texture uploads` in `preparing, all of it per frame` (and `textures <texture_ms> ms <texture_mb> MB` in the top-level frame line, plus `copies per frame` for title buffer copies/refreshes).
- **Copy-back**:
  - `<back_ms> copying back <back_mb> MB` in the top-level frame line.
  - Detailed breakdown in `copying back per frame: asking lists for misses <ms>, marked pages <ms>, read marks <ms>, declared ranges <ms>, lists' work after <ms>; ...` and `copying back <what> per frame: ...`.

## 5. Submission-level timing (`tools/overlap-ceiling.py`)

For submission-level timing and overlap analysis (measuring how much preparation, GPU execution, and copy-back could overlap across submissions), use `tools/overlap-ceiling.py`.
