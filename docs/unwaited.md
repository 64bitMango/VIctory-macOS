# Unwaited submissions (design, tier one)

Branch `perf/unwaited`, from `mac/upstream-next` 95ee7d6b. This is the design
written before any code. Everything it describes is behind one switch,
`VICTORY_UNWAITED=1`, off by default.

The copy-back changed again after this base (8d5647cc on
`perf/copy-back-threads`, being merged). The branch takes that merge before
any code is written: the settle copies `Host::execute`'s sequence after its
wait, and must copy the merged one.

Where a statement was checked by reading our code it says so, or names the
function. Where it is an estimate or an inference it says "inferred".

## 1. The short version

- Today `Host::execute` holds the submission lock from the first refresh to the
  last copy-back, and waits for the host GPU in the middle. Nothing else of ours
  runs while a submission is on the GPU.
- Tier one keeps exactly one submission in the air. After it is handed to the
  host queue the call returns. It is **settled** (waited for, then everything
  `Host::execute` does after its wait, unchanged and in the same order) before
  anything that could see its results or disturb its inputs: the next host
  submission, a present, a fence event, a page mapping, the making of a view
  heap, or a 1 ms poll that finds the GPU done.
- Nearly every title submission can be launched this way, including the ones
  with bottom-of-pipe writes, shader writes, after-work and title-memory
  programs, because what they owe is paid at the settle, never earlier.
- The honest part: **launching unwaited is not the same as overlapping.** Read
  against our code, no stage of the next submission's preparation is disjoint
  from the one in the air (section 7). Under the strict rules of tier one the
  overlap is close to nothing. The frame only gets shorter when one of three
  named rules is relaxed, each with its own risk, switch and checker rule.
  Tier one measures, per pair of submissions, how much each relaxation would
  have overlapped, so that the choice is made on numbers from one safe run.
- The crash of the second attempt is most likely **not** the fork's fault (e)
  and not caused by unwaited submissions at all (section 9).

## 2. Terms

- **Flight**: a host submission handed to the host queue and not yet settled.
  Tier one has at most one.
- **Launch**: `ExecuteCommandLists`, `Signal` of the host fence, the flight
  recorded, return without waiting.
- **Settle**: wait until the host fence reaches the flight's value, then the
  waited path's work after its wait, in its order: missed checks, withholding,
  marked pages back, read views made, declared ranges back, after-work,
  bottom-of-pipe writes, flights of copies ended, retired objects released.
  Then the queue Signals kept behind the flight, in order. Then its host lists
  are given back.
- **Landing**: the part of the settle that writes title memory.
- **Write footprint** of a flight: the ranges its lists declared
  (`Step::GpuWrite`, marked or not), rounded out to 64 KiB pages. **Unbounded**
  if a list writes through views it takes from the heap itself, or runs a
  program that writes title memory at computed addresses (`Step::Memory` with
  its writes flag): then it is every page of every copy.
- **Read footprint**: every range its `Step::Refresh` steps name (exact), every
  hull its `Step::Heap` steps reach (an over-estimate: 3.5 GB in the world),
  every copy if it has a `Step::Memory`; the staging buffers its upload list
  copies from; every slot of the view heaps its lists bind.
- **The console rule**: on the console `ExecuteCommandLists` returns before
  the GPU starts, and the GPU reads title memory and descriptor heaps when it
  executes. So a title must not change a byte or a slot that a submitted list
  reads until something tells it the list ran. Tier one does not rely on it.

## 3. Invariants

Each is enforced by construction and verified by the checker (section 10).

- **I1 One flight.** A launch finds no flight. Settles happen in launch order.
- **I2 Never earlier.** Nothing a flight owes is visible before its settle: no
  byte its shaders wrote reaches title memory, no bottom-of-pipe word is
  written, no kept Signal moves a fence or sets an event, no after-work runs.
  Later is always allowed (the console's GPU is not instant either).
- **I3 Host copies stand still.** While a flight is in the air no thread of
  ours writes a byte of any host copy, takes a snapshot, arms a page, or takes
  (guards and marks clean) a tracker page. So what the flight's shaders read is
  what its own preparation left, and at landing a host byte that differs from
  the snapshot was written by the GPU and by nothing else. This is the fork's
  (e) made absolute: tier one does not refresh anything beside a flight.
- **I4 Queue order is CPU order.** Steps that we run on the CPU but that are
  ordered with GPU work on the console run only after the flight is settled:
  `Step::Copy`, `Step::Run`, `Step::Fill`, snapshots, page tables, and
  `CopyPageMappingsBatchX`.
- **I5 Landing is the waited path's.** The settle runs the same statements as
  `Host::execute` after its wait, on the same state, with the submission lock
  held. Between launch and settle that state changes only as it changes during
  a waited submission's GPU wait: by the title's own threads.
- **I6 Nothing a flight names goes away.** Its host lists are held (so a title
  list reset meanwhile gets a fresh host list, `list::restart`), and with them
  every copy, texture, pipeline, heap and upload buffer they keep. The shared
  upload list is not reset, nothing retired is released, no texture is given
  up, no idle copy is released, until the settle.
- **I7 The host is quiet when it must be.** A view heap is made and filled
  only with no flight (`Host::quiet`, VIctory-macOS #31). Our own submissions
  settle the flight first.
- **I8 A missed event cannot strand it.** Whether the flight is done is asked
  of the host fence's completed value, never of the event alone, by the settle
  and by the poll.
- **I9 Off is today.** With the switch off no flight ever exists. The waited
  path is not restructured. The hooks added to shared functions are single
  tests of one flag that is never set.

## 4. Tier one: the mechanism

### 4.1 Launch

`queue::pump` gives a piece to the host through a new function beside
`Host::execute`, `Host::execute_unwaited(&[Arc<HostList>])`, when the switch
is on. Under the submission lock it:

1. settles the flight in the air, if any;
2. prepares exactly as `Host::execute` does (eviction, heap sync, idle sweep,
   uploads, steps, preludes, final flush): the same calls in the same order,
   a copy of that code, not a refactoring of it;
3. calls `ExecuteCommandLists` and `Signal`;
4. records the flight: fence value, the lists, what the waited path keeps
   until the run is over (batch, heaps, textures), its footprints, its pending
   bottom-of-pipe writes, the launch time;
5. returns.

Waited instead, by the waited code, when: diagnostics time each list or trace
the frame (`VICTORY_GPU_TIMING`, listed capture frames); the host refused the
fence signal; the checker asks for it. Our own submissions (present, ambient
and twin passes, read-backs, captures) always use `Host::execute`.

### 4.2 Settle

One function, `Host::settle_flight(by)`, called with the submission lock held:
`wait_for(value)` (which wakes every second and asks the fence, as today), the
device-removed check, then the waited path's sequence (section 2), then the
kept Signals under the flight's own lock (`fence::set`, `fence::let_go`), then
the flight is cleared, then (after the lock) the lists are recycled.

What settles, and how it is counted:

| By | When |
|---|---|
| next title piece | `execute_unwaited`, first thing |
| ours | top of `Host::execute`, top of `Host::quiet` |
| present | `queue::present`, before the frame is shown |
| fence event | `SetEventOnCompletion` for a value not reached, and the title's CPU `Signal` |
| page mapping | `CopyPageMappingsBatchX` |
| pump | a queue head that waits on a word the flight will write with a timestamp, or that is blocked for any reason while a flight is in the air |
| poll | a 1 ms thread, parked while there is no flight, asks the fence's completed value and settles when it is reached and both locks are free; also `GetCompletedValue` of a title fence |

Lock order is the pump's: the queues' held-work lock, then the submission
lock. The poll takes both with `try_lock` in that order and pumps after it
settled, because a landed word may release held work, as today's timer does.

### 4.3 Bottom-of-pipe words, fences, held work

This is what kept 13 of 19 submissions waiting in the second attempt.

- **Words.** A flight's `deferred.writes` are made at its settle. The pump's
  `read` gets an overlay: a wait on an address that a pending
  `Written::Value` covers is tested against the value that will be written.
  If it holds, the dependent segment is "ready" and goes to
  `execute_unwaited`, which settles the flight before it prepares anything.
  So the word is in memory before the dependent list is prepared, and the
  list is on the GPU behind its writer. This is the fork's (b), and with one
  flight it needs no dependency graph. Timestamps cannot be predicted: such a
  wait settles the flight first and is then read from memory.
- **Signals.** A queue Signal reached in the pump while a flight is in the
  air is kept behind the flight, whichever queue made it: there is one host
  queue, so "behind everything submitted so far" is never earlier than the
  console's "behind this queue's work". A Signal on a queue with no held work
  is kept the same way instead of being set at once. Fences are set with
  plain stores, so kept Signals are applied in order, under the lock that
  also decides whether a new Signal is kept or made.
- **Queue waits on fences.** The pump's `fence` gets the same overlay: a
  `FenceWait` that a kept Signal will satisfy counts as met, and what it
  releases is prepared after the settle like everything else.
- **The safety valve and the 2 ms held-work timer** stay as they are.

### 4.4 What runs beside a flight in tier one

The title's own threads, as during a waited GPU run today. Of ours: nothing
that touches shared state. `execute_unwaited` settles before it prepares.

Tier one therefore changes when the title's submitting thread gets its call
back and when results land. It does not yet move our CPU work under the GPU's.

## 5. Every way a flight's results can be observed

| What | How it stays correct with N in the air |
|---|---|
| Title CPU reads of ranges N's shaders wrote | They hold the bytes from before N until the settle. That is a state the console also has (the GPU has not got there yet). Everything that tells the title N ran comes after the landing (I2). Not trapped, and need not be: a title that reads earlier races its own GPU on the console. Longer than today by at most the time until the next settling call or 1 ms after the GPU is done. |
| Title CPU writes into pages N's shaders write | As during a waited run: unmarked ranges are merged byte by byte against the snapshot taken at N's steps; armed marked pages keep the bytes from before the title's first write (`Tracked::saved`); marked pages nobody armed (heap-view writes) keep the title's bytes and lose the shaders' (counted "kept the title's", 4 a frame today). The window is longer, so this count is watched (section 12). |
| Bottom-of-pipe words | Written at the settle, after the landing, as in the waited order. Pump waits on them: 4.3. |
| Fence values and events | Kept Signals are applied at the settle. `SetEventOnCompletion` for a value not reached settles first, so an event the title then blocks on is either set already or waits for work not yet submitted, as today. |
| Queries, read-backs | After-work and `missed` checks run at the settle. Texture read-backs asked from title threads are submissions of ours: `Host::execute` settles first. |
| N+1's shaders reading what N wrote, through host copies | GPU order on one host queue. N+1 is prepared after N landed, so its refreshes, snapshots and `update_busy` see the state the waited order gives them. |
| Host copies refreshed while N reads or writes them | Does not happen (I3). The checker aborts on the first host-copy write beside a flight. |
| Queue-ordered CPU steps of N+1 (title copies, fills, runs) | After the settle (I4). This is the fork's fault (e) in our terms: their test had three submissions each copy a value to the same bytes and read it; all read the last. Our `Step::Copy` writes title memory on the CPU at preparation, so it must never run while an earlier flight could still read those bytes. |
| Page mappings | `CopyPageMappingsBatchX` is a queue operation on the console; we map at once. It settles the flight first. |
| Descriptor and heap state | Heap sync runs after the settle. Title threads write views into host heaps while N runs, as they do during a waited run today (about 1,300 locked and 4,000 unlocked slot writes a frame). A view heap is made only with no flight (I7). |
| Resources released or recycled | I6. `restart` leaves a held host list to its holder (verified: it tests the `Arc` count); what a title list owned goes with that host list (`list::leave`). `release_retired`, `evict_unused`, `idle_sweep` and the reset of the shared upload list (`bindless::uploads`, which says "closed by the last submission, which has run") run only after a settle. |
| Presents | `queue::present` pumps, settles, then presents. `Host::present` submits a list of ours, which settles too. |
| Idle and timer path | The 1 ms poll settles a flight whose GPU work is done when the title makes no call. It asks the fence, not the event (I8). |
| D3DMetal's missed fence events | `wait_for` already wakes each second and asks the fence. The poll never waits on the event (the event is only waited on under the submission lock, as today). |
| Device removal | Checked at the settle, with the flight's lists for the report. |
| Our own diagnostics | Capture dumps, GPU timing and traced frames use the waited path. |

## 6. What forces a wait, and the counts to expect

Inferred from the second attempt's counters and from `timing2` (today's tree,
world, about 22 host submissions and 19 to 26 title pieces a frame):

| | a frame |
|---|---|
| Title pieces launched unwaited | all of them, 19 to 26, outside timed or traced frames |
| Settled by the next title piece | most: the title's submitting thread makes its next call at once |
| Settled by ours (read-backs 8 to 11 a frame, present, passes) | about 10 |
| Settled by the pump for a timestamp wait or a blocked head | 0 to 8 (8 segments a frame wait on a word today, all released by a submission) |
| Settled by the poll | few in the world, most in menus |
| Waited by the waited code | 0 unless diagnostics ask |

Of the second attempt's reasons to wait (title-memory programs 3, heap-view
writes 1, after-work 1, bottom-of-pipe 13 with 10 for that alone), none
remains a reason not to launch. All of them remain reasons why nothing may be
refreshed beside the flight.

## 7. How much overlaps, and what rule change buys more

Measured today (`timing2`, per world frame): preparing 190 to 210 ms, of it
heap slots the title copied 93, texture uploads 32, refresh batches 44,
snapshots and declarations 24, title copies and fills 12, page tables 8; GPU
162 ms waited (243 with per-list timing); copy-back 45 to 55.

The frame is the serial work of one thread, so the title's thread has no work
of its own to do between two submissions. Overlap has to come from our
preparation of N+1 running under N. Stage by stage, read against the code:

| Stage of N+1 | ms | Why it is not disjoint from N |
|---|---|---|
| Heap sync (`descriptor::sync_heaps`) | 93 | Writes host slots of heaps N's lists bind; N's heap-indexing shaders may read any slot. |
| Uploads (`bindless::uploads`) | 32 | Resets the one shared upload list, which N is running; rewrites staging buffers N's upload list may be copying from. |
| Refresh batches | 44 | Write host copies. N's heap-indexing lists read every hull (about 10,000 copies), so nearly every refresh of N+1 is inside N's read footprint. Inside N's write footprint a refresh destroys what the GPU is writing. |
| Snapshots, arming | 24 | Must hold the bytes N left. |
| Title copies, fills, runs | 12 | Queue-ordered (I4); fills read title memory N has yet to land in. |
| Page tables | 8 | Read by N's programs if the heap is the same. |
| Eviction, idle sweep, retired | small | Release what N may use. |

So under the strict rules: **few**. All submissions launch unwaited, and about
none of the 160 ms overlaps. Tier one as built is the part that has to be
right first (late landing, late words and fences, order, lifetime), run in the
game with the checker, at no gain and no added exposure of the host copies.

What buys overlap, in the order I would try them. Each is a rule change with
a risk, not a tuning.

- **R1 Heap sync beside the flight** (93 ms a frame available; inferred 40 to
  70 overlapped). Rule: slots the title rewrote since N was prepared are not
  slots N reads (the console rule, for descriptors). Host behaviour: the same
  kind of write title threads already make during a waited run, at up to
  39,000 a frame instead of about 5,000. Risk: if either fails, a shader of N
  reads a null or newer view; where that shader writes title memory the
  result lands in it. Checkable only for slots N's draws read through tables
  (known); not for heap-indexed reads. D3DMetal has lost views under heavy
  descriptor traffic once (#31, during a heap fill).
- **R2 Uploads beside the flight** (32 available; inferred 10 to 25). Needs an
  upload list per flight, and the rule that a staging buffer N's upload list
  reads is not rewritten until the settle. Strictly sound with those two; it
  is a change to shared code, so it is not in tier one.
- **R3 Refresh beside the flight, outside its write footprint** (44
  available; inferred 15 to 30; nothing beside the 4 or 5 flights a frame
  whose write footprint is unbounded, which are probably the heavy ones).
  This is what the second attempt's `prewarm` did. Rule: the console rule for
  bytes. Checkable byte by byte for exact read ranges (compare before
  overwriting, abort if a byte N declared reading would change); not for
  hulls, where the title legitimately writes neighbouring bytes all the time.
- **R4 Tier two: two in the air**, N+1 launched before N has landed, with
  per-flight write ranges and landing behind the next flight, as the fork
  does. That is where the copy-back's 45 ms and the remaining wait would go.
  It needs per-flight snapshots and pending sets (ours are per copy), and the
  marks bitmap split by flight.

The ceiling for R1 to R3 together is the 160 ms; a realistic sum is 70 to 120
ms of a 395 ms frame (inferred: the per-pair distribution is not in any log).

**Measure before relaxing.** Tier one's counters record, for every pair,
`min(time N's settle stood waiting for the GPU, time N+1 then spent in stage
S)` for S = heap sync, uploads, refresh. One game run with the switch on gives
the exact overlap each of R1, R2, R3 would have produced, with none of them
enabled.

### 7.1 The other overlap: N's copy-back beside N+1's preparation

The author of today's copy-back (`back.rs`, `shadow.rs`) listed what must hold
for N's copy-back to run while the submitting thread prepares N+1. Each point
checked against the code, and what it means here:

| Condition | In the code | Tier one | To lift it |
|---|---|---|---|
| `pending`, the snapshot, `compared` and arming are per host copy, not per submission | True: fields of `Shadow` and of its tracker; `Shadow::settled` says "one submission runs at a time" | Holds: N is landed, and its copies settled, before any step of N+1 runs | Key them by flight (R4) |
| Both mark bitmaps are global and cleared at copy-back | True: `shadow::bitmap(false/true)`, one buffer each holding written pages, read marks and the atomics' lock words; `copy_back_marks` and `take_reads` take and clear them, with "the GPU is done with it" as their safety argument | Holds: they are read at N's settle, before N+1 is submitted | Read and clear them as soon as N's GPU work is done, before N+1 launches; or a bitmap per flight |
| Copy-back writes the host copy; N+1's refresh and the GPU read and write the same bytes | True: `back::put` gives the copy title memory's bytes where the shaders wrote nothing, `back::again` reads whole pages into it | Holds: the landing is done on the submitting thread before N+1's first refresh (I3, I5) | A wait per copy, or per page, between N's landing and N+1's refresh, declaration and launch |
| Unknown write sets force a full wait | True: marks are known only after the run; `Step::Memory` writers reach any copy | Holds: nothing is refreshed beside any flight | Stays, until the marks are read (then the pages are known) |
| A refresh over a title-written page wipes N's shader bytes not yet copied back | True: this is `update_busy`'s case, right only for lists not yet on the GPU | Holds: no refresh before the landing | Never liftable inside N's write footprint |
| N's words and fence completion follow N's copy-back; N finishes before N+1's | True: the order in `Host::execute` | Holds: the settle keeps that order, and I1 | Stays |
| The withholding state, `memory::DROPPED`, `evict_unused`, `idle_sweep` and list reset assume the last submission has finished | True: `SUBMISSION_MISSED` and `MISSED_IN` say "completions run one at a time"; the others say so in their comments; add the shared upload list of `bindless::uploads` and `release_retired` | Holds: all run at or after the settle (I6) | Each needs its own answer |

One more that the list does not name, from section 9: the copy-back also
relies on a tracker's "unchanged since taken" meaning "title memory equals
the copy". Any design that takes tracker pages while an earlier landing is
still to come breaks that for the pages taken.

So tier one satisfies every condition by doing the landing where the waited
path does it, and gains nothing from it. What moving the landing would be
worth, from the same author's figures of today (per frame: marked pages about
31 ms of the copy-back, declared ranges about 2 ms; 23 of about 27
submissions could rest their guards, about 4 could not because their shaders
write title memory at computed addresses): at most the 33 ms, and only on
copies and pages N+1 does not touch, which for the big heap means a wait per
page rather than per copy. The 4 unbounded writers a frame agree with the
second attempt's counters (3 title-memory programs and 1 or 2 heap-view
writers of 19): beside those nothing can be refreshed under any rule until
their marks have been read. This belongs to R4.

## 8. From the Windows fork (avatarmanz)

Taken:

- (a) Words and fence signals are told at the landing, in submission order,
  never earlier: I2, 4.2.
- (b) A queue wait that a pending flight's write will satisfy counts as met
  and the dependent work goes behind that flight: 4.3, simpler with one flight.
- (e) Each submission's shaders read the bytes it was prepared with: I3 and
  I4. Their fix gave each flight bytes of its own; ours forbids the write.
- The merge of title writes under shader writes against a snapshot: we have it
  (`back::merge`, by byte, with runs).
- Counting why and for how long, per frame.

Cannot transfer:

- **Bytes of its own per flight.** Their card copies have a CPU side and a
  card side, and a submission's bytes are sent up from memory of its own. Our
  host copies are CPU-mapped shared buffers that shaders read in place at a
  fixed address per title address; a second version of a page cannot be shown
  to one flight only. This is why tier one cannot refresh beside a flight.
- **Landing through a second, never-guarded view.** Only section-backed title
  memory has a second view here, and our write tracking is page guards whose
  faults cost tens of microseconds under Wine. Landing stays the batched
  copy-back of `back.rs`, with the submission lock held.
- **Trapping CPU reads of in-flight ranges (c).** Would need no-access guards
  on ranges the title reads constantly; not needed while every signal is late.
- **A preparing thread, joined parts.** Their gain from 103 to 75 ms came from
  preparing on a thread of their own while the title's thread recorded. Our
  frame is one thread's serial work with nothing beside it to free.
- Hash passes on the card and the kernel write watch: no equivalent here.

## 9. What the second attempt's crash most likely was

Read: `git diff 2e95f06 ba4e5968`, the two run logs
(`victory-ab-as2-on`, `victory-ab-as2-off`), and the clean run of the merge
base (`victory-ab-bis-merge`, 515 world frames).

Against (e):

1. **(e) as the fork met it could not happen in ba4e596.** Their fault was a
   later submission's queue-ordered copies reaching bytes an earlier flight
   still read. ba4e596's `prewarm` ran only refreshes early; the next
   submission's `Step::Copy`, `Step::Run` and fills ran after the finish
   (verified in `submit_held`). What `prewarm` did carry into copies a flight
   could read were the title's own CPU writes, on the strength of the console
   rule. That is a real exposure and tier one removes it, but:
2. **There was almost no flight to disturb.** The on-run's own counters: one
   unwaited submission a frame of 19, "the GPU ran 0.0 ms beside the title's
   thread" (0.1 in one interval), while 75 to 240 MB a frame were "refreshed
   meanwhile". The unwaited submission was a trivial one that was done before
   the early refreshes began.
3. **The off-run crashed too, with no flight at all.** Its log has no
   unwaited line, so `VICTORY_ASYNC_SUBMIT` was unset. With it unset I find no
   behavioural difference between ba4e596 and its parent: the copy-back moved
   into `Host::settle` statement for statement, `prewarm` and `finish` are
   never entered, `owes` is not asked. The one difference is the order in
   which the lock and the thread's mark are dropped (edition 2024 drops the
   tail expression's temporary first), which 4336f40a put back and which
   changes nothing another thread can see.

So the evidence does not support (e), and does not support "the off path
changed" either. What it does fit (inferred, not reproduced): a race the
waited path had at that base and no longer has. The crash was a pointer read
as `0xfc007c007c007c00`, half-float data where the title kept a pointer: a
whole stale page of a host copy written over newer title bytes. The copy-back
of marked pages could do exactly that. `copy_back_marked` read a page of title
memory, and only at the end of the whole pass (44 ms a frame, about 2,000
pages) called `Tracked::taken`, which clears "changed" and guards the page
unconditionally. A title write in between was in title memory, not in the
copy, with the tracker saying "unchanged". The next submission whose shaders
marked that page then took "unchanged" to mean "title memory equals the copy
from before" and wrote every differing byte of the copy into title memory.
The batched copy-back merged at 18:44 (7e23a8c3, in our base) reads such pages
again after guarding them (`back::again`) and describes this window in its
comment. The as2 runs were at 17:15 and 17:22, before it. Two crashes in two
runs of one binary against one clean run of its parent is weak statistics for
a race; it is still the only mechanism I can point to in the code that
produces this signature with the switch off.

Consequences for this work: the checker also verifies, in waited mode, that a
page a tracker calls unchanged equals the copy (section 10, C4). If the race
is still open somewhere (the one-by-one path, `copy_back_page`), a run with
the checker on and the switch off names it.

## 10. The checker (`VICTORY_UNWAITED_CHECK`, off by default)

As built. `=1` ends the run through `host::fail` with one line naming the
rule at the first violation; `=log` logs the first twenty and counts the
rest. It works with `VICTORY_UNWAITED` set or unset: C3 and C4 are about the
copy-back itself, which is the same code on both paths.

- **C1 Host copy touched beside a flight** (I3): `Shadow::copy` (every
  refresh) and `Shadow::writing` (every declaration, snapshot, arming) by a
  thread that is not settling the flight. Names the copy and the range.
- **C2 Order** (I1, I2): a launch with a flight in the air or with launches
  and settles out of step; a settle out of sequence; a landing before the
  host fence reached the flight's value.
- **C3 Copy-back over a title write** (`shadow/check.rs`): a byte the
  shaders changed in the copy that the title changed too, to something else
  than what is about to be written. The checker keeps no bytes of its own.
  For unmarked declared ranges "before" is the copy-back's own snapshot, and
  only pages the title wrote since the copy took them are read, and only
  where the snapshot is also what title memory held (the page was unchanged
  at the declaration, or a refresh under the range took it since). For
  marked pages "before" is what the tracker saved at the title's first write
  to the armed page; a page the title did not write has nothing to undo.
  Ranges withheld, pages already written at their declaration and what
  title-memory programs write at computed addresses count as unchecked.
- **C4 The trackers' word**: "unchanged since taken" must mean "title memory
  equals the copy", because the copy-back of a marked page writes every byte
  in which they differ. Checked where it is made so: every page a copy-back
  takes and reads again; with `VICTORY_UNWAITED_CHECK_REFRESH=1` every page a
  refresh takes. And where it is relied on: a page shaders marked in the last
  256 submissions is read and compared when it is declared again, once per
  taking and again every 64 submissions. A title write racing the check has
  set "changed" first (the page is guarded before it is read), so a
  difference with "unchanged" still standing is a write the copy never saw:
  the fault of section 9. Pages holding withheld writes are exempt.
- **C5 Predicted waits**: every value the pump read through the overlay is
  in memory when the flight has landed.
- **C6 Lifetime** (I6): the shared upload list reset, retired objects
  released, textures given up or idle copies released beside a flight.
- **C7 Queue order** (I4): a copy into title memory (`copy_mapped` with a
  title destination: the title's buffer copies) beside a flight.

Not built: a rule on a kept Signal lower than the fence's value (a title may
signal a lower value itself; the order is by construction and tested).

Cost, measured (`shadow::check::copies::benchmark_checker_release`, release,
Wine on the M5 Pro, frames of 20 submissions landing 1,920 marked pages and
declaring 55 MB of unmarked ranges): 25.6 ms a frame without the checker,
35.6 ms with, 147 MB compared a frame. It is one page read and compare per
page a copy-back takes, plus the pages the title wrote under declared ranges.
A first version that read every lately marked page at every declaration cost
280 ms a frame. With `VICTORY_UNWAITED_CHECK_REFRESH=1` everything refreshed
is read a second time (143 MB a frame in the world: about as much again,
inferred). The frame line reports the real figure ("unwaited checker: ms/frame,
MB compared").

## 11. Tests and deliberate breaks

Under Wine, through the real host device and queue. The "shaders" are real
GPU copies into the copies' buffers and into the marks' bitmap, behind up to
six 16 MB copies that keep the submission running.

- `host::unwaited::tests::both_paths_leave_the_same_bytes`: two submissions
  (a declared range, a marked page, a bottom-of-pipe word, two kept Signals;
  then a queue-ordered title copy, shader writes over the first one's range
  and bytes, two words) with a thread writing other bytes of the same pages
  throughout, through `Host::execute` on one buffer and as flights on
  another. Title memory, words, fence, order of landings compared; the copies
  must be current afterwards. Twelve rounds a run
  (`VICTORY_UNWAITED_TEST_ROUNDS`), every other one with the submission lock
  held and the checker on, the others through `Host::execute_unwaited` where
  other tests' submissions settle the flights.
- `queue::tests::work_waiting_for_what_a_flight_will_tell_runs_behind_it`:
  `pump_unwaited` on held work of its own: the overlay for a word and for a
  fence, a timestamp wait settled by the pump.
- `a_flight_left_alone_is_settled_by_the_poll`.
- `shadow::check::copies`: C3 and C4 on a copy, each with its clean case.
- `unwaited::tests`: the overlay's arithmetic.

Breaks (`protections_undone_are_found`, `what_is_done_beside_a_flight_is_found`):
words written at the launch; a Signal made at once; Signals made last first;
the flight not holding its lists; the settle not waiting; a second launch
with no settle; a refresh, a declaration, a title copy, a release of retired
objects and an idle sweep beside a flight. Each is found by the script's own
expectations or by the rule named in the test.

## 12. Switches and counters

- `VICTORY_UNWAITED=1`: tier one. Off by default.
- `VICTORY_UNWAITED_CHECK=1` (or `log`), `VICTORY_UNWAITED_CHECK_REFRESH=1`.
- R1 to R3 get one switch each when they are built.

A line of its own after the frame-rate line, only with a switch set (per
frame): title pieces launched and waited for diagnostics; settles by cause;
flights by write footprint (none, declared, unbounded) and read footprint
(exact, hulls, every copy); ms from launch to settle; ms settles stood
waiting for the GPU; settles that found it done; ms of the settles after the
wait; "would have overlapped" for heap sync, uploads, refreshes and title
copies, and the three together (section 7); Signals kept; waits met by a word
to come; and the checker's cost, megabytes compared, ranges unchecked and
violations.

## 13. Where the code is

- `crates/victory-gpu/src/unwaited.rs`: the switches, what a flight will
  tell (words, Signals), the overlay, the poll thread, counters, the
  checker's report.
- `crates/victory-gpu/src/host/unwaited.rs`: `Host::execute_unwaited`,
  `launch_held`, `settle_held`: copies of `Host::execute`'s preparation and
  of what it does after its wait, each place naming the other.
- `crates/victory-gpu/src/shadow/check.rs`: C3 and C4.
- Hooks, each a test of a flag that is false with the switches off:
  `Host::execute` and `Host::quiet` (settle), `queue::pump` (which pump),
  `queue::signal`, `queue::present`, `queue::copy_page_mappings`,
  `fence::completed`, `fence::set_event_on_completion`, `fence::signal`; and
  the checker's in `Shadow::writing`, `Shadow::copy`, `update_free`,
  `copy_back_marks`, `copy_back_declared`, `pieces_at_once`, `idle_sweep`,
  `bindless::uploads`, `resource::evict_unused`, `host::release_retired`,
  `host::copy_mapped_noting`. A draw whose shader writes through the heap's
  views notes it on its host list (`draw.rs`), for the write footprint.
- `Host::execute` itself has one added statement: the settle.

## 14. R1 after the heap sync moved to helper threads (8020fafa)

`sync_heaps` now makes runs of slots on helper threads (`back::spread`), the
helpers marked as holding the submission lock too. "Heap sync beside a
flight" therefore means several threads writing host slots while the GPU
executes, not one, and the submitting thread waiting on them. For R1 this
changes the test to build (the stress test writes its 40,000 slots from as
many threads as the sync uses) and adds one requirement: nothing a helper
does may settle or launch (they hold no lock; `Host::settle_flight` already
refuses on a thread marked as holding the submission lock).
