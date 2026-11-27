# Heap slots: removing the per-frame descriptor heap sync

Design and feasibility notes for issue #14, on top of the inline views of
#20 (constant buffer and placed views carried in their records,
`VICTORY_VIEW_INLINE`). The idea is that each view gets a fixed slot in
one host heap when it is made, and shaders find that slot through the
title's heap bytes, so a title copying descriptors as memory costs the host
nothing. It came from a survey of another project. Only the idea is used
here: everything below is from our own code and the public D3D12 and Metal
documentation.

## Where the time goes today

From one diagnostics run (Series X, world, ~1,000 frames, the stats block
at the end of the run), per frame, on the submitting thread:

| step | ms | what it is |
|---|---|---|
| taking the heaps' changed pages | 0.6 | `Tracked::take_resting` |
| looking at the records | 4.9 | 434,000 records read and compared with what each slot was made from (`HostHeap::due_runs`, `views_current`) |
| making slots | 39.9 | 46,800 slots (`HostHeap::make_runs`), about 850 ns each |
| **heap syncs, all** | **46.0** | |

The slots made, by why (`Due`):

- 32,200 (69 %) whose record names no view any more. Of these, 26,600
  hold other bytes and 5,600 zeros. Only 6 are views of released
  resources. The samples show the title keeping its own data (floats,
  structures every 13 slots) in heap memory between its descriptors.
  Draws reading tables found **no** such slot (`table entries draws read
  that name no view: 0`).
- 8,900 the title rewrote with another view.
- 5,700 made for the first time (often slots that were nulled earlier).
- By kind: 5,500 constant buffer views, 300 placed textures left until
  read.

The time inside a made slot, by step, as shares of the timed slots:
what the slot views (`crate::bindless::Viewed`) 42 %, the host view
22 %, the stride, mark and address tables 17 %, what it holds (`kept`)
11 %, finding the record's view 9 %. A slot made null does all of these
except finding the view and the copy lookups of a buffer view.

## 1. How shaders would get the heap bytes on the GPU

How it works today: shaders take descriptors through SM 6.6
`ResourceDescriptorHeap[i]` (`victory_shader::emit::shader::heap_handle`),
with `i` the title's own slot number (table offset ÷ 32 plus the offsets
the code adds). So each host heap keeps the title's order, and every title
slot that changes has to be made again on the host.

**(a) The heap as host memory the GPU reads directly: not workable under
Wine + D3DMetal.**

- The title places its heaps in its own memory (`Placement`), so we do not
  choose the address.
- Turning the title's pages into a GPU resource would need D3D12's
  `ID3D12Device3::OpenExistingHeapFromAddress`. Nothing documents D3DMetal
  supporting it. Underneath, Metal's no-copy buffers need memory aligned to
  the host page, 16 KiB on Apple silicon.
- The other way round does not work either. A mapped upload buffer handed
  to the title as heap memory is invisible to Wine's `VirtualQuery`
  (`docs/macos.md`). `sync_heaps` probes pages with `read_into` before it
  reads them, and the heap is write-watched (`written: Tracked`). Both rely
  on Wine seeing that memory.

**(b) Changed heap pages taken to the GPU each frame: works, with what we
already have.** The heap is tracked already, and each sync gets its changed
pages for 0.6 ms. There are two ways to do it:

- (b1) Upload the title's bytes, 32 bytes a slot. Shaders then decode the
  record on the GPU: check the tag, read the fixed slot from bytes 12-15,
  and compare the generation with a GPU table. That is 434,000 slots ×
  32 B = 14 MB a frame, and the shader carries the decoding.
- (b2) **Recommended:** a *slot map*, one dword per title slot, written by
  the host during the comparison the sync already does (4.9 ms). The dword
  is the fixed host slot of the view the record names, or 0 when the
  record names none. This is the pattern of the view-stride table
  (`HostHeap::strides`, `VIEW_STRIDES`): a mapped buffer the host writes
  per slot and shaders read at the title's slot number. The host checks
  the tag, the generation and garbage on the CPU, as it does now.

**Lowering.** `heap_handle` would put one load in front of
`createHandleFromHeap`: `host = slot_map[i]` (a `rawBufferLoad` from a root
SRV, like `buffer_guard`'s stride lookup). When `host == 0`, the shader
takes a null of its own declared kind and dimension from a small fixed set
of null slots. The shader knows the kind it declared, so the host never
makes nulls.

**Cost per access.** One dependent 4-byte load per descriptor taken. Most
indices are uniform across a wave, so the load is once per wave and
usually cached. Buffer views at computed indices already pay one such load
(the stride table). Packing the stride and the slot into one 8-byte entry
keeps those accesses at one load. The cost on Apple GPUs is **not
measured**. It is the riskiest unknown of the full design (see below).

## 2. Null slots and view generations in the fixed-slot model

- **Nulls:** map entry 0, with the null chosen by the shader as above. The
  32,200 slots a frame made null today then cost one dword each.
- **Views of resources** (table entries with generations): the fixed slot
  belongs to the entry. When the resource is released (`forget`), the
  generation moves on and the sync writes 0 for records of the old
  generation as it meets them. The fixed slot is reused only when (1) no
  title slot maps to it and (2) the submissions that could read it have
  run (`crate::host::retire`, as `kept` does now). For (1), keep a count
  per fixed slot, changed by the sync's comparison: one down for the old
  record and one up for the new. Until both hold, the slot keeps its
  descriptor and its resource, so a stale title slot reads the released
  resource's old contents, as today, and never freed memory.
- **Inline views** (constant buffers, placed views) have no release event.
  Their fixed slots would be keyed by content and freed at count 0 after
  the fence. Constant buffer views move every frame (5,500 a frame), so
  that is thousands of slots handed out and freed each frame. Handing them
  out a page of 256 to a thread keeps that off shared locks. An
  alternative: the shader already reads heap constant buffers as raw
  views at their address, so with the address in the record (#20) it could
  read them through the title-memory page table and need no slot at all.
  That needs the record bytes on the GPU (b1) and changes what such
  programs keep current, so it is a later step.
- **`Viewed`, `kept`, strides and marks** move from title slots to fixed
  slots. They are set once when the view gets its slot, and a view counts
  as viewed while its count is above zero. This removes the largest step
  of making a slot (42 %).

## 3. What the inline records of #20 change

The sync is **already incremental**: `due` compares each record's key with
the key the slot was made from, and only records that changed are made
(46,800 of 434,000 looked at). Inline records make the key the view's
content for constant and placed views. So writing the same constant
buffer into a slot again is passed by without a table lookup, and the
views table no longer grows. They do not avoid the host work for a record
that really changed, because the host heap is in the title's order. What
does make a cheap first step is that **69 % of the slots made are records
that changed into bytes that are not records at all**.

## 4. What each design removes

| | look 4.9 | make null 32k | make views 14.6k | take 0.6 |
|---|---|---|---|---|
| stage 1 (`VICTORY_STALE_LEFT`) | stays | 26.6k of 32.2k removed (zeros and other bytes) | stays | stays |
| full design (slot map + fixed slots) | stays (writes the map, keeps counts) | one dword each | one dword each; views made once when written | stays |

## Staged plan

| stage | what | saves (of 46 ms) | effort | risk |
|---|---|---|---|---|
| 1 | `VICTORY_STALE_LEFT` (on by default since the A/B: heap syncs 35.4 -> 23.4 ms; `=0` turns it off): a sync leaves slots whose record became zeros or other bytes as they are (prototyped) | ~18-28 ms (estimate: null slots cost 60-80 % of a view's; plus slots written back to their old record cost nothing) | done; one A/B run | low to medium: see below |
| 1b | Bound what left slots retain: null them off the submitting thread after a few frames, or in the sweep of idle copies | 0 (keeps stage 1's saving bounded) | small | low |
| 2 | Probe the indirection's GPU cost under D3DMetal (a compute shader reading N descriptors directly vs through a map) | none: decides stage 3 | small | none |
| 3 | Slot map + fixed slots for table views, samplers and placed textures; `heap_handle` lowered through the map; nulls by declared kind; `Viewed`/`kept`/strides by fixed slot; counts from the sync's comparison | most of the rest: syncs to ~6-9 ms (look 4.9 + map writes and counts ~1-3) | large (weeks): every heap-access shader, the bindless bookkeeping, the dynamic slots and host copies' slot pairs | high: shader change everywhere, slot lifetime, GPU cost |
| 4 | Constant buffer (and placed buffer) views without slots, through the address in the record | slot churn of ~5,500 a frame | medium | medium |

The draws' own slot reads (131,000 slot locks a frame on the recording
threads) are outside the 46 ms. Stage 3 would also take them to the map.

### Stage 1 risks and what to measure

What changes when the switch is on:

- A shader that takes a left slot from the heap at an index from its data
  reads the view the slot had, not a null. The console would read the
  title's bytes as a descriptor there, so neither is the console's result.
- The slot keeps its resource and what it views (`crate::bindless`) until
  it is written again. Its buffer range stays in its copy's hull, so it is
  refreshed and its copy is not swept.

What stays the same:

- A draw reading such a slot through a table still makes it null
  (`made_already` finds the slot not made from its record).
- A record of ours that names no view (a released resource's) is still
  made null.

Measure with matched end screenshots: `heap syncs` ms and slots made, the
new `left as they were` count, `heaps' hulls`, refresh batch ms and MB,
`viewed textures looked at`, footprint, and the image.

## Prototype

- `VICTORY_STALE_LEFT` (on by default; `=0` turns it off) in `descriptor/host.rs`
  (`stale_left`, used in `HostHeap::due_runs`). There is a new counter,
  `SyncStaleLeft`, on the "heap sync slots" line.
- Test `a_sync_leaves_slots_whose_record_came_to_be_other_bytes_with_the_switch`
  (under Wine, needs the host device). It checks that with the switch,
  other bytes and zeros leave the slot made from its view, and a record
  written back costs nothing. A draw reading the slot makes it null. A
  record of ours naming no view is made null. Without the switch the slot
  is made null as before.

## Stage 2 of the revised plan: write marks, and draws' table records

Draws' table records left to the draws is on by default (`VICTORY_TABLES_TO_DRAWS=0` turns it off); write marks stay off by default (`VICTORY_WRITE_MARKS=1`), since small coverage gaps remain.

### Write marks at the source (`VICTORY_WRITE_MARKS=1`)

The idea is another fork's "renderer port 2" (avatarmanz,
`VICTORY_PORT_DESCRIPTORS`); the code is our own.

- **Stores.** The title fills its draws' and dispatches' tables with six
  32-byte record stores in two table fillers. At launch,
  `victory_loader::mark_stores::patch` checks each site's form (the
  offset mask with 7FE0h, the record load, the store of the known form). A site
  that matches becomes a rel32 jump to a stub near the image. The stub does
  the store, then sets the slot's byte and its 128-slot chunk's byte in the
  maps of `victory_gpu::write_marks`, and jumps back. A site of another
  form is left alone and logged. When any site is not patched, no heap
  is marked and every sync reads changed pages as before.
- **Flags.** The stub compares, so it changes the flags. It leaves them
  changed only where the bytes after the store are shown to set every
  status flag before any read (`flags_dead_after`: a `movzx`, then a `cmp`
  between registers). Everywhere else it keeps them (`pushfq`/`popfq`). All
  six sites show the flags dead.
- **Copies and our calls.** The title's `memcpy`/`memmove` imports go
  through ours, which mark each slot whose copied record is a view record.
  Our `Create*View` and copy calls mark theirs (`descriptor::mirror`).
- **Sync.** `descriptor::due_in_heap` takes the marked slots
  (`Maps::take`) in place of the write watch's changed pages, and compares
  only those.
- **Fallback.** Every `VICTORY_WRITE_MARKS_EVERY` frames (8) the write
  watch is asked. Each frame `VICTORY_WRITE_MARKS_CHECK` of its pages (16)
  are read whole, leaving out the slots this sync took by their marks, so
  each slot is in one part only. (That is the bug the other fork's PR #354
  fixed: overlapping parts, and marked slots counted as missed.) A new view
  record found there that no mark told of is made and counted as
  "unmarked", with a warning for the first 20 and then powers of two.
- **Not covered.** A slot whose view record an unmarked writer overwrote
  with other bytes keeps its view until the fallback reads its page.
  `VICTORY_STALE_LEFT`, on by default, leaves such slots anyway.

### Draws' table records left to the draws (`VICTORY_TABLES_TO_DRAWS=1`)

After avatarmanz's commit 1f1a4336 (and its `copied_waits`), fitted to our
sync:

- With the switch, a heap notes the frame each slot was last read by a draw
  through a table (`HostHeap::note_tabled`).
- When a slot read so within the last 8 frames gets a new view record, a
  sync passes it by. The draw that reads it makes it (`made_already` finds
  it not made from its record).
- A new constant buffer or placed view elsewhere waits 4 frames for a draw.
  After that it is made by a sync (`HostHeap::waited_out`), unless a draw
  read it meanwhile.
- The fallback's pages leave a slot as a sync left it (the slot notes the
  record it was left with, `HostHeap::left_as`). A record no sync left
  there, which no mark told of, is made and counted.
- It works without write marks too, over the changed pages.
- shortcut: a shader that takes such a slot by an index from its data
  meanwhile reads the view the slot had.
- Not ported: that fork's parallel comparison (44629de2). With write marks
  little is compared. Its "slots with no view that show none are left
  alone" is already how our `due` works.

### First A/B run (Series S, PC upscaler)

| | heap syncs (median) |
|---|---|
| base | 24.3 ms |
| `VICTORY_WRITE_MARKS=1` | 22.4 ms |
| both switches | 17.3 ms |

- All six stores were patched.
- With both switches the fallback reported ~1,800 unmarked view records a
  frame. These were false: slots whose mark a sync had taken and then
  left to the draws or to wait were still due when the fallback read their
  page. The fallback counted them and made them, undoing part of what
  `VICTORY_TABLES_TO_DRAWS` saves. Fixed: the fallback leaves such slots as
  they were left.
- With write marks alone the fallback finds ~1.1-1.7 truly unmarked view
  records a frame. Most sit at offset 0xffa0 of a 64 KiB block. Their
  writer is not known yet. The warning now names the block's last write
  fault (`writer`), whether the slot had been made from another record
  (`made_before`), and the offset in the block. The stats line tallies the
  writers.
- "Title copies into the heap marked 0" is right as measured. The
  `memcpy`/`memmove` imports are hooked (4,500 copies a frame into the
  heap), but what they copy there is constant data, not view records.

### Second A/B run

| | heap syncs (median) |
|---|---|
| base | 24.1 ms |
| `VICTORY_TABLES_TO_DRAWS=1` alone | 19.2 ms |
| both switches | 17.3 ms |

- Unmarked view records: 1.12 a frame.
- The writers named by the guards turned out not to be the records'
  writers. A guard names only the first write to a block after the block
  is guarded again. The named code is the title's frame allocator:
  - +0x18849c links a freshly taken 64 KiB block (the block's first 8
    bytes);
  - +0x4e9b4e2 writes the block's footer (its last 0x18 bytes);
  - +0x325bd6c fills 12-byte entries of another table.
- None of them stores a view record. The writer outside the title's code
  is most likely the C runtime's `memcpy` behind our hook, filling a new
  block with constant data. The log now names its module.
- What the misses share is where they sit. 14 of 23 warnings are at offset
  0xffa0, the place of an allocation of up to 0x48 bytes that ends just
  before the footer. Their writer is still not found: 118,000 32-byte
  stores in the title's code leave a static search no hope.
- Write marks therefore read the two records before every block's footer
  once a frame (`VICTORY_WRITE_MARKS_TAIL`, on with write marks, `=0`
  off), about 1,200 records. A record no mark told of there is made within
  a frame and counted apart (`block tails: … unmarked view records
  there`).

### What to compare in an A/B run

The stats block (diagnostics build) has a line per 5 s: `heap syncs per
frame: … ms in … syncs, … slots looked at, … made; write marks (…): …
slots taken by their marks …, fallback …: … unmarked view records …; draws'
table records left to the draws …`. Compare it with `preparing, all of it
per frame: … heap syncs …` and `heap sync time per frame`. The launcher
prints `write marks    6 of 6 descriptor record stores patched`.

## Uncertain

- The cost of a null slot against a view's is inferred from step shares,
  not measured by kind, which is why stage 1's saving is a range.
- Whether hull retention under stage 1 costs refresh time that offsets the
  saving.
- The GPU cost of one more dependent load per descriptor on Apple GPUs
  through D3DMetal (stage 2 answers it).
- Whether D3DMetal implements `OpenExistingHeapFromAddress`. Untested; the
  design does not depend on it.
