//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! A record the title has not filled runs empty and is charged nothing
//! ([`unfilled_empty`]): a draw of 2^23 vertices or indices an instance or
//! more, a mesh dispatch of more than 65,535 groups (it leaves all ones
//! there). Clamped instead, two such draws took a list's whole budget and
//! every record after them ran empty, and such a mesh dispatch ran 65,535
//! groups of whatever lay beside it.
//!
//! With a count buffer ExecuteIndirect runs only the first min(max, count)
//! records, so the clamp reads the count word and stops there: the maximum
//! can be far above the count, and the records between never run.
//!
//! The clamps of a run of indirect draws go to the host list before the one
//! that draws ([`bound_before`]; `VICTORY_HOIST_CLAMPS=0` turns it off):
//! each clamp before its draw puts a dispatch of ours and three barriers
//! between every two draws, which ends and restarts the host's render pass
//! each time.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u1), UAV(u0), SRV(t1)"
// flags: 1 the title's record count is read, 2 unfilled records run empty.
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint flags; };
// The title's records, read only; the clamped ones go to the list's own.
ByteAddressBuffer args : register(t0);
RWByteAddressBuffer dest : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty. Its second word counts the unfilled records.
RWByteAddressBuffer budget : register(u1);
// The title's record count: read only with flag 1.
ByteAddressBuffer countWord : register(t1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// The vertices or indices an instance from which a draw record is none of
// the title's.
static const uint MOST_DRAW = 1u << 23;
// Whether the record at `at` is a slot the title has not filled: it runs
// empty and costs nothing.
bool unfilled(uint at) {
    if ((flags & 2) == 0 || kind == 2) {
        return false;
    }
    uint first = args.Load(at);
    if (kind <= 1) {
        return first >= MOST_DRAW;
    }
    return first > 65535u;
}
// Clamps the record at `at` into `dest`; its cost.
uint clamped(uint at) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        if (unfilled(at)) {
            count = 0;
            instances = 0;
        }
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        // The rest of the record as it is: first vertex and instance (4 words
        // in all), or first index, base vertex and first instance (5).
        if (kind == 1) {
            dest.Store3(at + 8, args.Load3(at + 8));
        } else {
            dest.Store2(at + 8, args.Load2(at + 8));
        }
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dest.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        if (unfilled(at)) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 128;
        dest.Store3(at, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    // ExecuteIndirect runs min(max, count) records: no others are clamped or
    // charged. The loop still counts tiles up to `records`, a constant, as
    // the compiler cannot prove that a value read from memory is the same
    // in every lane, and barriers need that; tiles past `walked` only wait
    // at them.
    uint walked = records;
    if ((flags & 1) != 0) {
        walked = min(records, countWord.Load(0));
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * stride;
        uint cost = 0;
        if (record < walked) {
            cost = clamped(at);
            if (unfilled(at)) {
                uint was;
                budget.InterlockedAdd(4, 1u, was);
            }
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0 && tile * 64 < walked) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    // The first record that does not fit takes what is
                    // left, so those after it run empty too.
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            dest.Store(at, 0);
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// Built from [`WORDS_SOURCE`], which also reads the budget as words
    /// through root view 5 (the other views are those of [`SOURCE`]).
    words: bool,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    make_fxc().or_else(|e| {
        tracing::info!(error = %e, "argument clamp: building the one-thread form");
        make_words()
    })
}

/// The clamp for Wine's HLSL compiler (vkd3d-shader), which has no `Load`
/// on `RWByteAddressBuffer`, no stores to indexed group-shared memory and
/// no `[RootSignature]`: [`SOURCE`]'s clamp, read through structured views
/// of the same buffers, and one thread charging the records in order
/// instead of a group of 64. The results are the same.
const WORDS_SOURCE: &str = r#"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint flags; };
StructuredBuffer<uint> argWords : register(t0);
RWByteAddressBuffer budget : register(u1);
RWByteAddressBuffer dest : register(u0);
StructuredBuffer<uint> countWords : register(t1);
RWStructuredBuffer<uint> budgetWords : register(u2);
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;
static const uint MOST_DRAW = 1u << 23;
uint3 load3(uint at) { return uint3(argWords[at / 4], argWords[at / 4 + 1], argWords[at / 4 + 2]); }
bool unfilled(uint at) {
    if ((flags & 2) == 0 || kind == 2) {
        return false;
    }
    uint first = argWords[at / 4];
    if (kind <= 1) {
        return first >= MOST_DRAW;
    }
    return first > 65535u;
}
uint clamped(uint at) {
    uint cost;
    if (kind <= 1) {
        uint count = min(argWords[at / 4], 1u << 24);
        uint instances = min(argWords[at / 4 + 1], (1u << 26) / max(count, 1u));
        if (unfilled(at)) {
            count = 0;
            instances = 0;
        }
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        if (kind == 1) {
            dest.Store3(at + 8, uint3(argWords[at / 4 + 2], argWords[at / 4 + 3], argWords[at / 4 + 4]));
        } else {
            dest.Store2(at + 8, uint2(argWords[at / 4 + 2], argWords[at / 4 + 3]));
        }
    } else if (kind == 2) {
        uint3 g = min(load3(at), uint3(65535, 65535, 65535));
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dest.Store3(at, g);
    } else {
        uint3 g = load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        if (unfilled(at)) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 128;
        dest.Store3(at, g);
    }
    return cost;
}
[numthreads(1, 1, 1)]
void main() {
    uint left = BUDGET - min(budgetWords[0], BUDGET);
    uint emptied = budgetWords[1];
    uint walked = records;
    if ((flags & 1) != 0) {
        walked = min(records, countWords[0]);
    }
    for (uint record = 0; record < walked; record++) {
        uint at = record * stride;
        uint cost = clamped(at);
        if (unfilled(at)) {
            emptied = emptied + 1u;
        }
        if (cost > left) {
            // The first record that does not fit takes what is left, so
            // those after it run empty too.
            dest.Store(at, 0);
            left = 0;
        } else {
            left -= cost;
        }
    }
    budget.Store(0, BUDGET - left);
    budget.Store(4, emptied);
}
"#;

fn make_words() -> windows::core::Result<Clamp> {
    use crate::compute::Param;
    let params = [
        Param::Constants { register: 0, count: 4 },
        Param::Srv { register: 0, space: 0 },
        Param::Uav { register: 1 },
        Param::Uav { register: 0 },
        Param::Srv { register: 1, space: 0 },
        Param::Uav { register: 2 },
    ];
    let (root, pso) = crate::compute::pipeline(WORDS_SOURCE, s!("clamp"), 0, &params)?;
    Ok(Clamp { root, pso, words: true })
}

fn make_fxc() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso, words: false })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` (a title
/// buffer's copy, only read) into the list's own argument buffer, left
/// ready to be read as arguments: that buffer, for [`done`] after the
/// execution. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp is not available.
///
/// `count` is the host address of the title's record count (a title
/// buffer's copy, only read): the clamp then stops at the lesser of it and
/// `records`, which is where ExecuteIndirect stops. A count that is not
/// 4-byte aligned is not read: every record is clamped.
///
/// Title buffers' copies are never transitioned: they are common between
/// submissions and every use promotes them (shader and root views, the
/// input assembler, these reads), with unordered-access barriers for order.
/// Only the list's argument buffer moves, common, unordered access,
/// indirect argument, common: its states are known whatever order lists
/// run in.
// shortcut: a buffer a shader writes through an unordered view and the
// input assembler then reads in one submission is, by the letter of D3D12,
// promoted to a write state the input assembler may not use (the debug
// layer cannot see view accesses; GPU-based validation might); upgrade to
// per-list tracked buffer states if a driver or that validation minds.
pub(crate) fn bound(
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    count: Option<u64>,
) -> Option<ID3D12Resource> {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return None;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let count = count.filter(|at| at.is_multiple_of(4));
    RECORDS[0].fetch_add(u64::from(records), Ordering::Relaxed);
    if count.is_some() {
        RECORDS[1].fetch_add(u64::from(records), Ordering::Relaxed);
    }
    let out = list.arguments(u64::from(stride) * u64::from(records.max(1)))?;
    let l = &list.list;
    // SAFETY: an open list; our buffers, and title buffers' copies the
    // caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        run(l, c, budget, (gpu, count), (kind, stride, records), out.GetGPUVirtualAddress());
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
    }
    Some(out)
}

/// Records the clamp's dispatch: the records at `gpu`, under the count at
/// `count` if any, clamped into the memory at host address `dest`, in
/// unordered access.
///
/// # Safety
/// `l` is open; `budget`, `gpu`, `count` and `dest` are addresses in live
/// buffers the list holds.
unsafe fn run(
    l: &ID3D12GraphicsCommandList,
    c: &Clamp,
    budget: u64,
    (gpu, count): (u64, Option<u64>),
    (kind, stride, records): (u32, u32, u32),
    dest: u64,
) {
    let flags = u32::from(count.is_some()) | u32::from(unfilled_empty()) << 1;
    let constants = [kind, stride, records, flags];
    // SAFETY: the caller's; our root signature and pipeline.
    unsafe {
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, dest);
        // Set even when not read: a root view has to be set.
        l.SetComputeRootShaderResourceView(4, count.unwrap_or(gpu));
        if c.words {
            l.SetComputeRootUnorderedAccessView(5, budget);
        }
        // One group walks every record (up to the count).
        l.Dispatch(records.min(1), 1, 1);
    }
}

/// Whether a record the title has not filled runs empty and is charged
/// nothing to its list's budget (the module notes):
/// `VICTORY_INDIRECT_UNFILLED=0` clamps it like any other, as before.
/// Read once, here.
fn unfilled_empty() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::UNFILLED.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_INDIRECT_UNFILLED"))
}

/// Counts the unfilled records a list's clamps ran empty (the second word
/// of its budget, once it has run).
pub(crate) fn note_unfilled(records: u32) {
    UNFILLED.fetch_add(u64::from(records), Ordering::Relaxed);
}

/// Unfilled records run empty since the last call (the frame-rate line).
pub(crate) fn take_unfilled() -> u64 {
    UNFILLED.swap(0, Ordering::Relaxed)
}
static UNFILLED: AtomicU64 = AtomicU64::new(0);

/// Whether the clamps of a run of indirect draws go to the host list
/// before the one that draws (`VICTORY_HOIST_CLAMPS=0`: each before its
/// draw).
pub(crate) fn hoisting() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_HOIST_CLAMPS").map_or(true, |v| v.trim() != "0"))
}

/// As [`bound`], recorded in `clamps`, the host list that runs before
/// `into`, for an execution `into` records: the clamped records go to one
/// of `into`'s own buffers ([`HostList::clamp_room`]), which the execution
/// reads at the offset returned once `clamps` has ended
/// ([`HostList::clamp_end`]). `None` also when there is no room: the
/// caller clamps in `into`.
///
/// The clamps of a run charge the budget of `clamps` one after another, in
/// the order of the draws (a barrier for shader writes before each): the
/// records past the budget are the same ones from frame to frame. Clamps
/// that did not wait for each other made whole mesh objects flicker on
/// the sister fork (leaves, chairs, clothes; 2026-11-25).
pub(crate) fn bound_before(
    clamps: &HostList,
    into: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    count: Option<u64>,
) -> Option<(ID3D12Resource, u64)> {
    let (c, budget) = (clamp()?, clamps.budget()?);
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let count = count.filter(|at| at.is_multiple_of(4));
    let (out, at) = into.clamp_room(clamps, u64::from(stride) * u64::from(records.max(1)))?;
    RECORDS[0].fetch_add(u64::from(records), Ordering::Relaxed);
    if count.is_some() {
        RECORDS[1].fetch_add(u64::from(records), Ordering::Relaxed);
    }
    HOISTED.fetch_add(1, Ordering::Relaxed);
    let l = &clamps.list;
    // SAFETY: an open list; our buffers (`out` in unordered access), and
    // title buffers' copies the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes, the clamp
        // before) first.
        l.ResourceBarrier(&[uav_barrier()]);
        run(l, c, budget, (gpu, count), (kind, stride, records), out.GetGPUVirtualAddress() + at);
    }
    Some((out, at))
}

/// Clamps recorded in the host list before their draw's since the last
/// call (the frame-rate line).
pub(crate) fn take_hoisted() -> u64 {
    HOISTED.swap(0, Ordering::Relaxed)
}
static HOISTED: AtomicU64 = AtomicU64::new(0);

/// Records the clamps were asked to walk (the maximum of each execution)
/// since the last call, and the part of them under a count buffer, where
/// the walk stops at the count.
pub(crate) fn take_records() -> [u64; 2] {
    RECORDS.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}
static RECORDS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// After the execution: the list's argument buffer back to common.
pub(crate) fn done(list: &HostList, arguments: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            arguments,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    thread_local! {
        /// This thread's [`super::unfilled_empty`], where a test sets it.
        pub(super) static UNFILLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }

    /// The most vertices a draw record of the title's has, and the
    /// instances such a record is clamped to (2^26 vertex instances).
    const MOST: u32 = (1 << 23) - 1;
    /// A draw record of [`MOST`] vertices and any instances: clamped to 8
    /// instances, two of them fit a list's budget.
    const LARGE: [u32; 4] = [MOST, u32::MAX, 0, 0];

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word in the list's
    /// argument buffer after (zero when it runs empty). The title's copy,
    /// one record more included, stays as written. `None` without a host
    /// GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        clamped_under(kind, words, records, calls, None)
    }

    /// As [`clamped`], with the title's record count `count` (a word in a
    /// buffer of its own) if there is one.
    fn clamped_under(
        kind: u32,
        words: &[u32],
        records: u32,
        calls: u32,
        count: Option<u32>,
    ) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let (counter, word) = host.shared_buffer(4).unwrap();
            word.cast::<u32>().write_volatile(count.unwrap_or(0));
            let counted = count.map(|_| counter.GetGPUVirtualAddress());
            let mut out = None;
            for _ in 0..calls {
                let clamped = super::bound(&list, gpu, spec, counted).unwrap();
                super::done(&list, &clamped);
                out = Some(clamped);
            }
            list.close();
            host.execute(&[&list]);
            for at in 0..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()], "the title's copy as written");
            }
            let mut out_data = std::ptr::null_mut();
            out?.Map(0, None, Some(&mut out_data)).unwrap();
            let first = |i: usize| out_data.cast::<u32>().add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Rejected records do not wrap the budget: of 128 large draws (2^26
    /// vertex instances each once clamped, less 8) two run, where a
    /// wrapping add let four; of 600 mesh dispatches of 65,535 groups 16,
    /// not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &LARGE, 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[65_535, 1, 1], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// Records the title has not filled run empty and cost nothing: the
    /// draws between them all run, where clamped (as before, and with
    /// `VICTORY_INDIRECT_UNFILLED=0`) the unfilled ones took the budget and
    /// the draws after them ran empty. Likewise mesh dispatches of more
    /// than 65,535 groups, which were 65,535.
    #[test]
    fn unfilled_records_run_empty_and_cost_nothing() {
        // In turn: 2^23 vertices (the least that is no draw), a draw of
        // 100, all ones, a draw of 7.
        let mut words = vec![1 << 23, 5, 0, 0, 100, 1, 0, 0];
        words.extend([u32::MAX; 4]);
        words.extend([7, 1, 0, 0]);
        let before = super::UNFILLED.load(super::Ordering::Relaxed);
        let Some(draws) = clamped(0, &words, 64, 1) else {
            return; // no host GPU here
        };
        for (i, &count) in draws.iter().enumerate() {
            assert_eq!(count, [0, 100, 0, 7][i % 4], "record {i}");
        }
        // Counted for the frame line (other tests may add their own).
        assert!(super::UNFILLED.load(super::Ordering::Relaxed) >= before + 32);
        // The largest draw of the title's still runs.
        let most = clamped(0, &[MOST, 1, 0, 0], 4, 1).unwrap();
        assert_eq!(most, [MOST; 4]);
        let meshes = clamped(10, &[u32::MAX, u32::MAX, u32::MAX, 5, 1, 1, 65_536, 1, 1, 65_535, 1, 1], 8, 1).unwrap();
        assert_eq!(meshes, [0, 5, 0, 65_535, 0, 5, 0, 65_535]);
        // Indirect dispatches are not told apart: clamped as before.
        let dispatches = clamped(2, &[70_000, 1, 1], 2, 1).unwrap();
        assert_eq!(dispatches, [65_535, 65_535]);

        UNFILLED.set(Some(false));
        let draws = clamped(0, &words, 64, 1);
        let meshes = clamped(10, &[u32::MAX, u32::MAX, u32::MAX, 5, 1, 1], 4, 1);
        UNFILLED.set(None);
        let draws = draws.unwrap();
        // 5 * 2^23, 100 and 2^26 fit; the next unfilled record does not,
        // and nothing after it runs.
        assert_eq!(draws[..5], [1 << 23, 100, 1 << 24, 7, 0]);
        assert!(draws[5..].iter().all(|&count| count == 0));
        assert_eq!(meshes.unwrap(), [65_535, 5, 65_535, 5]);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit ends the budget: after two draws of
    /// 3 * 2^24 vertex instances a third takes what is left, and the small
    /// ones after it run empty.
    #[test]
    fn a_record_that_does_not_fit_ends_the_budget() {
        let mut words = [1 << 22, 12, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, false, false, false, false, false]);
    }

    /// Two lists clamping into their own argument buffers, run in either
    /// order: every barrier starts from the buffer's current state (the
    /// debug layer's rule; runs light19 and light21 had the title's GPU
    /// heap moved to unordered access under the input assembler), the
    /// title's copy is never transitioned, and all end common.
    #[test]
    fn argument_buffers_move_from_their_current_states() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let out = super::bound(l, gpu, (2, 12, 1), None).unwrap();
                    super::done(l, &out);
                    l.close();
                    (out, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            host.execute(&[&lists[order[0]], &lists[order[1]]]);
            let mut run = issued();
            run.extend(order.iter().flat_map(|&i| recorded[i].1.clone()));
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
            replay(&run, &mut states).unwrap();
            assert!(run.iter().all(|t| t.0 != key(&title)), "the title's copy is never transitioned");
            assert!(states.values().all(|&s| s == 0), "all common again");
            for (out, _) in &recorded {
                let mut p = std::ptr::null_mut();
                // SAFETY: a mapped buffer of ours, the lists have run.
                let clamped = unsafe {
                    out.Map(0, None, Some(&mut p)).unwrap();
                    p.cast::<[u32; 3]>().read_volatile()
                };
                assert_eq!(clamped, [65_535, 1, 1]);
            }
            // SAFETY: as above.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [70_000, 1, 1], "the title's copy as written");
        }
    }

    /// Hoisted clamps (`bound_before`) leave the clamped records in the
    /// drawing list's own buffer, one run of records after the other, and
    /// charge the budget of the list before in the order of the draws: of
    /// two draws of four large records (2^26 vertex instances each once
    /// clamped, less 8) the first two records of the first fit, none of the second;
    /// under a count of 1 only the first is clamped. Every barrier starts
    /// from the buffer's current state, the title's copy is never
    /// transitioned and all end common.
    #[test]
    fn hoisted_clamps_fill_the_drawing_lists_buffer() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (per, records) = (4usize, 4u32);
        let bytes = (per * records as usize * 4) as u64;
        let (title, data) = host.shared_buffer(bytes).unwrap();
        let (counter, word) = host.shared_buffer(4).unwrap();
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for count in [None, Some(1u32)] {
            let (clamps, draws) = (host.new_list().unwrap(), host.new_list().unwrap());
            issued();
            // SAFETY: our mapped buffers, written before the lists run and
            // read after they have; live buffers.
            unsafe {
                for at in 0..per * records as usize {
                    data.cast::<u32>().add(at).write_volatile(LARGE[at % per]);
                }
                word.cast::<u32>().write_volatile(count.unwrap_or(0));
                let gpu = title.GetGPUVirtualAddress();
                let counted = count.map(|_| counter.GetGPUVirtualAddress());
                let spec = (0, per as u32 * 4, records);
                let first = super::bound_before(&clamps, &draws, gpu, spec, counted).expect("room");
                let second = super::bound_before(&clamps, &draws, gpu, spec, counted).expect("room");
                assert!(first.0 == second.0 && second.1 == first.1 + bytes, "one buffer, in turn");
                draws.clamp_end(&clamps);
                assert!(super::bound_before(&clamps, &draws, gpu, spec, counted).is_none(), "the run has ended");
                clamps.close();
                draws.close();
                let run = issued();
                host.execute(&[&clamps, &draws]);
                let mut states: std::collections::HashMap<usize, u32> =
                    [key(&title), key(&counter), key(&first.0)].into_iter().map(|k| (k, 0)).collect();
                replay(&run, &mut states).unwrap();
                assert!(run.iter().all(|t| t.0 == key(&first.0)), "only the list's buffer is transitioned");
                assert_eq!(run.len(), 3, "to unordered access, indirect argument and common");
                assert!(states.values().all(|&s| s == 0), "all common again");
                let mut out = std::ptr::null_mut();
                first.0.Map(0, None, Some(&mut out)).unwrap();
                let out = out.cast::<u32>();
                let counts: Vec<u32> = (0..2 * records as usize).map(|i| out.add(i * per).read_volatile()).collect();
                match count {
                    None => assert_eq!(counts, [MOST, MOST, 0, 0, 0, 0, 0, 0]),
                    // The words past the count stay as the fresh buffer's.
                    Some(_) => assert_eq!(counts, [MOST, 0, 0, 0, MOST, 0, 0, 0]),
                }
                for at in 0..per * records as usize {
                    assert_eq!(data.cast::<u32>().add(at).read_volatile(), LARGE[at % per], "the title's copy as written");
                }
            }
        }
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[MOST, u32::MAX, 0, 0, 0], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
        let once = clamped(1, &[MOST, u32::MAX, 0, 0, 0], 64, 1).unwrap();
        assert_eq!(once.iter().filter(|&&count| count != 0).count(), 2, "which a first execution has room for");
    }

    /// The walk stops at the count: of 128 large draws under a count of 3
    /// the first two run and the third runs empty, as they do without one,
    /// and the others are never read (the argument buffer's fresh words stay
    /// zero). A count above the records changes nothing; a count of zero
    /// walks none.
    #[test]
    fn the_walk_stops_at_the_count() {
        let Some(under) = clamped_under(0, &LARGE, 128, 1, Some(3)) else {
            return; // no host GPU here
        };
        assert_eq!(under[..3], [MOST, MOST, 0]);
        assert!(under[3..].iter().all(|&count| count == 0));
        let all = clamped(0, &LARGE, 128, 1).unwrap();
        assert_eq!(under[..3], all[..3]);
        let above = clamped_under(0, &LARGE, 128, 1, Some(1 << 20)).unwrap();
        assert_eq!(above, all);
        let none = clamped_under(0, &LARGE, 128, 1, Some(0)).unwrap();
        assert!(none.iter().all(|&count| count == 0));
    }

    /// Only the records that run are charged to the budget: after a call
    /// under a count of 1 a second finds room for that record again, where
    /// the same draws with no count spent it all on records that never ran.
    #[test]
    fn records_past_the_count_are_not_charged() {
        let Some(twice) = clamped_under(0, &LARGE, 128, 2, Some(1)) else {
            return; // no host GPU here
        };
        assert_eq!(twice[0], MOST);
        let spent = clamped(0, &LARGE, 128, 2).unwrap();
        assert!(spent.iter().all(|&count| count == 0));
    }
}
