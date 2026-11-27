//! IR to DXIL, one machine instruction at a time (decision 0003).
//!
//! Every register dword is an `i32` (SCC an `i1`); float operations bitcast
//! their operands. Lane masks exist only as the per-lane booleans of
//! structurisation; branches and guards are built from them. A value the
//! emitter cannot express is `Missing` with a reason, and only fails the
//! program when something with an effect (an export, a branch) needs it.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::UserData;
use super::bitcode::{Attr, Bin, Cast, Module, Term, V, pred};
use super::container::SigElement;
use super::shader::{Shader, VIEW_INTEGER, VIEW_SIGNED, VIEW_STRIDE_BITS, VIEW_STRIDES, flags};
use crate::ShaderError;
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::io::{Barycentric, Elsewhere, Export, Input, Interp, Io, Target};
use crate::ir::{Block, EXEC, Exit, Inst, Phi, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo, Usage, UsedResource};
use crate::resource::{Address, CODE, MEMORY, Resources, Role, Site, Source, USER_DATA};
use crate::structure::{Atom, Bool, LaneExit, Map, Set, Structured};

mod alu;
mod bvh;
mod counted;
mod fence;
mod flow;
mod halves;
mod image;
mod mem;
mod operand;
mod sop;
pub(crate) use counted::{Extra, by_address};
pub(crate) use fence::writes_root_views;
pub use halves::kept_wide;
use halves::*;
pub(crate) use halves::{
    branches_around, count_wide, idle_skips, once_blocks, take_needs, take_wide,
};

type Why = &'static str;

#[derive(Debug, Clone, Copy)]
enum Val {
    I32(V),
    Bit(V),
    Missing(Why),
}

pub(crate) struct Lowered {
    pub dxil: Vec<u8>,
    /// Bit n: immediate sampler n is a comparison sampler.
    pub cmp_samplers: u64,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    pub root_uavs: Vec<u8>,
    /// Reads title memory at computed addresses (the page table).
    pub memory: bool,
    /// Writes there (the written-pages bitmap).
    pub memory_writes: bool,
    /// Reads buffer views' addresses or texture views' sizes from
    /// descriptors (the view addresses).
    pub view_addresses: bool,
    /// Takes descriptors from the resource heap itself (root type 5, not
    /// a table), and whether as unordered access views.
    pub heap: bool,
    pub heap_writes: bool,
    /// Marks the pages it writes through buffer views (`emit`, "Write
    /// marks").
    pub view_marks: bool,
    pub user_data: Vec<UserData>,
    pub samplers: Vec<[u32; 4]>,
    /// Mesh: per clip or cull distance output (its element), the channels
    /// every store gives +0.0: distances that clip and cull nothing.
    pub idle_distances: Vec<(usize, u8)>,
    /// Heap handles made with a non-uniform index.
    #[cfg(test)]
    pub nonuniform_handles: u32,
    /// Heap handles written.
    #[cfg(test)]
    pub heap_handles: u32,
    /// Whether the program uses wave operations.
    #[cfg(test)]
    pub wave_ops: bool,
    /// Whether it declares rasterizer-ordered views.
    #[cfg(test)]
    pub rovs: bool,
    /// Barriers added for narrow host waves ([`split_wave_barriers`]).
    #[cfg(test)]
    pub split_barriers: usize,
    /// The `WaveSize` the program declares.
    #[cfg(test)]
    pub wave_size: Option<u32>,
    /// Exchanges between the halves of console waves ([`Halves`]).
    #[cfg(test)]
    pub exchanges: u32,
    /// Group barriers written.
    #[cfg(test)]
    pub barriers: usize,
}

pub(crate) struct Parts<'a> {
    pub native: &'a NativeProgram,
    /// The container's `PSV0` part: the resources the console compiler
    /// declared ([`super::declares_structured_buffer`]).
    pub psv: Option<&'a [u8]>,
    pub p: &'a Program,
    pub st: &'a Structured,
    pub io: &'a Io,
    pub res: &'a Resources,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    /// Mesh: rows of vertex outputs left out of `outputs`
    /// ([`translate_keeping`](super::translate_keeping)); exports to them
    /// store nothing.
    pub dropped_rows: Vec<u32>,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
    /// Writes through buffer views mark their pages (`emit`, "Write marks").
    pub view_marks: bool,
    /// 64-bit image atomics run under per-texel locks (`emit`, "Atomic
    /// locks").
    pub atomic_locks: bool,
    /// Compute-stage biased samples run as gradient samples scaled by
    /// `2^bias` ([`Options::bias_as_gradients`](super::Options)).
    pub bias_as_gradients: bool,
    /// See [`Options::narrow_waves`](super::Options).
    pub narrow_waves: bool,
    /// A size query on a writable image asks for level 0, not an undefined
    /// level ([`Options::uav_size_level0`](super::Options)).
    pub uav_size_level0: bool,
    /// See [`Options::relax`](super::Options).
    pub relax: super::Relax,
}

enum Stop {
    /// A phi whose incoming value has no DXIL form was read: redo with it missing.
    Retry(Value, Why),
    Fail(ShaderError),
}

#[cfg(any(test, windows))]
pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    lower_with_layout(parts, false, false, false, Extra::default())
}

/// [`lower`], with the trial interface layout and the mesh store order
/// [`super::Options`] chooses; `barycentrics`: the barycentric input VGPRs
/// read as values are `SV_Barycentrics` (else reading one stops the
/// program with [`BARYCENTRICS_READ`]; only a program that cannot do
/// without them is made again with them, `super::emit_with`).
pub(crate) fn lower_with_layout(
    parts: &Parts,
    compact: bool,
    mesh_stores_last: bool,
    barycentrics: bool,
    extra: Extra,
) -> Result<Lowered, ShaderError> {
    // A wave64 compute program for narrow host waves is first made as
    // halves of console waves; one that form is not exact for is made as
    // before (and does not run on such a host).
    let refused = std::cell::Cell::new(None);
    NEEDS.with_borrow_mut(Vec::clear);
    for split in [true, false] {
        let mut bad = HashMap::new();
        let made = loop {
            refused.set(None);
            let mut l = Lower::new(parts, &bad, split, &refused);
            let halves = l.halves.is_some();
            l.mesh_stores_last = mesh_stores_last;
            l.barycentrics = barycentrics;
            l.extra = extra;
            match l.run(compact) {
                Ok(l) => break (Ok(l), halves),
                Err(Stop::Retry(v, why)) => {
                    if bad.insert(v, why).is_some() {
                        break (
                            Err(ShaderError::Unsupported {
                                offset: 0,
                                what: "a phi that does not settle",
                            }),
                            halves,
                        );
                    }
                }
                Err(Stop::Fail(e)) => break (Err(e), halves),
            }
        };
        match made {
            (r, false) => {
                if split && let Some(why) = refused.get() {
                    WIDE.set(Some(why));
                }
                return r;
            }
            (Ok(l), true) if refused.get().is_none() => return Ok(l),
            (r, _) => WIDE.set(Some(refused.get().unwrap_or(match r {
                Err(ShaderError::Unsupported { what, .. }) => what,
                _ => "a program the halves form does not translate",
            }))),
        }
    }
    unreachable!("the second form has no halves")
}

const TWO_PI: f32 = std::f32::consts::TAU;
/// Why a program that reads the barycentric input VGPRs as values stops
/// (before it runs again with `SV_Barycentrics`).
pub(crate) const BARYCENTRICS_READ: Why = "barycentrics read directly";
/// `s_sendmsg` id of a mesh or NGG stage's output allocation request.
const GS_ALLOC_REQ: u16 = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// A GFX10 buffer format (the unified format of typed buffer loads): the
/// channel widths from the lowest bits up, and how the channels convert.
/// Formats are numbered in families (`8`, `16`, `8_8`, ...: AMD names
/// list channels from the highest bits), each with its numeric formats in
/// the order unorm, snorm, uscaled, sscaled, uint, sint, float.
fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    // Numbers and order from LLVM's `UfmtGFX10` (`SIDefines.h`); the GFX10
    // buffer formats have no 5_9_9_9. Channel order from Mesa
    // (`ac_formats.c`): R11G11B10_FLOAT is 10_11_11, and 10_10_10_2 has
    // its 2-bit channel in the lowest bits.
    const FAMILIES: [(&[u32], &[Numeric]); 14] = [
        (&[8], NORM),
        (&[16], ALL),
        (&[8, 8], NORM),
        (&[32], WIDE),
        (&[16, 16], ALL),
        (&[11, 11, 10], ALL),     // 10_11_11
        (&[10, 11, 11], ALL),     // 11_11_10
        (&[2, 10, 10, 10], NORM), // 10_10_10_2
        (&[10, 10, 10, 2], NORM), // 2_10_10_10
        (&[8, 8, 8, 8], NORM),
        (&[32, 32], WIDE),
        (&[16, 16, 16, 16], ALL),
        (&[32, 32, 32], WIDE),
        (&[32, 32, 32, 32], WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

/// How far a float channel of `width` bits (16, or the unsigned 11- and
/// 10-bit floats of the packed formats) shifts up to be a half:
/// the small floats are a half's exponent (5 bits, biased 15) and the top
/// of its mantissa, without the sign, so `legacyF16ToF32` widens them
/// exactly, denormals, infinities and NaNs included.
fn half_shift(width: u32) -> u32 {
    debug_assert!(matches!(width, 10 | 11 | 16), "{width}-bit float");
    if width == 16 { 0 } else { 15 - width }
}

/// A descriptor load's address given by root parameters alone: root
/// parameter and byte offset.
type EntryAddress = (usize, u32);

struct Lower<'a> {
    s: Shader,
    x: &'a Parts<'a>,
    bad: &'a HashMap<Value, Why>,
    lane_fail: Map<Value, Why>,
    vals: Map<Value, Val>,
    floats: Map<Value, V>,
    /// Values (missing in `vals`) whose low 16 bits are known: a 16-bit
    /// result written over a value with no DXIL form keeps its high half.
    low16: Map<Value, V>,
    /// DXIL block per IR block (unreachable: none).
    block: Vec<Option<usize>>,
    cur: usize,
    bools: Map<(usize, Bool), V>,
    lane_bits: Map<Value, V>,
    /// IR phi location by value.
    phi_at: Map<Value, (usize, usize)>,
    mask_phis: Map<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    /// Phis that code reads. Left a std set: [`Lower::live_edges`] starts
    /// its walk in this set's iteration order, and few phis are read.
    read_phis: HashSet<Value>,
    params: Map<Value, (usize, u32)>,
    user_data: Option<V>,
    user_reads: BTreeMap<u32, UserData>,
    /// Handles of root SRVs and UAVs read as bounded views, by user-data
    /// dword ([`Lower::root_view`]).
    root_views: Map<u32, V>,
    samplers: Vec<[u32; 4]>,
    descriptors: Map<(Site, bool), &'a Source>,
    scalar: Map<Site, Address>,
    exports: Map<Site, &'a Export>,
    interps: Map<Site, &'a Interp>,
    /// The lists of `x.res.waterfall` as sets: they are tested for every
    /// VALU instruction, and a bindless shader has hundreds of compares.
    waterfall_compares: Set<Site>,
    waterfall_reads: Set<Value>,
    waterfall_loads: Set<Site>,
    handles: Map<(usize, Site, u8, bool, u32), V>,
    /// Handles made in the entry block for descriptors at addresses the
    /// root parameters alone give (table and heap loads with no other
    /// term), by base value, byte offset, dword, offset register, sampler
    /// heap and properties ([`Lower::entry_address`]).
    entry_handles: Map<(EntryAddress, u8, bool, u32), V>,
    /// Handles of descriptors chosen by control flow, made in their join
    /// block, by the choice (its `Debug` text), sampler heap and properties.
    join_handles: Map<(String, bool, u32), V>,
    /// Blocks splitting the edge from one IR block to another.
    edges: Map<(usize, usize), usize>,
    /// Per IR block: a `done` export is behind every lane entering it.
    exported: Vec<bool>,
    /// Reverse-postorder index per IR block.
    rpo: Vec<usize>,
    /// DXIL block each IR block ends in (guarded stores split blocks).
    tail: Vec<Option<usize>>,
    /// The program writes UAVs: barriers fence them too.
    writes_uavs: bool,
    /// [`Lowered::heap`], [`Lowered::heap_writes`].
    heap: [bool; 2],
    /// src0 of the current instruction as DPP fetched it.
    src0: Option<V>,
    /// Compute code that reads other lanes: thread ids follow the host
    /// lanes, so lane n is console lane n.
    console_lanes: bool,
    /// Pixel: an ordered section (POPS), whose UAVs the host makes
    /// rasterizer-ordered views.
    pops: bool,
    /// The program uses title memory (addresses it computes, the BVH
    /// instruction, root parameters through the user-data pointer): its root
    /// SRVs and UAVs are read there too ([`Lower::raw_view`]).
    root_memory: bool,
    /// Mesh: the thread index in the group (made in the entry block).
    flat: Option<V>,
    /// Mesh: IR blocks whose `s_barrier` is left out.
    no_barrier: Set<usize>,
    /// Group barriers before these instructions (index = length: at the
    /// block's end), for a one-wave group split across host waves
    /// ([`split_wave_barriers`]).
    split_barriers: Set<(usize, usize)>,
    /// The program runs as halves of console waves ([`Halves`]).
    halves: Option<Halves>,
    /// Halves in a group of several console waves: the values alike in
    /// every lane of a console wave ([`wave_alike`]), once asked for.
    alike: std::cell::OnceCell<Set<Value>>,
    /// Halves: the branches sunk below the code they skip ([`sunk_skips`]):
    /// the skipped block and the skip's target, by the branch's block.
    sunk: Map<usize, (usize, usize)>,
    /// The sunk branches met so far, by the skipped block: the condition,
    /// whether the skip is taken when it holds, and the DXIL block the
    /// skip's edge then goes through.
    sinking: Map<usize, (V, bool, usize)>,
    /// See [`Extra`].
    extra: Extra,
    /// The threads past a dispatch's counts of threads leave at the end of
    /// the entry block ([`Lower::leave_past_counts`]).
    counted: Option<counted::Counted>,
    /// Why the halves form is not the console's program, once it is known.
    refused: &'a std::cell::Cell<Option<Why>>,
    /// Guarded regions open ([`Self::begin_guarded`]): not every lane is
    /// in the DXIL block being written.
    guarded: u32,
    /// Vertex and mesh outputs holding 16-bit values: row, whether two per
    /// dword ([`packed_half`]).
    half_outputs: HashMap<u8, bool>,
    /// Mesh: an export's output stores all come after the values they
    /// store ([`Options::mesh_stores_last`](super::Options)).
    mesh_stores_last: bool,
    /// Barycentric input VGPRs read as values are `SV_Barycentrics`
    /// ([`lower_with_layout`]); else reading one stops the program.
    barycentrics: bool,
    /// How lane masks become data.
    mask_data: MaskData,
    /// The instruction being lowered may read lane masks as data.
    mask_ok: bool,
    /// Ballots of masks read as data, by low dword: (low, high).
    ballots: Map<Value, (V, V)>,
    /// SCC of a lane-width mask operation, "the result is not zero", by
    /// SCC value: the mask it tests (low dword, high dword in wave64).
    mask_sccs: Map<Value, (Value, Option<Value>)>,
    /// Masks whose ballot is being built: a mask whose lane bit needs its
    /// own value as a number (an opaque mask keyed by a dword it shares
    /// with another) would otherwise recurse until the stack overflows.
    balloting: Set<Value>,
    /// Block writing each value.
    def_block: Map<Value, usize>,
    /// IR block being lowered.
    cur_ir: usize,
    /// Whole-wave sections by block and start index.
    sections: HashMap<usize, HashMap<usize, super::wave::Section>>,
    /// Values written in a section that code outside it reads.
    section_out: Set<Value>,
    /// Values of [`Self::section_out`] that have a shape only per lane
    /// ([`super::wave::Section::lanes`]): the block and start of their
    /// section.
    section_lanes: Map<Value, (usize, usize)>,
    /// The program's constant table (the code from the first byte read
    /// through `s_getpc_b64` on): global, first byte, dwords.
    code_table: Option<(V, u32, u32)>,
    /// Loop headers' iteration counters (see [`LOOP_LIMIT`]): the phi by
    /// IR block, and its incoming (value, DXIL block) pairs.
    counters: HashMap<usize, (V, Vec<(V, usize)>)>,
    /// The block a loop stopped by [`LOOP_LIMIT`] leaves the program from.
    abort: Option<usize>,
    /// The program tests rays against tree nodes (`image_bvh*`): its loops
    /// are cut at [`super::trace_loop_limit`] when that is set.
    traces: bool,
    /// The 16/64-bit round mode (MODE bits 3:2) each block starts with, and
    /// the one in force at the instruction being lowered (see [`round16`]).
    round16: Vec<Option<u8>>,
    mode16: Option<u8>,
}

/// The 16/64-bit round mode (MODE bits 3:2: 0 nearest even, 3 toward zero)
/// after `inst`, from `mode` before it (`None`: not known).
fn round16_after(inst: &Inst, mode: Option<u8>) -> Option<u8> {
    match (inst.name(), &inst.fields) {
        ("s_round_mode", Fields::Sopp { imm }) => Some((imm >> 2 & 3) as u8),
        (
            "s_setreg_imm32_b32" | "s_setreg_b32",
            Fields::Sop {
                src, imm: Some(h), ..
            },
        ) => {
            let (id, off, size) = (h & 0x3F, h >> 6 & 0x1F, (h >> 11 & 0x1F) + 1);
            if id != 1 || off >= 4 || off + size <= 2 {
                return mode;
            }
            match src.first() {
                Some(&Operand::Literal(x)) if off <= 2 && off + size >= 4 => {
                    Some((x >> (2 - off) & 3) as u8)
                }
                _ => None,
            }
        }
        _ => mode,
    }
}

/// The 16/64-bit round mode each block starts with: nearest even at the
/// program's start, then as the MODE writes on the way set it; `None` where
/// paths that set it differently join (or a write is not a constant).
fn round16(p: &crate::ir::Program) -> Vec<Option<u8>> {
    // None: not reached yet; Some(m): the mode (m None: not known).
    let mut entry: Vec<Option<Option<u8>>> = vec![None; p.blocks.len()];
    entry[0] = Some(Some(0));
    let order = crate::ir::reverse_postorder(&p.blocks);
    loop {
        let mut changed = false;
        for &b in &order {
            let Some(mut m) = entry[b] else { continue };
            for inst in &p.blocks[b].insts {
                m = round16_after(inst, m);
            }
            for s in p.blocks[b].exit.successors() {
                let new = match entry[s] {
                    None => Some(m),
                    Some(old) if old == m => continue,
                    Some(_) => Some(None),
                };
                if entry[s] != new {
                    entry[s] = new;
                    changed = true;
                }
            }
        }
        if !changed {
            return entry.into_iter().map(Option::flatten).collect();
        }
    }
}

/// Iterations after which a loop is left (per entry into it), unless
/// [`super::loop_limit`] says otherwise. The console runs a loop as long as
/// its data says; data the host has not mirrored yet (a null view reads
/// zeros) can make that four billion iterations, which hangs the host GPU
/// until the driver resets it. A lane that reaches the limit leaves the
/// program as killed (a pixel discards). The limit is also what such a loop
/// costs: a dispatch lasts as long as its slowest lane, and one lane's 2^20
/// passes took 60 to 195 ms on an Apple M5 GPU, where 2^14 take 1 to 3 ms.
// shortcut: a genuine loop of more iterations than this is cut short too.
pub(super) const LOOP_LIMIT: u32 = 1 << 14;

/// Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt`, scalar
/// compares, bit tests) are ballots of the lanes' booleans, taken where the
/// mask is written: structurisation's `active` masks hold only lanes that
/// reach that block, so the ballot misses none.
/// A page table lookup ([`Lower::memory`]).
struct MemoryEntry {
    handle: V,
    offset: V,
    /// Block × 4096 + page in the block.
    entry: V,
    block: V,
    /// The level-1 index: address bits 28-39.
    l1: V,
    /// No host copy holds the page.
    miss: V,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskData {
    /// Compute: host lane n is console lane n (the program declares the
    /// console's wave size), so ballots are the console's masks bit for bit
    /// and any use is exact.
    Console,
    /// Pixel and vertex programs: ballots over the host's wave, whatever its
    /// size. Only uses that mean the same on any wave are allowed: counts,
    /// the first lane, the count below a lane, equality (the HLSL wave
    /// operations the console compiler expands). Lane numbers stay host
    /// lanes throughout (`v_mbcnt`, `v_readlane` at a computed lane).
    Host,
    /// Mesh programs: their lanes and waves (LDS per console wave) are the
    /// console's but no host wave size can be declared. Only a mask's bit
    /// count is taken, over the host's wave (the console compiler's test of
    /// a value for uniformity, `WaveActiveAllEqual`): lane numbers stay the
    /// console's, so the first lane and the count below a lane are refused.
    None,
}

/// Why a wave reduction refuses the halves form ([`Halves`]).
const REDUCTION: Why = "a wave reduction where the halves cannot meet";

/// Why a one-wave group of a wave64 program has no halves form: a read of
/// LDS in divergent code may see what other lanes wrote there since the
/// last place a barrier could go.
const LDS_APART: Why = "LDS read where the halves cannot meet";

/// Why a program with branches taken wave-wide has no halves form: a
/// scalar branch where its lanes are together tests what lanes may hold
/// differently ([`scalar_parting`]), so that the halves could part there.
const SCALAR_APART: Why = "a scalar branch the halves may take apart";

/// What a value holding a lane mask read as a number stops with.
const LANE_MASK: Why = "a lane mask read as data";

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// Whether code other than a branch reads the SCC of a lane-width mask
/// operation whose result is a mask register: "some bit is set", the
/// console wave's decision read as a scalar (a poll's "some lane is not
/// ready", selected into an all-or-none mask), which needs the console's
/// wave ([`Lower::mask_sccs`]).
fn reads_mask_scc(x: &Parts) -> bool {
    let mut sccs = Set::default();
    for i in x.p.blocks.iter().flat_map(|b| &b.insts) {
        if let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            ..
        } = &i.fields
            && let (Some(d), Some(scc)) = (i.write(Reg::S(*code)), i.write(Reg::Scc))
            && i.write(Reg::S(EXEC)).is_none()
            && x.st.masks.contains_key(&d)
        {
            sccs.insert(scc);
        }
    }
    !sccs.is_empty()
        && x.p.blocks.iter().flat_map(|b| &b.insts).any(|i| {
            !matches!(i.fields, Fields::Sopp { .. })
                && i.read(Reg::Scc).is_some_and(|v| sccs.contains(&v))
        })
}

/// Where half `h` (dword h / 2, the high half when odd) of the parameter
/// whose first element is in row `a` comes from, when the console packs
/// 16-bit values two per dword: a row and channel of `elements`, none for
/// padding. The console fills its parameters with the 16-bit elements taken
/// by first channel, then row: each takes whole dwords of the first
/// parameter with room for it (four dwords each), its channels in order.
fn packed_half(elements: &[SigElement], a: u8, h: u8) -> Option<(u8, u8)> {
    let mut halves: Vec<&SigElement> = elements
        .iter()
        .filter(|e| e.half && e.kind == 0 && e.row.is_some())
        .collect();
    halves.sort_by_key(|e| (e.col, e.row));
    // (first element's row, (row, channel) per half)
    type Param = (Option<u32>, Vec<Option<(u8, u8)>>);
    let mut params: Vec<Param> = Vec::new();
    for e in halves {
        let need = usize::from(e.cols.div_ceil(2) * 2);
        let i = match params.iter().position(|p| p.1.len() + need <= 8) {
            Some(i) => i,
            None => {
                params.push((e.row, Vec::new()));
                params.len() - 1
            }
        };
        let row = u8::try_from(e.row?).ok()?;
        let slots = &mut params[i].1;
        slots.extend((0..e.cols).map(|c| Some((row, e.col + c))));
        slots.resize(slots.len().next_multiple_of(2), None);
    }
    // Two parameters starting in one row would leave the half ambiguous.
    let mut found = params.iter().filter(|p| p.0 == Some(u32::from(a)));
    match (found.next(), found.next()) {
        (Some(p), None) => p.1.get(usize::from(h)).copied().flatten(),
        _ => None,
    }
}

/// Vertex or mesh outputs of 16-bit values (see [`Lower::half_outputs`]),
/// by row: whether two per dword.
fn half_outputs(io: &StageIo) -> HashMap<u8, bool> {
    let entries = match io {
        StageIo::Vs { outputs, .. } => outputs,
        StageIo::Ms(ms) => &ms.outputs,
        _ => return HashMap::new(),
    };
    entries
        .iter()
        .filter(|o| o.param & 0x80 == 0 && o.fp16)
        .map(|o| (o.row, o.pack2))
        .collect()
}

fn unsupported(offset: usize, what: Why) -> Stop {
    Stop::Fail(ShaderError::Unsupported { offset, what })
}

/// The scalar instruction defining an SSA value in this block and its operands.
fn scalar_def<'a, const N: usize>(
    block: &'a Block,
    value: Value,
    op: &str,
) -> Option<(&'a Inst, [Operand; N])> {
    let inst = block
        .insts
        .iter()
        .find(|i| i.writes.iter().any(|w| w.1 == value))?;
    let Fields::Sop { src, .. } = &inst.fields else {
        return None;
    };
    (inst.name() == op).then_some((inst, src.as_slice().try_into().ok()?))
}

fn sgpr_operand(inst: &Inst, operand: Operand) -> Option<Value> {
    match operand {
        Operand::Scalar {
            code: code @ 0..=101,
            dwords: 1,
        } => inst.read(Reg::S(code)),
        _ => None,
    }
}

fn scalar_constant(operand: Operand) -> Option<u32> {
    match operand {
        Operand::Const(code) => crate::resource::inline(code),
        Operand::Literal(value) => Some(value),
        _ => None,
    }
}

/// The dwords an `s_load_dword*` reads.
fn smem_dwords(name: &str) -> Option<u32> {
    match name.strip_prefix("s_load_dword")? {
        "" => Some(1),
        "x2" => Some(2),
        "x4" => Some(4),
        "x8" => Some(8),
        "x16" => Some(16),
        _ => None,
    }
}

impl<'a> Lower<'a> {
    fn new(
        x: &'a Parts<'a>,
        bad: &'a HashMap<Value, Why>,
        split: bool,
        refused: &'a std::cell::Cell<Option<Why>>,
    ) -> Self {
        let p = x.p;
        let mut phi_at = Map::default();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phi_at.insert(phi.value, (b, i));
            }
        }
        let mut s = Shader::new(x.native.stage == HwStage::Ps);
        s.inputs = x.inputs.clone();
        s.outputs = x.outputs.clone();
        let insts = || p.blocks.iter().flat_map(|b| &b.insts);
        let writes_uavs = insts().any(|i| {
            matches!(i.fields, Fields::Image { .. } | Fields::Buffer { .. })
                && (i.name().contains("store") || i.name().contains("atomic"))
        });
        let root_memory = x
            .res
            .scalar
            .iter()
            .any(|s| matches!(s.1.param, MEMORY | USER_DATA))
            || insts().any(|i| i.name().starts_with("image_bvh"));
        let mut console_lanes = false;
        let mut split_barriers = Set::default();
        let mut halves = None;
        if let StageIo::Cs(cs) = x.native.io {
            s.threads = Some(cs.threads);
            s.lds_dwords = cs.lds_bytes() / 4;
            // Code reading other lanes needs the console's wave size and
            // lane order; so do lane masks read as numbers, and LDS shared
            // within one wave, where the console compiler leaves out barriers.
            console_lanes = insts().any(|i| {
                let n = i.name();
                n.starts_with("ds_swizzle")
                    || n.contains("permute")
                    || n.starts_with("v_mbcnt")
                    || matches!(n, "v_readlane_b32" | "v_writelane_b32")
                    || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
                    || reads_mask_as_data(x.st, i)
            }) || {
                // Masks held as numbers or constant lane sets: the lane
                // tests its own bit by its console lane.
                let whens = x.st.exits.iter().filter_map(|e| match *e {
                    LaneExit::Branch { when, .. } => Some(when),
                    _ => None,
                });
                let mut all: Vec<Bool> =
                    x.st.guards
                        .iter()
                        .flatten()
                        .flatten()
                        .copied()
                        .chain(whens)
                        .collect();
                all.sort_unstable();
                all.dedup();
                all.into_iter().any(|g| {
                    x.st.bools
                        .support(g)
                        .iter()
                        .any(|a| matches!(a, Atom::Opaque(_) | Atom::Constant(_)))
                })
            };
            console_lanes |= reads_mask_scc(x);
            let wave = if x.native.wave64 { 64 } else { 32 };
            let lanes = cs.threads.iter().product::<u32>();
            let one_wave = lanes <= wave;
            // Where the threads of a group meet at a barrier: with one
            // console wave, wherever the wave's lanes are all together;
            // with several, only where every lane runs once.
            let meet = |live: bool| {
                if one_wave {
                    wave_blocks(x, live)
                } else {
                    once_blocks(p)
                }
            };
            // The console's one wave is several host waves: LDS no longer
            // passes between its lanes without a barrier.
            let mut synced = false;
            if x.narrow_waves && one_wave && lanes > 32 && s.lds_dwords > 0 {
                // shortcut: a read in divergent code of what divergent code
                // wrote gets no barrier (none could be placed there); the
                // program then runs as before.
                match split_wave_barriers_in(p, &meet(true)) {
                    Ok(b) => {
                        split_barriers = b;
                        synced = true;
                    }
                    // (the halves form is what was asked for: why not, and
                    // where the halves would have to meet)
                    Err(apart) if split && x.native.wave64 => {
                        refused.set(Some(LDS_APART));
                        NEEDS.with_borrow_mut(|n| n.extend(apart));
                    }
                    Err(_) => {}
                }
            }
            // A host whose waves are narrower than WaveSize(64) does not run
            // the dispatch (D3DMetal: nothing written): ask for the console's
            // width only where wave operations need it, or LDS is not synced.
            let fits = x.narrow_waves && (synced || lanes <= 32);
            if console_lanes || one_wave && s.lds_dwords > 0 && !fits {
                s.wave_size = Some(wave);
            }
            // Cross-lane code of a wave64 program in 32-lane host waves:
            // whole console waves in a group (or one host wave), LDS synced.
            let lds_fits = fits || !one_wave || s.lds_dwords == 0;
            if split
                && x.narrow_waves
                && console_lanes
                && x.native.wave64
                && lds_fits
                && (lanes <= 32 || lanes % 64 == 0)
            {
                s.wave_size = Some(32);
                halves = Some(Halves {
                    whole: lanes <= 32,
                    count: lanes.div_ceil(32),
                    full: meet(false),
                    meet: Vec::new(),
                    once: Vec::new(),
                    solo: solo_blocks(x, cs.threads, cs.thread_id_vgprs()),
                    flat: None,
                    sites: 0,
                });
            }
        }
        let mut no_barrier = Set::default();
        if let StageIo::Ms(ms) = &x.native.io {
            s.prims = x.prims.clone();
            s.threads = Some([ms.threads, 1, 1]);
            s.lds_dwords = ms.lds_bytes.div_ceil(4);
            s.mesh = Some(super::shader::Mesh {
                max_vertices: ms.max_vertices,
                max_primitives: ms.max_primitives,
                topology: ms.topology,
            });
            // Barriers order the lanes' LDS accesses; the console's
            // allocation request needs them too, but the host does not. One
            // lane sends that request, and what it dominates (the path for
            // counts out of range) runs in that lane alone, so its barrier
            // would hang the host.
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut index = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                index[b] = n;
            }
            let idom = crate::ir::idoms(&p.blocks, &order, &index);
            let alloc: Vec<usize> = (0..p.blocks.len())
                .filter(|&b| {
                    p.blocks[b].insts.iter().any(|i| {
                        i.name() == "s_sendmsg"
                            && matches!(i.fields, Fields::Sopp { imm } if imm & 0xF == GS_ALLOC_REQ)
                    })
                })
                .collect();
            for &b in &order {
                let mut d = b;
                let dominated = loop {
                    if alloc.contains(&d) {
                        break true;
                    }
                    if idom[d] == d || idom[d] == usize::MAX {
                        break false;
                    }
                    d = idom[d];
                };
                if ms.lds_bytes == 0 || dominated {
                    no_barrier.insert(b);
                }
            }
        }
        let mut l = Lower {
            s,
            x,
            bad,
            lane_fail: Map::default(),
            vals: Map::default(),
            floats: Map::default(),
            low16: Map::default(),
            block: vec![None; p.blocks.len()],
            cur: 0,
            bools: Map::default(),
            lane_bits: Map::default(),
            phi_at,
            mask_phis: Map::default(),
            pending_masks: Vec::new(),
            data_phis: Vec::new(),
            read_phis: HashSet::new(),
            params: x.res.params.iter().map(|&(v, p, d)| (v, (p, d))).collect(),
            user_data: None,
            user_reads: BTreeMap::new(),
            root_views: Map::default(),
            samplers: Vec::new(),
            descriptors: x
                .res
                .descriptors
                .iter()
                .map(|d| ((d.site, d.role == Role::Sampler), &d.source))
                .collect(),
            scalar: x.res.scalar.iter().copied().collect(),
            exports: x.io.exports.iter().map(|e| (e.site, e)).collect(),
            interps: x.io.interps.iter().map(|(s, i)| (*s, i)).collect(),
            waterfall_compares: x.res.waterfall.compares.iter().copied().collect(),
            waterfall_reads: x.res.waterfall.reads.iter().copied().collect(),
            waterfall_loads: x.res.waterfall.loads.iter().copied().collect(),
            handles: Map::default(),
            entry_handles: Map::default(),
            join_handles: Map::default(),
            edges: Map::default(),
            exported: vec![false; p.blocks.len()],
            rpo: vec![usize::MAX; p.blocks.len()],
            tail: vec![None; p.blocks.len()],
            writes_uavs,
            heap: [false; 2],
            src0: None,
            console_lanes,
            pops: matches!(x.native.io, StageIo::Ps { db_shader_control, .. }
                if db_shader_control & 1 << 16 != 0),
            root_memory,
            flat: None,
            no_barrier,
            split_barriers,
            halves,
            alike: std::cell::OnceCell::new(),
            sunk: Map::default(),
            sinking: Map::default(),
            extra: Extra::default(),
            counted: None,
            refused,
            guarded: 0,
            half_outputs: half_outputs(&x.native.io),
            mesh_stores_last: false,
            barycentrics: false,
            mask_data: match x.native.stage {
                HwStage::Cs => MaskData::Console,
                HwStage::Ps | HwStage::Vs => MaskData::Host,
                _ => MaskData::None,
            },
            mask_ok: false,
            ballots: Map::default(),
            mask_sccs: Map::default(),
            balloting: Set::default(),
            def_block: p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| {
                    block
                        .insts
                        .iter()
                        .flat_map(move |i| i.writes.iter().map(move |w| (w.1, b)))
                        .chain(block.phis.iter().map(move |phi| (phi.value, b)))
                })
                .collect(),
            cur_ir: 0,
            sections: HashMap::new(),
            section_out: Set::default(),
            section_lanes: Map::default(),
            code_table: None,
            counters: HashMap::new(),
            abort: None,
            traces: p.blocks.iter().any(|b| b.insts.iter().any(|i| i.name().starts_with("image_bvh"))),
            round16: round16(p),
            mode16: Some(0),
        };
        // Several console waves in a group: code one wave skips and another
        // runs is run by all, so that the halves can meet in it.
        // (not where all waves go one way: the group meets there as it is)
        if l.halves.as_ref().is_some_and(|h| h.count > 2) {
            let same = l.group_uniform();
            l.sunk = sunk_skips(x);
            l.sunk.retain(|&d, _| match x.st.exits[d] {
                LaneExit::Branch { when, .. } => !x
                    .st
                    .bools
                    .support(when)
                    .iter()
                    .all(|a| matches!(a, Atom::Uniform(scc) if same.contains(scc))),
                _ => false,
            });
        }
        // Halves: the blocks all threads of a group run together. They are
        // where the halves meet in a group of several console waves; an
        // exchange in one outside every loop is made once.
        let group = l
            .halves
            .is_some()
            .then(|| (l.group_blocks(false), l.group_blocks(true)));
        if let (Some(h), Some((all, live))) = (&mut l.halves, group) {
            if h.count > 2 {
                h.full = all;
            }
            h.meet = h.full.iter().zip(&live).map(|(&f, &l)| f || l).collect();
            let cycle = in_cycle(p);
            h.once = live.iter().zip(&cycle).map(|(&l, &c)| l && !c).collect();
        }
        // Branches taken wave-wide: the halves go the same way only where
        // the scalar branches they meet at test what both hold alike.
        if split
            && !x.st.wide.is_empty()
            && !scalar_parting(p, x.st, &wave_blocks(x, true)).is_empty()
        {
            refused.set(Some(SCALAR_APART));
        }
        l
    }

    /// The scalar values every thread of a compute group holds alike: made
    /// by scalar instructions of the SGPRs a group starts with (the user
    /// SGPRs and the group's id), constants, and loads of memory the
    /// dispatch does not write (descriptor tables and root constants;
    /// buffers bound as constants or read-only, which the title cannot have
    /// bound for writing in the same dispatch; tables in the code). Lane
    /// masks, EXEC and whatever comes from a VGPR or a wave are left out.
    ///
    /// A join is one where every thread comes to it the same way: of such
    /// values, in a block all threads that have not ended run together
    /// ([`Self::together`]) and enter from such blocks only, so that the
    /// block a thread came from is the one all came from. A loop's count is
    /// such a join of itself: all are taken to hold alike, and those that
    /// do not hold on that footing are taken out until the rest do. (By
    /// the order the threads run those blocks in, which is one order for
    /// all: the first value or branch to differ would have been made of
    /// values that did not.)
    fn group_uniform(&self) -> Set<Value> {
        let p = self.x.p;
        let StageIo::Cs(cs) = self.x.native.io else {
            return Set::default();
        };
        let (ids, _) = cs.system_sgprs();
        const NOT: [&str; 12] = [
            "saveexec", "getpc", "setpc", "swappc", "rfe", "getreg", "setreg", "gpr_idx", "wqm",
            "quadmask", "call", "movrel",
        ];
        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut joins: Vec<(usize, &Phi)> = order
            .iter()
            .flat_map(|&b| p.blocks[b].phis.iter().map(move |phi| (b, phi)))
            .filter(|(_, phi)| {
                matches!(phi.reg, Reg::S(c) if c <= 105)
                    && !self.x.st.masks.contains_key(&phi.value)
            })
            .collect();
        loop {
            let mut same: Set<Value> = joins.iter().map(|j| j.1.value).collect();
            for &(reg, v) in &p.inputs {
                if let Reg::S(c) = reg
                    && (c < cs.user_sgprs() || ids.contains(&Some(c)))
                {
                    same.insert(v);
                }
            }
            for &b in &order {
                for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                    let name = inst.name();
                    let pure = match &inst.fields {
                        Fields::Sop { .. } => !NOT.iter().any(|n| name.contains(n)),
                        Fields::Smem { .. } => self.static_load(inst, (b, i)),
                        _ => false,
                    };
                    if pure && inst.reads.iter().all(|r| same.contains(&r.1)) {
                        same.extend(inst.writes.iter().map(|w| w.1));
                    }
                }
            }
            if joins.is_empty() {
                return same;
            }
            // (a join below a sunk branch is each wave's own)
            let together = self.together(&same, true, false);
            let before = joins.len();
            joins.retain(|&(b, phi)| {
                together[b]
                    && p.blocks[b].preds.iter().all(|&q| together[q])
                    && phi.args.iter().all(|a| same.contains(a))
            });
            if joins.len() == before {
                return same;
            }
        }
    }

    /// Whether a scalar load reads memory the dispatch does not write.
    fn static_load(&self, inst: &Inst, site: Site) -> bool {
        let name = inst.name();
        if name.starts_with("s_load_dword") {
            return self.scalar.get(&site).is_some_and(|a| a.param != MEMORY);
        }
        if !name.starts_with("s_buffer_load_dword") {
            return false;
        }
        let Some(&src) = self.descriptors.get(&(site, false)) else {
            return false;
        };
        // (a root view other than of constants is title memory in a program
        // that uses it, which the program may write)
        let memory = self.root_memory
            && matches!(src, Source::Built { address, .. }
                if self.x.native.layout.get(address.param)
                    .is_some_and(|p| p.root == RootType::Descriptor));
        matches!(src, Source::Built { address, .. } if address.param == CODE)
            || match self.used_kind(src) {
                Some(u) => u.kind == ResourceType::Cbv || u.kind == ResourceType::Srv && !memory,
                None => false,
            }
    }

    /// The blocks all threads of a compute group run together, each as
    /// often as the others: every branch that decides whether or how often
    /// a thread runs them (directly or through the branches around it)
    /// tests an SCC of `same`, values all threads hold alike, so all take
    /// it one way. A group barrier there has every thread of the group, as
    /// in a block each runs once ([`once_blocks`], which these include),
    /// each pass of a loop all go round together too.
    ///
    /// `live`: all threads that have not ended, that is: a branch whose
    /// other way is the program's end does not count (as [`wave_blocks`]:
    /// the threads that ended never reach the barrier; Apple's GPUs make
    /// the rest meet).
    ///
    /// `sunk`: a branch only taken after the code it skips
    /// ([`Self::sunk`]) decides nothing either.
    fn together(&self, same: &Set<Value>, live: bool, sunk: bool) -> Vec<bool> {
        let p = self.x.p;
        let st = self.x.st;
        let succ = |b: usize| p.blocks[b].exit.successors();
        let decides = |q: usize| match st.exits[q] {
            _ if sunk && self.sunk.contains_key(&q) => false,
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                taken != not_taken
                    && !st
                        .bools
                        .support(when)
                        .iter()
                        .all(|a| matches!(a, Atom::Uniform(scc) if same.contains(scc)))
            }
            LaneExit::Indirect => true,
            _ => succ(q).len() > 1,
        };
        let early = |q: usize| live && succ(q).iter().any(|&s| ends(p, s));
        undecided(p, |q| decides(q) && !early(q))
    }

    /// [`Self::together`] of the [`Self::group_uniform`] values.
    fn group_blocks(&self, live: bool) -> Vec<bool> {
        self.together(&self.group_uniform(), live, true)
    }

    /// The dword of the program's code at byte `byte` (a constant table
    /// the compiler appends to the code), as an internal constant array.
    // shortcut: an index outside the table reads its last dword (the
    // console would read whatever memory follows the code).
    fn code_dword(&mut self, byte: V) -> Result<V, Why> {
        let (g, base, len) = match self.code_table {
            Some(t) => t,
            None => {
                let first = self
                    .x
                    .res
                    .scalar
                    .iter()
                    .map(|s| s.1)
                    .chain(self.x.res.descriptors.iter().flat_map(|d| {
                        d.source.leaves().into_iter().filter_map(|l| match l {
                            Source::Built { address, .. } => Some(*address),
                            _ => None,
                        })
                    }))
                    .filter(|a| a.param == CODE)
                    .map(|a| a.offset & !3)
                    .min()
                    .ok_or("a code table")?;
                let code = &self.x.native.code;
                let words: Vec<u32> = code
                    .get(first as usize..)
                    .ok_or("a code table past the code")?
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        u32::from_le_bytes(w)
                    })
                    .collect();
                if words.is_empty() {
                    return Err("a code table past the code");
                }
                let elems: Vec<V> = words.iter().map(|&w| self.s.m.i32(w)).collect();
                let arr = self.s.m.array(words.len() as u32, self.s.t.i32);
                let init = self.s.m.aggregate(arr, &elems);
                let g = self.s.m.constant_global("code", init);
                let t = (g, first, words.len() as u32);
                self.code_table = Some(t);
                t
            }
        };
        let c = self.cur;
        let k = self.s.m.i32(base);
        let rel = self.s.m.bin(c, Bin::Sub, byte, k);
        let two = self.s.m.i32(2);
        let index = self.s.m.bin(c, Bin::LShr, rel, two);
        let last = self.s.m.i32(len - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let at = self.s.m.element(c, g, index);
        Ok(self.s.m.load(c, at))
    }

    /// A global memory access (`global_*`): loads from the program's
    /// constant table, from a root view at the address the title sets, or
    /// from title memory at an address the program computes; stores and
    /// atomics through a root view.
    fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if !name.starts_with("global_load_") {
            return self.flat_write(inst, b, i);
        }
        let rest = &name["global_load_".len()..];
        let (n, sub) = match rest {
            "dword" => (1, None),
            "dwordx2" => (2, None),
            "dwordx3" => (3, None),
            "dwordx4" => (4, None),
            "ubyte" => (1, Some((8, false))),
            "sbyte" => (1, Some((8, true))),
            "ushort" => (1, Some((16, false))),
            "sshort" => (1, Some((16, true))),
            _ => return Err(name),
        };
        let a = *self
            .scalar
            .get(&(b, i))
            .ok_or("an untracked global address")?;
        let guard = self.guard(b, i)?;
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        let c = self.cur;
        let bytes = sub.map_or(4 * n, |(width, _)| width / 8);
        let (h, off) = if a.param == MEMORY {
            let (lo, hi) = self.global_address(inst)?;
            let (h, off) = self.memory(lo, hi, bytes, false, guard);
            (Some(h), off)
        } else {
            let mut off = self.s.m.i32(a.offset);
            if let Some(d) = a.dynamic {
                let d = self.i32_of(d)?;
                off = self.s.m.bin(c, Bin::Add, off, d);
            }
            if a.param == CODE {
                (None, off)
            } else {
                let src = Source::Built {
                    address: a,
                    bits: 0,
                    words: [0, 0],
                };
                let (h, off) = self.raw_view((b, i), &src, off, bytes, (11, 0), false, guard)?;
                (Some(h), off)
            }
        };
        let c = self.cur;
        let dword = |l: &mut Self, byte: V| -> Result<V, Why> {
            match h {
                None => l.code_dword(byte),
                Some(h) => {
                    let c = l.cur;
                    let (u, mask, al) = (l.s.m.undef(l.s.t.i32), l.s.m.i8(1), l.s.m.i32(4));
                    let r = l.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        l.s.t.res_ret,
                        139,
                        &[h, byte, u, mask, al],
                    );
                    let x = l.s.m.extract(c, r, 0);
                    Ok(l.as_int(x))
                }
            }
        };
        if let (None, Some(h), true) = (sub, h, n > 1 && Self::wide_loads()) {
            for (k, x) in self.raw_dwords(h, off, n).into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        for k in 0..n {
            let x = match sub {
                None => {
                    let four = self.s.m.i32(4 * k);
                    let at = self.s.m.bin(c, Bin::Add, off, four);
                    dword(self, at)?
                }
                Some((width, signed)) => {
                    // shortcut: the field is taken not to cross a dword.
                    let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
                    let at = self.s.m.bin(c, Bin::And, off, not3);
                    let w = dword(self, at)?;
                    let byte = self.s.m.bin(c, Bin::And, off, three);
                    let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                    let x = self.s.m.bin(c, Bin::LShr, w, sh);
                    let left = self.s.m.i32(32 - width);
                    let x = self.s.m.bin(c, Bin::Shl, x, left);
                    self.s
                        .m
                        .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left)
                }
            };
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// The 64-bit address a `global_*` instruction accesses in each lane:
    /// its SGPR pair plus its VGPR offset, or its VGPR pair, plus its
    /// immediate; the high dword cut to address bits 32-47.
    fn global_address(&mut self, inst: &Inst) -> Result<(V, V), Why> {
        const WHY: Why = "a global address";
        let Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            saddr,
            offset,
            ..
        } = inst.fields
        else {
            return Err(WHY);
        };
        let read = |l: &mut Self, r: Reg| -> Result<V, Why> {
            let v = inst.read(r).ok_or(WHY)?;
            l.i32_of(v)
        };
        let (lo, hi) = match saddr {
            Some(Operand::Scalar { code, .. }) if code != NULL => {
                let (lo, hi) = (read(self, Reg::S(code))?, read(self, Reg::S(code + 1))?);
                let x = read(self, Reg::V(reg))?;
                self.add64(lo, hi, x, false)
            }
            _ => (read(self, Reg::V(reg))?, read(self, Reg::V(reg + 1))?),
        };
        let k = self.s.m.i32(offset as i32 as u32);
        let (lo, hi) = self.add64(lo, hi, k, offset < 0);
        let mask = self.s.m.i32(0xFFFF);
        Ok((lo, self.s.m.bin(self.cur, Bin::And, hi, mask)))
    }

    /// A global store or atomic (`global_store_*`, `global_atomic_*`)
    /// through a root view at the address the title sets, or into title
    /// memory at an address the program computes.
    fn flat_write(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat {
            dst, data, cache, ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        let a = *self
            .scalar
            .get(&(b, i))
            .ok_or("an untracked global address")?;
        if a.param == CODE {
            return Err("a global store into the code");
        }
        let n = match name.strip_prefix("global_store_") {
            None => 1,
            Some("dword") => 1,
            Some("dwordx2") => 2,
            Some("dwordx3") => 3,
            Some("dwordx4") => 4,
            Some(_) => return Err(name),
        };
        // DXIL atomic operations: add, and, or, xor, signed and unsigned min
        // and max, exchange; subtraction adds the negation. A float minimum
        // or maximum is the signed integer one where the value is positive
        // and the reversed unsigned one where negative (IEEE order of the
        // bits), as two atomics, each a no-op in the lanes the other serves.
        let ops: Option<&[(u32, u32)]> = match name.strip_prefix("global_atomic_") {
            None => None,
            Some("add" | "sub") => Some(&[(0, 0)]),
            Some("and") => Some(&[(1, 0)]),
            Some("or") => Some(&[(2, 0)]),
            Some("xor") => Some(&[(3, 0)]),
            Some("smin") => Some(&[(4, 0)]),
            Some("smax") => Some(&[(5, 0)]),
            Some("umin") => Some(&[(6, 0)]),
            Some("umax") => Some(&[(7, 0)]),
            Some("swap") => Some(&[(8, 0)]),
            Some("fmin") if !cache.glc => Some(&[(4, i32::MAX as u32), (7, 0)]),
            Some("fmax") if !cache.glc => Some(&[(5, i32::MIN as u32), (6, u32::MAX)]),
            Some(_) => return Err(name),
        };
        if ops.is_none() && !name.starts_with("global_store_") {
            return Err(name);
        }
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let d = data.as_ref().ok_or("global store data")?;
        let vals = self.vgprs(inst, d, n)?;
        let (memory, root) = if a.param == MEMORY {
            (Some(self.global_address(inst)?), None)
        } else {
            let mut off = self.s.m.i32(a.offset);
            if let Some(d) = a.dynamic {
                let d = self.i32_of(d)?;
                off = self.s.m.bin(c, Bin::Add, off, d);
            }
            let src = Source::Built {
                address: a,
                bits: 0,
                words: [0, 0],
            };
            match self.root_address(&src) {
                Some((lo, hi)) => {
                    // As `raw_view`, looked up inside the guard below.
                    let (lo, hi) = self.add64(lo, hi, off, false);
                    let mask = self.s.m.i32(0xFFFF);
                    (Some((lo, self.s.m.bin(self.cur, Bin::And, hi, mask))), None)
                }
                None => {
                    let h = self.handle((b, i), &src, false, (11 | 0x1000, 0))?;
                    let off = match self.root_view_offset(&src) {
                        Some(at) => self.s.m.bin(self.cur, Bin::Add, off, at),
                        None => off,
                    };
                    (None, Some((h, off)))
                }
            }
        };
        let (iu, void, i32t) = (self.s.m.undef(self.s.t.i32), self.s.t.void, self.s.t.i32);
        // Title memory is looked up (and its pages marked written) only in
        // the lanes that write.
        let at = self.begin_guarded(guard);
        let (h, off) = match (memory, root) {
            (Some((lo, hi)), _) => self.memory(lo, hi, 4 * u32::from(n), true, None),
            (_, Some(r)) => r,
            _ => unreachable!(),
        };
        if name.starts_with("global_store_") {
            let mut args = vec![h, off, iu];
            args.extend(&vals);
            args.resize(3 + 4, iu);
            args.push(self.s.m.i8((1u8 << n) - 1));
            args.push(self.s.m.i32(4));
            self.s.op(
                self.cur,
                "rawBufferStore.i32",
                Attr::NoUnwind,
                void,
                140,
                &args,
            );
            self.mark_view_write(h, off, 4 * u32::from(n), None);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let ops = ops.expect("an atomic");
        let mut x = vals[0];
        let c = self.cur;
        if name == "global_atomic_sub" {
            let zero = self.s.m.i32(0);
            x = self.s.m.bin(c, Bin::Sub, zero, x);
        }
        let negative = {
            let zero = self.s.m.i32(0);
            self.s.m.cmp(c, pred::SLT, x, zero)
        };
        let returned = match *dst {
            Some(Operand::Vector { reg, .. }) if cache.glc => Some(reg),
            _ => None,
        };
        let mut first = None;
        for (k, &(op, neutral)) in ops.iter().enumerate() {
            let value = if ops.len() == 1 {
                x
            } else {
                let n = self.s.m.i32(neutral);
                // The first serves positive values, the second negative ones.
                if k == 0 {
                    self.s.m.select(self.cur, negative, n, x)
                } else {
                    self.s.m.select(self.cur, negative, x, n)
                }
            };
            let op = self.s.m.i32(op);
            let r = self.s.op(
                self.cur,
                "atomicBinOp.i32",
                Attr::NoUnwind,
                i32t,
                78,
                &[h, op, off, iu, iu, value],
            );
            first.get_or_insert(r);
        }
        self.mark_view_write(h, off, 4, None);
        let zero = self.s.m.i32(0);
        let r = self.end_guarded(at, &[(first.expect("an atomic"), zero)])[0];
        if let Some(reg) = returned {
            self.write(inst, Reg::V(reg), r, guard);
        }
        Ok(())
    }

    /// Finds the whole-wave sections and the values they leave.
    fn find_sections(&mut self) {
        let p = self.x.p;
        let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
        for b in 0..p.blocks.len() {
            let found = super::wave::sections(p, b);
            for (&start, sec) in &found {
                for i in start + 1..sec.end {
                    for &(_, w) in &p.blocks[b].insts[i].writes {
                        inside.insert(w, (b, start, sec.end));
                    }
                }
            }
            if !found.is_empty() {
                self.sections.insert(b, found);
            }
        }
        if inside.is_empty() {
            return;
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                self.section_out
                    .extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && !(sb == b && i > start && i < end)
                    {
                        self.section_out.insert(r);
                    }
                }
            }
        }
    }

    /// An instruction inside a whole-wave section: the values that leave
    /// the section are made from what the simulation found them to be.
    fn section_inst(&mut self, b: usize, i: usize, start: usize) {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        for &(_, w) in &inst.writes {
            let val = if self.section_out.contains(&w) {
                if self.sections[&b][&start].lanes.contains_key(&w) {
                    self.section_lanes.insert(w, (b, start));
                }
                let shape = self.sections[&b][&start].shapes.get(&w).copied();
                match shape.ok_or(UNKNOWN).and_then(|sh| self.shape(sh)) {
                    Ok(x) => Val::I32(x),
                    Err(why) => Val::Missing(why),
                }
            } else {
                Val::Missing(UNKNOWN)
            };
            self.vals.insert(w, val);
        }
    }

    /// A value a whole-wave section leaves, as DXIL wave operations over the
    /// host's active lanes: the console's result with console lanes; on
    /// another wave size, what the HLSL wave operation the console compiler
    /// expanded gives on that wave.
    fn shape(&mut self, shape: super::wave::Shape) -> Result<V, Why> {
        use super::wave::{Op, Shape};
        let (op, src, mask, ident, prefix) = match shape {
            Shape::K(k) => return Ok(self.s.m.i32(k)),
            Shape::Value(v) => return self.i32_of(v),
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Masked(src, mask, ident) => {
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                return Ok(self.s.m.select(c, on, x, id));
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
            Shape::SumD(lo, hi, mask, high) => {
                if self.mask_data == MaskData::None {
                    return Err("a wave reduction in a mesh shader");
                }
                // shortcut: the host has no doubles (Metal): each lane's
                // double is rounded to f32 and the sum taken in f32. The
                // programs found sum 1.0 per lane (a count), which f32
                // holds exactly to 2^24.
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let (lo, hi) = (self.i32_of(lo)?, self.i32_of(hi)?);
                let x = self.f64_to_f32_bits(lo, hi);
                let z = self.s.m.i32(0);
                let x = self.s.m.select(c, on, x, z);
                self.s.flags |= flags::WAVE_OPS;
                let f = self.as_float(x);
                let (sum, signed) = (self.s.m.i8(0), self.s.m.i8(0));
                let r = self.s.op(
                    c,
                    "waveActiveOp.f32",
                    Attr::NoUnwind,
                    self.s.t.f32,
                    119,
                    &[f, sum, signed],
                );
                let r = self.as_int(r);
                let r = self.whole_wave(r, REDUCTION, Some(z), |l, lo, hi| {
                    let (lo, hi) = (l.as_float(lo), l.as_float(hi));
                    let r = l.s.m.bin(l.cur, Bin::Add, lo, hi);
                    l.as_int(r)
                });
                return Ok(self.f32_bits_to_f64_dword(r, high));
            }
            Shape::Half(_, src, mask, ident, hi) => {
                // The high half is a console lane's: compute only.
                if self.mask_data != MaskData::Console {
                    return Err("a 16-bit wave reduction outside compute");
                }
                if self.halves.as_ref().is_some_and(|h| !h.whole) {
                    self.refuse("a 16-bit wave reduction across two host waves");
                }
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                let x = self.s.m.select(c, on, x, id);
                self.s.flags |= flags::WAVE_OPS;
                // shortcut: the f16 sum is taken in f32 and rounded once; the
                // console rounds each step, in its own order.
                let low = self.s.m.i32(0xFFFF);
                let h = self.s.m.bin(c, Bin::And, x, low);
                let f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
                let (sum, signed) = (self.s.m.i8(0), self.s.m.i8(0));
                let r = self.s.op(
                    c,
                    "waveActiveOp.f32",
                    Attr::NoUnwind,
                    self.s.t.f32,
                    119,
                    &[f, sum, signed],
                );
                let r = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[r]);
                let r = self.s.m.bin(c, Bin::And, r, low);
                // Lane `hi` runs the section on the console even where it is
                // elsewhere on the host; there it holds the identity.
                let i32t = self.s.t.i32;
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let b = self
                    .s
                    .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
                let word = self.s.m.extract(c, b, hi / 32);
                let (sh, one, z) = (self.s.m.i32(hi % 32), self.s.m.i32(1), self.s.m.i32(0));
                let bit = self.s.m.bin(c, Bin::LShr, word, sh);
                let bit = self.s.m.bin(c, Bin::And, bit, one);
                let there = self.s.m.cmp(c, pred::NE, bit, z);
                let lane = self.s.m.i32(hi);
                let at = self.read_lane(x, lane);
                let top = self.s.m.select(c, there, at, id);
                let high = self.s.m.i32(0xFFFF_0000);
                let top = self.s.m.bin(c, Bin::And, top, high);
                return Ok(self.s.m.bin(c, Bin::Or, top, r));
            }
        };
        // A mesh program's lanes are numbered as the console's, its host
        // waves are not the console's: only a minimum or maximum (the loop
        // that serves each distinct value once) means the same on any wave.
        if self.mask_data == MaskData::None
            && (prefix.is_some()
                || !matches!(
                    op,
                    Op::MinU | Op::MaxU | Op::MinI | Op::MaxI | Op::MinF | Op::MaxF
                ))
        {
            return Err("a wave reduction in a mesh shader");
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let on = self.boolean(m, c)?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);
        self.s.flags |= flags::WAVE_OPS;
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let arg = if float { self.as_float(x) } else { x };
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        let r = match (op, prefix) {
            (Op::AddU | Op::AddF, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    &format!("wavePrefixOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    121,
                    &[arg, k0, k1],
                );
                if inclusive {
                    self.s.m.bin(c, Bin::Add, r, arg)
                } else {
                    r
                }
            }
            (_, Some(_)) => return Err("a wave prefix of an operation other than a sum"),
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s
                    .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, k, u],
                )
            }
        };
        let r = if float { self.as_int(r) } else { r };
        if self.half_alone() {
            return Ok(r);
        }
        // Two host waves hold the console's wave: the halves' results combine.
        if prefix.is_some() {
            // an upper lane's prefix has the lower half's total before it
            let (k0, ku) = (self.s.m.i8(0), self.s.m.i8(u8::from(!float)));
            let total = self.s.op(
                c,
                &format!("waveActiveOp.{sfx}"),
                Attr::NoUnwind,
                ty,
                119,
                &[arg, k0, ku],
            );
            let total = if float { self.as_int(total) } else { total };
            let z = self.s.m.i32(0);
            return Ok(self.whole_wave(
                total,
                "a wave prefix sum where the halves cannot meet",
                Some(z),
                |l, lo, _| {
                    let c = l.cur;
                    let upper = l.upper_half(c);
                    if float {
                        let (lo, r) = (l.as_float(lo), l.as_float(r));
                        let sum = l.s.m.bin(c, Bin::Add, lo, r);
                        let sum = l.as_int(sum);
                        let r = l.as_int(r);
                        l.s.m.select(c, upper, sum, r)
                    } else {
                        let sum = l.s.m.bin(c, Bin::Add, lo, r);
                        l.s.m.select(c, upper, sum, r)
                    }
                },
            ));
        }
        Ok(self.whole_wave(r, REDUCTION, Some(id), |l, lo, hi| {
            let c = l.cur;
            let pick = |l: &mut Self, p: u8| {
                let lower = l.s.m.cmp(c, p, lo, hi);
                l.s.m.select(c, lower, lo, hi)
            };
            match op {
                Op::AddU => l.s.m.bin(c, Bin::Add, lo, hi),
                Op::Or => l.s.m.bin(c, Bin::Or, lo, hi),
                Op::And => l.s.m.bin(c, Bin::And, lo, hi),
                Op::Xor => l.s.m.bin(c, Bin::Xor, lo, hi),
                Op::MinU => pick(l, pred::IULT),
                Op::MaxU => pick(l, pred::IUGT),
                Op::MinI => pick(l, pred::SLT),
                Op::MaxI => pick(l, pred::SGT),
                Op::AddF | Op::MinF | Op::MaxF | Op::AddH => {
                    let (a, b) = (l.as_float(lo), l.as_float(hi));
                    let r = match op {
                        Op::AddF => l.s.m.bin(c, Bin::Add, a, b),
                        Op::MaxF => l.binary(35, a, b),
                        _ => l.binary(36, a, b),
                    };
                    l.as_int(r)
                }
            }
        }))
    }

    /// Halves: `a` of the first lane in EXEC of the console's wave (lane 0's
    /// with no lane in EXEC), where `first` is this host wave's first
    /// lane's. Where every lane in the block is in EXEC that is the lower
    /// half's `first`; else each half finds its first lane in EXEC, and the
    /// lower half's counts if it has one.
    fn first_lane(&mut self, inst: &Inst, a: V, first: V) -> V {
        const WHAT: Why = "v_readfirstlane_b32 where the halves cannot meet";
        let b = self.cur_ir;
        let st = self.x.st;
        let exec = inst
            .read(Reg::S(EXEC))
            .and_then(|v| st.masks.get(&v))
            .copied();
        let all = exec.is_some_and(|e| !st.bools.meet(&[(st.reach[b], true), (e, false)]));
        // (where threads have ended the lower half may hold none)
        if all && self.halves_full(b) || !self.halves_meet(b) || self.guarded != 0 {
            return self.whole_wave(first, WHAT, None, |_, lo, _| lo);
        }
        let c = self.cur;
        let Some(on) = exec.and_then(|e| self.boolean(e, c).ok()) else {
            self.refuse("v_readfirstlane_b32 under an untracked EXEC");
            return first;
        };
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
        let lanes = self.s.m.extract(c, ballot, 0);
        let (z, k31) = (self.s.m.i32(0), self.s.m.i32(31));
        let lane = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lanes]);
        let lane = self.s.m.bin(c, Bin::And, lane, k31);
        let at = self.read_lane(a, lane);
        let none = self.s.m.cmp(c, pred::EQ, lanes, z);
        let own = self.s.m.select(c, none, first, at);
        let (lo_lanes, hi_lanes) = self.exchange(c, b, lanes, z);
        let (lo, hi) = self.exchange(c, b, own, z);
        let in_lo = self.s.m.cmp(c, pred::NE, lo_lanes, z);
        let in_hi = self.s.m.cmp(c, pred::NE, hi_lanes, z);
        let upper = self.s.m.select(c, in_hi, hi, lo);
        self.s.m.select(c, in_lo, lo, upper)
    }

    /// Halves in a group of several console waves: whether `v_readfirstlane_b32`
    /// `inst`, where the halves cannot meet for it ([`Self::first_lane`]),
    /// reads a value every lane of the console wave holds alike
    /// ([`wave_alike`]), so that each host wave's own first lane has the
    /// console's. Where they can meet the exchange stays as it was.
    fn first_alike(&self, inst: &Inst) -> bool {
        let Some(h) = self.halves.as_ref().filter(|h| h.count > 2) else {
            return false;
        };
        let (p, st, b) = (self.x.p, self.x.st, self.cur_ir);
        let all = inst
            .read(Reg::S(EXEC))
            .and_then(|v| st.masks.get(&v))
            .is_some_and(|&e| !st.bools.meet(&[(st.reach[b], true), (e, false)]));
        let whole = all && h.full[b] || !h.meet[b] || self.guarded != 0;
        if !(whole && (!h.full[b] || self.guarded != 0)) {
            return false;
        }
        let Fields::Valu(v) = &inst.fields else {
            return false;
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return false;
        };
        let alike = self.alike.get_or_init(|| {
            let StageIo::Cs(cs) = &self.x.native.io else {
                return Set::default();
            };
            let [tx, ty, tz] = cs.threads;
            // (console waves are runs of 64 ids along x)
            let tid = (tx % 64 == 0 || (ty, tz) == (1, 1))
                .then(|| p.inputs.iter().find(|i| i.0 == Reg::V(0)).map(|i| i.1))
                .flatten();
            wave_alike(p, st, tid, &self.waterfall_reads)
        });
        inst.read(Reg::V(reg)).is_some_and(|x| alike.contains(&x))
    }

    /// The lanes of the mask whose first lane a `v_readlane_b32` reads
    /// (`s_ff1_i32_b64` of the mask as its lane, the console compiler's
    /// read of the first lane of a set), where the host has every lane of
    /// the mask in the block and each lane there is one: a mask written in
    /// the block of lanes in EXEC ([`Structured::active`]) that every lane
    /// entering the block is in, so that it is never empty where the read
    /// runs (the console reads lane 63 by an empty one).
    fn mask_led(&self, inst: &Inst) -> Option<Bool> {
        let (p, st, b) = (self.x.p, self.x.st, self.cur_ir);
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let &Operand::Scalar { code, .. } = v.src.get(1)? else {
            return None;
        };
        let lane = inst.read(Reg::S(code))?;
        let here = |v: Value| {
            p.blocks[b]
                .insts
                .iter()
                .find(|i| i.writes.iter().any(|w| w.1 == v))
        };
        let first = here(lane)?;
        let Fields::Sop { src, .. } = &first.fields else {
            return None;
        };
        let &Operand::Scalar { code, .. } = src.first()? else {
            return None;
        };
        let mask = first.read(Reg::S(code))?;
        let &lanes = st.masks.get(&mask)?;
        (first.name() == "s_ff1_i32_b64"
            && here(mask).is_some()
            && st.active.contains(&mask)
            && !st.bools.meet(&[(st.reach[b], true), (lanes, false)]))
        .then_some(lanes)
    }

    /// Halves: `a` of the first lane of `lanes` in the console's wave
    /// ([`Self::mask_led`]): each half finds its first lane of them, and
    /// the lower half's counts if it has one.
    fn first_of(&mut self, lanes: Bool, a: V) -> V {
        let b = self.cur_ir;
        let c = self.cur;
        if self.half_alone() {
            return a;
        }
        let on = match self.boolean(lanes, c) {
            Ok(on) if self.halves_meet(b) && self.guarded == 0 => on,
            _ => {
                self.refuse("v_readlane_b32 of a mask's first lane where the halves cannot meet");
                return a;
            }
        };
        self.s.flags |= flags::WAVE_OPS;
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
        let lanes = self.s.m.extract(c, ballot, 0);
        let (z, k31) = (self.s.m.i32(0), self.s.m.i32(31));
        let lane = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lanes]);
        let lane = self.s.m.bin(c, Bin::And, lane, k31);
        let at = self.read_lane(a, lane);
        let none = self.s.m.cmp(c, pred::EQ, lanes, z);
        let own = self.s.m.select(c, none, a, at);
        let (lo_lanes, _) = self.exchange(c, b, lanes, z);
        let (lo, hi) = self.exchange(c, b, own, z);
        let in_lo = self.s.m.cmp(c, pred::NE, lo_lanes, z);
        self.s.m.select(c, in_lo, lo, hi)
    }

    /// Halves: a host wave's result `r` of an operation over the wave (the
    /// same in all its lanes) made the console wave's: `both` of the lower
    /// and the upper half's. `absent`: what a half with no lane gives (a
    /// reduction's identity), if the operation has such a value; another
    /// needs every lane of both halves there.
    fn whole_wave(
        &mut self,
        r: V,
        what: Why,
        absent: Option<V>,
        both: impl FnOnce(&mut Self, V, V) -> V,
    ) -> V {
        if self.half_alone() {
            return r;
        }
        let b = self.cur_ir;
        let meet = match absent {
            Some(_) => self.halves_meet(b),
            None => self.halves_full(b),
        };
        if !meet || self.guarded != 0 {
            self.refuse(what);
            return r;
        }
        let absent = absent.unwrap_or(r);
        let (lo, hi) = self.exchange(self.cur, b, r, absent);
        both(self, lo, hi)
    }

    fn run(mut self, compact: bool) -> Result<Lowered, Stop> {
        let p = self.x.p;
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        for (n, &b) in order.iter().enumerate() {
            self.block[b] = Some(if n == 0 {
                self.s.entry
            } else {
                self.s.m.new_block()
            });
        }
        // Loop headers count iterations (`LOOP_LIMIT`).
        for &b in &order {
            let header = p.blocks[b]
                .preds
                .iter()
                .any(|&q| self.block[q].is_some() && self.rpo[b] <= self.rpo[q]);
            if header {
                let blk = self.block[b].expect("reachable");
                let phi = self.s.m.phi(blk, self.s.t.i32);
                self.counters.insert(b, (phi, Vec::new()));
            }
        }
        self.cur = self.s.entry;
        self.find_sections();
        self.inputs()?;
        if let Some(mesh) = self.s.mesh {
            self.mesh_prologue(mesh).map_err(|w| unsupported(0, w))?;
            self.block[order[0]] = Some(self.cur);
        }
        self.leave_past_counts(order[0]);
        for &b in &order {
            self.cur = self.block[b].expect("reachable");
            self.cur_ir = b;
            let preds: Vec<usize> = p.blocks[b]
                .preds
                .iter()
                .copied()
                .filter(|&q| self.block[q].is_some())
                .collect();
            self.exported[b] = !preds.is_empty() && preds.iter().all(|&q| self.exported_out(q));
            for (i, phi) in p.blocks[b].phis.iter().enumerate() {
                let val = match phi.reg {
                    Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                    _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                    Reg::Scc => {
                        let v = self.s.m.phi(self.cur, self.s.t.i1);
                        self.data_phis.push((v, b, i));
                        Val::Bit(v)
                    }
                    _ => {
                        let v = self.s.m.phi(self.cur, self.s.t.i32);
                        self.data_phis.push((v, b, i));
                        Val::I32(v)
                    }
                };
                self.vals.insert(phi.value, val);
            }
            let mut i = 0;
            self.mode16 = self.round16[b];
            while i < p.blocks[b].insts.len() {
                if let Some(&lanes) = self.x.st.discards.get(&(b, i)) {
                    // A pixel's discard: the lanes stay on as helpers.
                    let at = self.cur;
                    let offset = p.blocks[b].insts[i].machine.offset;
                    let kill = self
                        .boolean(lanes, at)
                        .map_err(|w| unsupported(offset, w))?;
                    self.s
                        .op(at, "discard", Attr::NoUnwind, self.s.t.void, 82, &[kill]);
                }
                if self.split_barriers.contains(&(b, i)) {
                    self.group_barrier();
                }
                self.inst(b, i)?;
                self.mode16 = round16_after(&p.blocks[b].insts[i], self.mode16);
                if let Some(end) = self.sections.get(&b).and_then(|s| s.get(&i)).map(|s| s.end) {
                    for j in i + 1..end {
                        self.section_inst(b, j, i);
                    }
                    i = end;
                    continue;
                }
                i += 1;
            }
            if self.split_barriers.contains(&(b, p.blocks[b].insts.len())) {
                self.group_barrier();
            }
            self.mask_ok = self.mask_data == MaskData::Console;
            self.tail[b] = Some(self.cur);
            self.exit(b)?;
        }
        self.fill_phis()?;
        debug_assert!(self.counts_kept());
        let user_data = self.user_reads.values().copied().collect();
        self.s.samplers = self.samplers.len() as u32;
        let root_cbvs = self.s.root_cbvs.iter().map(|r| r.0).collect();
        let root_srvs = self.s.root_srvs.iter().map(|r| r.0).collect();
        let root_uavs = self.s.root_uavs.iter().map(|r| r.0).collect();
        let memory = self.s.memory.is_some();
        let memory_writes = self.s.memory_dirty.is_some();
        let view_addresses = self.s.view_addresses.is_some();
        let cmp_samplers = self.s.cmp_samplers;
        let idle_distances = match self.s.mesh {
            Some(_) => (0..self.s.outputs.len())
                .filter(|&id| matches!(self.s.outputs[id].kind, 6 | 7))
                .map(|id| {
                    let (stored, live) = self
                        .s
                        .m
                        .stored_channels("dx.op.storeVertexOutput", id as u32);
                    (id, stored & !live)
                })
                .collect(),
            None => Vec::new(),
        };
        if compact {
            let interface = if self.s.pixel {
                &mut self.s.inputs
            } else {
                &mut self.s.outputs
            };
            if !super::container::compact_mesh_interface(interface) {
                return Err(unsupported(
                    0,
                    "unexpected layout for compact mesh interface",
                ));
            }
        }
        Ok(Lowered {
            cmp_samplers,
            root_cbvs,
            root_srvs,
            root_uavs,
            memory,
            memory_writes,
            view_addresses,
            heap: self.heap[0],
            heap_writes: self.heap[1],
            view_marks: self.s.view_writes.is_some(),
            #[cfg(test)]
            nonuniform_handles: self.s.nonuniform_handles,
            #[cfg(test)]
            heap_handles: self.s.m.written_calls("dx.op.createHandleFromHeap") as u32,
            #[cfg(test)]
            wave_ops: self.s.flags & flags::WAVE_OPS != 0,
            #[cfg(test)]
            rovs: self.s.flags & flags::ROVS != 0,
            #[cfg(test)]
            split_barriers: self.split_barriers.len(),
            #[cfg(test)]
            wave_size: self.s.wave_size,
            #[cfg(test)]
            exchanges: self.halves.as_ref().map_or(0, |h| h.sites),
            #[cfg(test)]
            barriers: self.s.m.written_calls("dx.op.barrier"),
            dxil: self.s.finish(),
            user_data,
            samplers: self.samplers,
            idle_distances,
        })
    }

    fn exported_out(&self, b: usize) -> bool {
        self.exported[b] || self.has_done_export(b)
    }

    fn has_done_export(&self, b: usize) -> bool {
        (0..self.x.p.blocks[b].insts.len())
            .any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.done))
    }

    /// Values of the program's input registers.
    fn inputs(&mut self) -> Result<(), Stop> {
        let io_inputs: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        for &(reg, v) in &self.x.p.inputs {
            let val = match reg {
                Reg::S(_) if self.params.contains_key(&v) => continue, // loaded on first use
                // Most programs only interpolate with them: loaded on first
                // use as a value (`get`).
                Reg::V(_)
                    if self.barycentrics
                        && matches!(io_inputs.get(&v), Some(Input::Barycentric { .. })) =>
                {
                    continue;
                }
                Reg::S(n) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Above the SGPRs the hardware initializes, a register
                    // holds what the last wave left, so no program depends
                    // on it: a read the paths only seem to allow (a branch
                    // past the write, taken only with one past the read)
                    // reads zero, as VGPRs do.
                    None if n <= 105 && self.x.io.initialized_sgprs.is_some_and(|k| n >= k) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    // The primitive mask (pixel), which console code only
                    // moves to M0 for its interpolation (spilled and reloaded
                    // on the way); the host interpolates by itself, so the
                    // value goes unread.
                    None if self.s.pixel && self.x.io.initialized_sgprs == Some(n + 2) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    // The ordered section's collision id, and (`operand`) the
                    // id of the last wave that left it: 0, so no wave waits;
                    // the host's rasterizer-ordered views order the accesses.
                    None if self.pops && self.x.io.initialized_sgprs == Some(n + 1) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    None => Val::Missing("a hardware SGPR input"),
                },
                Reg::Scc => Val::Missing("SCC on entry"),
                Reg::V(_) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Registers no one set read as zero (the validator
                    // refuses reads of undef).
                    None => Val::I32(self.s.m.i32(0)),
                },
            };
            self.vals.insert(v, val);
        }
        Ok(())
    }

    fn input(&mut self, input: Input) -> Val {
        let (kind, col) = match input {
            Input::VertexId => (1, 0),
            Input::InstanceId => (2, 0),
            Input::Position(c) => (3, c),
            Input::Attribute { row, channel } => {
                return match self.find_input(row, channel) {
                    Some((id, col)) => self.load_input(id, col),
                    None => Val::Missing("a vertex input the signature lacks"),
                };
            }
            Input::Barycentric { .. } if !self.barycentrics => {
                return Val::Missing(BARYCENTRICS_READ);
            }
            Input::Barycentric { kind, index } => return self.barycentric(kind, index),
            Input::PrimitiveId => return Val::Missing("the primitive id"),
            // The console code tests it against 0 (`v_cmp_ne_u32`), as the
            // host's value can be.
            Input::FrontFace => (13, 0),
            Input::ThreadIdInGroup(c) => return self.thread_id(c),
            Input::GroupId(c) => return self.compute_id("groupId.i32", 94, c),
            Input::GroupInfo => return Val::Missing("the thread-group info SGPR"),
            Input::MeshGroup => {
                let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
                    unreachable!()
                };
                let n = self.s.m.i32(self.x.mesh_stride);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Mul, g, n));
            }
            Input::MergedWaveInfo => {
                // shortcut: only the wave index (bits 24-27) is set; the
                // console's other bits (lane counts) are read by no program.
                let f = self.mesh_thread();
                let shift = self.s.m.i32(self.wave().trailing_zeros());
                let w = self.s.m.bin(self.s.entry, Bin::LShr, f, shift);
                let k24 = self.s.m.i32(24);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Shl, w, k24));
            }
            _ => return Val::Missing("a pixel system input"),
        };
        let free = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let id = match self.s.inputs.iter().position(|e| e.kind == kind) {
            Some(id) => {
                if kind != 3 {
                    self.s.inputs[id].row.get_or_insert(free);
                }
                id
            }
            None if kind >= 3 => return Val::Missing("a pixel system input the signature lacks"),
            None => {
                // The console signature leaves out vertex and instance ids; PC
                // gives them a row after the others.
                let row = self
                    .s
                    .inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .map(|r| r + 1)
                    .max()
                    .unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: ["", "SV_VertexID", "SV_InstanceID"][kind as usize].into(),
                    index: 0,
                    kind,
                    comp: 5,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: 1,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.load_input(id, col)
    }

    /// A component of a compute thread or group id, read in the entry block.
    fn compute_id(&mut self, name: &str, opcode: u32, c: u8) -> Val {
        let (entry, c) = (self.s.entry, self.s.m.i32(u32::from(c)));
        Val::I32(
            self.s
                .op(entry, name, Attr::ReadNone, self.s.t.i32, opcode, &[c]),
        )
    }

    fn find_input(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.inputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            // `then`, not `then_some`: the channel is only past `col` when
            // the condition holds.
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    fn load_input(&mut self, id: usize, col: u8) -> Val {
        if self.s.inputs[id].half {
            // shortcut: 16-bit vertex inputs (the fetch routine's) are not
            // unpacked; pixel inputs are (`interp`).
            return Val::Missing("a 16-bit input");
        }
        self.load_any_input(id, col)
    }

    /// An input element's channel as declared (16-bit ones as 32-bit floats).
    fn load_any_input(&mut self, id: usize, col: u8) -> Val {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.undef(self.s.t.i32),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            let z = self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32);
            Val::I32(z)
        } else if int {
            Val::I32(v)
        } else {
            Val::I32(self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32))
        }
    }

    /// The DXIL form of an IR value.
    fn get(&mut self, v: Value) -> Result<Val, Why> {
        if self.phi_at.contains_key(&v) {
            self.read_phis.insert(v);
        }
        let found = self.vals.get(&v).copied();
        // Most values have a DXIL form, which is the answer whatever they
        // are; only one without needs to know whether it is a lane mask.
        if let Some(val @ (Val::I32(_) | Val::Bit(_))) = found {
            return Ok(val);
        }
        if self.mask_ok && (self.x.st.masks.contains_key(&v) || self.x.st.highs.contains_key(&v)) {
            return self.mask_dword(v).map(Val::I32);
        }
        if self.mask_ok
            && let Some(&(lo, hi)) = self.mask_sccs.get(&v)
        {
            return self.mask_scc(lo, hi).map(Val::Bit);
        }
        if let Some(Val::Missing(w)) = found {
            return Err(w);
        }
        if let Some(&(param, dword)) = self.params.get(&v) {
            let val = self.user(param, dword);
            self.vals.insert(v, val);
            return Ok(val);
        }
        if let Some(loaded) = self.lazy_barycentric(v) {
            loaded?;
            return Ok(self.vals[&v]);
        }
        Err("a value defined after its use")
    }

    /// Loads `v` if it is a barycentric input VGPR not loaded yet (see
    /// `inputs`): `None` if it is not one.
    fn lazy_barycentric(&mut self, v: Value) -> Option<Result<(), Why>> {
        if !self.barycentrics {
            return None;
        }
        let (kind, index) = self.x.io.inputs.iter().find_map(|&(w, i)| match i {
            Input::Barycentric { kind, index } if w == v => Some((kind, index)),
            _ => None,
        })?;
        let val = self.barycentric(kind, index);
        self.vals.insert(v, val);
        Some(match val {
            Val::Missing(w) => Err(w),
            _ => Ok(()),
        })
    }

    /// Barycentric `index` (0 i, 1 j) of `kind` as a value: a channel of
    /// `SV_Barycentrics` declared with the same interpolation mode.
    ///
    /// GFX10 interpolates P0 + i (P1 - P0) + j (P2 - P0), so i and j are
    /// the weights of vertices 1 and 2. The host's weights (x, y, z) are
    /// those of vertices 0, 1 and 2, vertex 0 being the provoking vertex,
    /// which is the console's P0 (`v_interp_mov` P0 is lowered to a flat
    /// input): i is y and j is z.
    /// shortcut: triangles only; on lines and points the console's i and j
    /// are not checked against the host's weights.
    fn barycentric(&mut self, kind: Barycentric, index: u8) -> Val {
        let Some(mode) = interp_mode(kind) else {
            return Val::Missing("pull-model barycentrics");
        };
        if index > 1 {
            return Val::Missing("pull-model barycentrics");
        }
        // One element per interpolation mode (the console's own, if its
        // signature declares one): DXIL allows two.
        let bary = |e: &SigElement| e.kind == 28;
        let id = match self
            .s
            .inputs
            .iter()
            .position(|e| bary(e) && (e.interp == mode || e.usage == 0))
        {
            Some(id) => id,
            None => {
                let index = self.s.inputs.iter().filter(|e| bary(e)).count() as u32;
                if index >= 2 {
                    return Val::Missing("barycentrics of a third interpolation mode");
                }
                // A system value the validator wants unpacked (packing
                // location -1), like other non-packed system values.
                self.s.inputs.push(SigElement {
                    name: "SV_Barycentrics".into(),
                    index,
                    kind: 28,
                    comp: 9,
                    interp: mode,
                    row: None,
                    col: 0,
                    cols: 3,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        let e = &mut self.s.inputs[id];
        if e.col != 0 || e.cols < 3 {
            return Val::Missing("a console barycentrics element without i and j");
        }
        e.row = None;
        e.interp = mode;
        self.s.flags |= flags::BARYCENTRICS;
        self.load_any_input(id, index + 1)
    }

    fn i32_of(&mut self, v: Value) -> Result<V, Why> {
        match self.get(v)? {
            Val::I32(x) => Ok(x),
            Val::Bit(x) => Ok(self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32)),
            Val::Missing(w) => Err(w),
        }
    }

    /// A user-data dword: a root parameter's value.
    fn user(&mut self, param: usize, dword: u32) -> Val {
        let rp = self.x.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => Val::I32(self.s.m.i32(0)),
            RootType::Constants | RootType::Table => Val::I32(self.user_dword(param, dword)),
            // Its address, where root views are title memory.
            RootType::Descriptor if self.root_memory => Val::I32(self.user_dword(param, dword)),
            _ => Val::Missing("a root descriptor"),
        }
    }

    /// User-data dword `dword` of root parameter `param` as the host passes
    /// it (for a root descriptor: its GPU address, low dword first).
    fn user_dword(&mut self, param: usize, dword: u32) -> V {
        let rp = self.x.native.layout[param];
        let index = u32::from(rp.user_data) + dword;
        self.user_at(UserData {
            index,
            slot: rp.slot,
            root: rp.root,
            dword,
        })
    }

    /// A root SRV or UAV (`uav`) of root parameter `param`, in a program
    /// that does not read title memory: the resource-heap view the host
    /// makes of the host copy holding the buffer, whose index comes in the
    /// root descriptor's own first user-data dword (module notes, "Root SRVs
    /// and root UAVs"); offsets from the root address move by
    /// [`Lower::root_view_offset`]. Host root views have no bounds; a view's
    /// reads past the copy give 0 and its writes there are dropped.
    fn root_view(&mut self, param: usize, uav: bool) -> V {
        let rp = self.x.native.layout[param];
        // The descriptor's two user-data dwords hold its address on the
        // console; such a program never reads them as data (`user`).
        let index = u32::from(rp.user_data);
        if let Some(&h) = self.root_views.get(&index) {
            return h;
        }
        let i = self.user_at(UserData {
            index,
            slot: rp.slot,
            root: RootType::Descriptor,
            dword: if uav {
                super::ROOT_UAV_VIEW
            } else {
                super::ROOT_SRV_VIEW
            },
        });
        let props = if uav { (11 | 0x1000, 0) } else { (11, 0) };
        let h = self.s.heap_handle(self.s.entry, i, false, false, props);
        self.root_views.insert(index, h);
        h
    }

    /// For a root SRV or UAV read through the view of its host copy
    /// ([`Lower::root_view`]): the root address's byte offset in that view,
    /// from the descriptor's second user-data dword. None for other sources.
    fn root_view_offset(&mut self, src: &Source) -> Option<V> {
        let Source::Built { address, .. } = src else {
            return None;
        };
        let rp = *self.x.native.layout.get(address.param)?;
        let cbv = self
            .used_kind(src)
            .is_some_and(|u| u.kind == ResourceType::Cbv);
        if self.root_memory || rp.root != RootType::Descriptor || cbv {
            return None;
        }
        Some(self.user_at(UserData {
            index: u32::from(rp.user_data) + 1,
            slot: rp.slot,
            root: RootType::Descriptor,
            dword: super::ROOT_VIEW_OFFSET,
        }))
    }

    /// User-data dword `u.index`, which the host fills as `u` says, loaded
    /// in the entry block.
    fn user_at(&mut self, u: UserData) -> V {
        let index = u.index;
        self.user_reads.insert(index, u);
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        self.s.m.extract(entry, r, index % 4)
    }

    /// In a program that uses title memory, the address of the root SRV or
    /// UAV a descriptor `src` is built around: `(lo, hi)`, `hi` cut to bits
    /// 32-47. None for other descriptors and programs.
    fn root_address(&mut self, src: &Source) -> Option<(V, V)> {
        let Source::Built { address, .. } = src else {
            return None;
        };
        let rp = self.x.native.layout.get(address.param)?;
        let cbv = self
            .used_kind(src)
            .is_some_and(|u| u.kind == ResourceType::Cbv);
        if !self.root_memory || rp.root != RootType::Descriptor || cbv {
            return None;
        }
        let lo = self.user_dword(address.param, 0);
        let hi = self.user_dword(address.param, 1);
        let mask = self.s.m.i32(0xFFFF);
        Some((lo, self.s.m.bin(self.s.entry, Bin::And, hi, mask)))
    }

    /// A raw view `src` at byte offset `off` (`props` as [`Self::handle`]
    /// takes them): its handle and the offset in it. A root view of a
    /// program that uses title memory is title memory at the root address
    /// plus the offset ([`Self::memory`]: `bytes` accessed, `write`, the
    /// `active` lanes), so the access is bounds-checked (host root views
    /// have no bounds) and sees the copies the program's other accesses see.
    #[allow(clippy::too_many_arguments)]
    fn raw_view(
        &mut self,
        site: Site,
        src: &Source,
        off: V,
        bytes: u32,
        props: (u32, u32),
        write: bool,
        active: Option<V>,
    ) -> Result<(V, V), Why> {
        let Some((lo, hi)) = self.root_address(src) else {
            let h = self.handle(site, src, false, props)?;
            let off = match self.root_view_offset(src) {
                Some(at) => self.s.m.bin(self.cur, Bin::Add, off, at),
                None => off,
            };
            return Ok((h, off));
        };
        // The hardware adds the 32-bit offset to the 48-bit base.
        let (lo, hi) = self.add64(lo, hi, off, false);
        let mask = self.s.m.i32(0xFFFF);
        let hi = self.s.m.bin(self.cur, Bin::And, hi, mask);
        Ok(self.memory(lo, hi, bytes, write, active))
    }

    /// A dword of a lane mask as a number: the ballot of the lanes' bits,
    /// taken in the block that writes the mask (see [`MaskData`]).
    fn mask_dword(&mut self, v: Value) -> Result<V, Why> {
        let (lo, high) = match self.x.st.highs.get(&v) {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        if !self.x.st.active.contains(&lo) && !self.ballots.contains_key(&lo) {
            let xy = self.mask_number(lo)?;
            self.ballots.insert(lo, xy);
        }
        let (x, y) = match self.ballots.get(&lo) {
            Some(&xy) => xy,
            None => {
                let at = match self.def_block.get(&lo) {
                    // Program inputs (EXEC on entry): the lanes that
                    // start, not those a count of threads leaves out.
                    None => self.counted.as_ref().map_or(self.s.entry, |c| c.start),
                    Some(&b) if b == self.cur_ir => self.cur,
                    Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
                };
                if !self.balloting.insert(lo) {
                    return Err("a lane mask whose lanes depend on its own value");
                }
                let bit = self.boolean(m, at);
                self.balloting.remove(&lo);
                let bit = bit?;
                let i32t = self.s.t.i32;
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let r = self
                    .s
                    .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
                self.s.flags |= flags::WAVE_OPS;
                let mut xy = (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1));
                if self.halves.as_ref().is_some_and(|h| !h.whole) {
                    let b = self.def_block.get(&lo).copied().unwrap_or(0);
                    if self.halves_meet(b) && !(at == self.cur && self.guarded != 0) {
                        // each half's ballot is a dword of the console's mask
                        let z = self.s.m.i32(0);
                        xy = self.exchange(at, b, xy.0, z);
                    } else {
                        // One thread alone there: its bit, in its half's
                        // dword. Other lanes there too: the other half's
                        // are missing.
                        // (what reads it stops the form, if anything does)
                        if !self.halves.as_ref().is_some_and(|h| h.solo[b]) {
                            NEEDS.with_borrow_mut(|n| n.push(b));
                            return Err("a lane mask read as data where the halves cannot meet");
                        }
                        let (upper, z) = (self.upper_half(at), self.s.m.i32(0));
                        xy = (
                            self.s.m.select(at, upper, z, xy.0),
                            self.s.m.select(at, upper, xy.0, z),
                        );
                    }
                }
                self.ballots.insert(lo, xy);
                xy
            }
        };
        Ok(if high { y } else { x })
    }

    /// SCC of a lane-width mask operation read as a scalar ([`Self::mask_sccs`]):
    /// whether mask `lo`, `hi` has some bit set, any lane's (inside EXEC or
    /// not, as the console tests the whole register: `s_orn2` of a mask
    /// with none set has every bit set). Made where the mask's ballot is.
    fn mask_scc(&mut self, lo: Value, hi: Option<Value>) -> Result<V, Why> {
        let x = self.mask_dword(lo)?;
        let x = match hi {
            Some(hi) => {
                let y = self.mask_dword(hi)?;
                let at = self.ballot_block(lo)?;
                self.s.m.bin(at, Bin::Or, x, y)
            }
            None => x,
        };
        let at = self.ballot_block(lo)?;
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(at, pred::NE, x, z))
    }

    /// The DXIL block where [`Self::mask_dword`] takes the ballot of `lo`.
    fn ballot_block(&self, lo: Value) -> Result<usize, Why> {
        Ok(match self.def_block.get(&lo) {
            None => self.counted.as_ref().map_or(self.s.entry, |c| c.start),
            Some(&b) if b == self.cur_ir => self.cur,
            Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
        })
    }

    /// Both dwords of a mask holding lanes that do not reach the block
    /// writing it (a complement, a copy made where EXEC is not known), so
    /// no ballot there gives it: with console lanes, the scalar operation
    /// that writes it applied to its operands' numbers, as the console does.
    fn mask_number(&mut self, lo: Value) -> Result<(V, V), Why> {
        const ELSEWHERE: Why = "a lane mask holding lanes elsewhere read as data";
        if self.mask_data != MaskData::Console {
            return Err(ELSEWHERE);
        }
        let p = self.x.p;
        // A register the compiler reuses: a mask on one path, data on another.
        if let Some(&(b, i)) = self.phi_at.get(&lo)
            && p.blocks[b].phis[i]
                .args
                .iter()
                .any(|a| !self.x.st.masks.contains_key(a))
        {
            return Err("a register joining a lane mask and data, read as data");
        }
        let &b = self.def_block.get(&lo).ok_or(ELSEWHERE)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))
            .ok_or(ELSEWHERE)?;
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            src,
            ..
        } = &inst.fields
        else {
            return Err(ELSEWHERE);
        };
        let wave64 = self.x.native.wave64;
        let width = if wave64 { "_b64" } else { "_b32" };
        let op = inst
            .name()
            .strip_prefix("s_")
            .and_then(|n| n.strip_suffix(width))
            .ok_or(ELSEWHERE)?;
        if inst.write(Reg::S(*code)) != Some(lo) || inst.write(Reg::S(EXEC)).is_some() {
            return Err(ELSEWHERE);
        }
        let at = if b == self.cur_ir {
            self.cur
        } else {
            self.tail[b].ok_or("a lane mask written after its use")?
        };
        // Dword `h` of source `k` (inline constants sign-extend).
        let word = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            let x = match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or(ELSEWHERE)?;
                    return l.i32_of(v);
                }
                Some(&Operand::Const(c)) => {
                    let x = crate::resource::inline(c).ok_or(ELSEWHERE)?;
                    match h {
                        0 => x,
                        _ if (x as i32) < 0 => u32::MAX,
                        _ => 0,
                    }
                }
                Some(&Operand::Literal(x)) if h == 0 => x,
                Some(Operand::Literal(_)) => 0,
                _ => return Err(ELSEWHERE),
            };
            Ok(l.s.m.i32(x))
        };
        let mut out = [self.s.m.i32(0); 2];
        for h in 0..1 + u8::from(wave64) {
            let a = word(self, 0, h)?;
            let ones = self.s.m.i32(u32::MAX);
            out[usize::from(h)] = match op {
                "mov" => a,
                "not" => self.s.m.bin(at, Bin::Xor, a, ones),
                "cselect" => {
                    let c = word(self, 1, h)?;
                    let scc = inst.read(Reg::Scc).ok_or(ELSEWHERE)?;
                    let cond = match self.get(scc)? {
                        Val::Bit(x) => x,
                        Val::I32(x) => {
                            let z = self.s.m.i32(0);
                            self.s.m.cmp(at, pred::NE, x, z)
                        }
                        Val::Missing(w) => return Err(w),
                    };
                    self.s.m.select(at, cond, a, c)
                }
                _ => {
                    let c = word(self, 1, h)?;
                    // (operation, complement of src1, complement of the result)
                    let (f, not_c, not_r) = match op {
                        "and" => (Bin::And, false, false),
                        "or" => (Bin::Or, false, false),
                        "xor" => (Bin::Xor, false, false),
                        "andn2" => (Bin::And, true, false),
                        "orn2" => (Bin::Or, true, false),
                        "nand" => (Bin::And, false, true),
                        "nor" => (Bin::Or, false, true),
                        "xnor" => (Bin::Xor, false, true),
                        _ => return Err(ELSEWHERE),
                    };
                    let c = if not_c {
                        self.s.m.bin(at, Bin::Xor, c, ones)
                    } else {
                        c
                    };
                    let r = self.s.m.bin(at, f, a, c);
                    if not_r {
                        self.s.m.bin(at, Bin::Xor, r, ones)
                    } else {
                        r
                    }
                }
            };
        }
        Ok((out[0], out[1]))
    }
}

impl<'a> Lower<'a> {
    fn guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        match self.x.st.guards[b][i] {
            None => Ok(None),
            Some(g) => Ok(Some(self.boolean(g, self.cur)?)),
        }
    }

    /// Whether every lane outside the guard of the export at (b, i) runs
    /// the `done` export later in block b without its own guard holding,
    /// i.e. is discarded there.
    fn discarded_at_done(&self, b: usize, i: usize) -> bool {
        let guards = &self.x.st.guards[b];
        let g = guards[i].unwrap_or(Bool::TRUE);
        self.exports.iter().any(|(&(eb, ei), e)| {
            e.done && eb == b && ei > i && {
                let done = guards[ei].unwrap_or(Bool::TRUE);
                !self.x.st.bools.meet(&[(done, true), (g, false)])
            }
        })
    }

    // ---- results ----

    fn set(&mut self, v: Option<Value>, val: Val) {
        if let Some(v) = v {
            self.vals.insert(v, val);
        }
    }

    /// Writes VGPR `reg` (or the SGPR of a lane read), under `guard`.
    fn write(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let x = match (Self::sdwa_dst(inst), reg) {
            (Some((sel, unused)), Reg::V(_)) => self.place_bits(inst, reg, x, sel, unused),
            _ => x,
        };
        self.write_placed(inst, reg, x, guard);
    }

    /// [`Self::write`] of a dword already placed (SDWA applied).
    fn write_placed(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let Some(new) = inst.write(reg) else { return };
        let x = match (guard, reg) {
            (Some(g), Reg::V(_)) => {
                let o = inst.read(reg);
                let read = o.is_some_and(|o| self.read_phis.contains(&o));
                let old = match o.map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    // The validator refuses reads of undef.
                    _ => self.s.m.i32(0),
                };
                // What lanes keep is zero where it has no DXIL form (as
                // above), so keeping a phi does not make it live.
                if let Some(o) = o.filter(|_| !read) {
                    self.read_phis.remove(&o);
                }
                self.s.m.select(self.cur, g, x, old)
            }
            _ => x,
        };
        self.vals.insert(new, Val::I32(x));
    }

    /// The SDWA destination selection: a byte or word, and what the other
    /// bits get (0 zeros, 1 the sign above the field, 2 the old bits).
    fn sdwa_dst(inst: &Inst) -> Option<(u8, u8)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        (s.dst_sel < 6).then_some((s.dst_sel, s.dst_unused))
    }

    /// The low byte or word of `x` placed at `sel` in VGPR `reg`.
    fn place_bits(&mut self, inst: &Inst, reg: Reg, x: V, sel: u8, unused: u8) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8u32, 8 * u32::from(sel))
        } else {
            (16, 16 * u32::from(sel - 4))
        };
        let left = self.s.m.i32(32 - width);
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - width - at);
        if unused == 1 {
            // Bits below the field shift in as zeros.
            return self.s.m.bin(c, Bin::AShr, t, right);
        }
        let field = self.s.m.bin(c, Bin::LShr, t, right);
        if unused != 2 {
            return field;
        }
        let old = match inst.read(reg).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let keep = self.s.m.i32(!((((1u64 << width) - 1) as u32) << at));
        let old = self.s.m.bin(c, Bin::And, old, keep);
        self.s.m.bin(c, Bin::Or, old, field)
    }

    fn write_f(&mut self, inst: &Inst, f: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let mut f = f;
        match v.omod {
            0 => {}
            k => {
                // On the flushed result; a zero or denormal product comes out
                // +0 (measured on RDNA2 with IEEE mode off).
                let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                let c = self.cur;
                let k = self.s.m.f32(factor.to_bits());
                let r = self.ftz(f);
                let r = self.s.m.bin(c, Bin::Mul, r, k);
                f = if self.x.relax.denormals {
                    r
                } else {
                    let x = self.as_int(r);
                    let (exp, zero) = (self.s.m.i32(0x7F80_0000), self.s.m.i32(0));
                    let e = self.s.m.bin(c, Bin::And, x, exp);
                    let tiny = self.s.m.cmp(c, pred::EQ, e, zero);
                    let x = self.s.m.select(c, tiny, zero, x);
                    self.as_float(x)
                };
            }
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        let x = self.as_int(f);
        self.write(inst, Reg::V(reg), x, guard);
        if guard.is_none()
            && Self::sdwa_dst(inst).is_none()
            && let Some(new) = inst.write(Reg::V(reg))
        {
            self.floats.insert(new, f);
        }
        Ok(())
    }

    fn write_i(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            return Err("integer output modifiers");
        }
        match v.dst {
            Some(Operand::Vector { reg, .. }) => self.write(inst, Reg::V(reg), x, guard),
            Some(Operand::Scalar { code, .. }) => self.write(inst, Reg::S(code), x, None),
            _ => return Err("a result without a register"),
        }
        Ok(())
    }

    /// Starts code only the lanes where `guard` holds run (a store under a
    /// lane guard): returns where it started and the join block.
    fn begin_guarded(&mut self, guard: Option<V>) -> Option<(usize, usize)> {
        let g = guard?;
        let (then, join) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(self.cur, Term::CondBr(g, then, join));
        let head = self.cur;
        self.cur = then;
        self.guarded += 1;
        Some((head, join))
    }

    /// Ends guarded code; each (value inside, value outside) becomes a phi.
    fn end_guarded(&mut self, at: Option<(usize, usize)>, values: &[(V, V)]) -> Vec<V> {
        let Some((head, join)) = at else {
            return values.iter().map(|v| v.0).collect();
        };
        let then = self.cur;
        self.s.m.terminate(then, Term::Br(join));
        self.cur = join;
        self.guarded -= 1;
        values
            .iter()
            .map(|&(inside, outside)| {
                let ty = self.s.m.type_of(inside);
                let phi = self.s.m.phi(join, ty);
                self.s.m.set_phi(phi, vec![(inside, then), (outside, head)]);
                phi
            })
            .collect()
    }

    /// Code that depends on the lane layout: wave operations, and in compute
    /// the console's wave size (lanes are thread ids in group order there).
    /// Pixel lanes match only within a quad.
    fn cross_lane(&mut self, quad: bool) -> Result<(), Why> {
        if self.s.mesh.is_some() {
            // shortcut: no wave size for mesh shaders in SM 6.6, so host
            // lanes cannot follow the console's; emulate through group-shared
            // memory if mesh programs turn out to need these.
            return Err("a cross-lane operation in a mesh shader");
        }
        if self.s.threads.is_none() && !(quad && self.s.pixel) {
            return Err("a cross-lane operation outside compute");
        }
        debug_assert!(self.s.threads.is_none() || self.console_lanes);
        self.s.flags |= flags::WAVE_OPS;
        Ok(())
    }

    /// Component `c` of the thread id in the group. With console lanes, the
    /// thread is relabelled so that its host lane is its console lane: the
    /// flat id is the wave's first lane's, rounded down to the wave size,
    /// plus the lane index. (A bijection while each host wave holds a
    /// wave-aligned run of flat ids, as the console's waves do.)
    fn thread_id(&mut self, c: u8) -> Val {
        if !self.console_lanes {
            return self.compute_id("threadIdInGroup.i32", 95, c);
        }
        let [x, y, _] = self.s.threads.expect("compute");
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let f = if self.halves.is_some() {
            self.halves_flat()
        } else {
            let flat = self.s.op(
                e,
                "flattenedThreadIdInGroup.i32",
                Attr::ReadNone,
                i32t,
                96,
                &[],
            );
            let first = self.s.op(
                e,
                "waveReadLaneFirst.i32",
                Attr::NoUnwind,
                i32t,
                118,
                &[flat],
            );
            let wave = self.s.wave_size.expect("set with console lanes");
            let mask = self.s.m.i32(!(wave - 1));
            let base = self.s.m.bin(e, Bin::And, first, mask);
            let lane = self.lane_index();
            self.s.m.bin(e, Bin::Add, base, lane)
        };
        self.s.flags |= flags::WAVE_OPS;
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        Val::I32(match c {
            0 => self.s.m.bin(e, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                self.s.m.bin(e, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(e, Bin::UDiv, f, xy),
        })
    }

    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    fn mesh_prologue(&mut self, mesh: super::shader::Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        let flat = self.mesh_thread();
        let guard = (threads > mesh.max_primitives).then(|| self.s.m.cmp(e, pred::IULT, flat, p));
        let at = self.begin_guarded(guard);
        let z = self.s.m.i32(0);
        self.s.op(
            self.cur,
            "emitIndices",
            Attr::NoUnwind,
            void,
            169,
            &[flat, z, z, z],
        );
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }

    /// Halves: the thread's flat id in the group, relabelled so that its
    /// host lane is its console lane modulo 32 (as [`Self::thread_id`]: a
    /// bijection while each host wave holds a 32-aligned run of flat ids).
    fn halves_flat(&mut self) -> V {
        if let Some(f) = self.halves.as_ref().and_then(|h| h.flat) {
            return f;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        self.s.flags |= flags::WAVE_OPS;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let mask = self.s.m.i32(!31);
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self
            .s
            .op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.halves.as_mut().expect("halves").flat = Some(f);
        f
    }

    /// Halves: whether a wave operation in the block being lowered is the
    /// console's as the host wave takes it: the group is one host wave, or
    /// one thread runs the block alone.
    fn half_alone(&self) -> bool {
        self.halves
            .as_ref()
            .is_none_or(|h| h.whole || h.solo[self.cur_ir])
    }

    /// Halves: whether the two halves can exchange in IR block `b`.
    fn halves_meet(&self, b: usize) -> bool {
        self.halves.as_ref().is_some_and(|h| h.meet[b])
    }

    /// Halves: whether they can with every lane of both there.
    fn halves_full(&self, b: usize) -> bool {
        self.halves.as_ref().is_some_and(|h| h.full[b])
    }

    /// Halves: the form is not the console's program (it is made again
    /// without, [`lower_with_layout`]).
    fn refuse(&self, why: Why) {
        if self.refused.get().is_none() {
            self.refused.set(Some(why));
        }
        if self.halves.is_some() {
            NEEDS.with_borrow_mut(|n| n.push(self.cur_ir));
        }
    }

    /// The console's decision of the branch closing IR block `b`, which the
    /// whole wave takes ([`Structured::wide`]): `when` is the lane's own
    /// bit, and the branch is taken when some lane's holds (`execnz`,
    /// `vccnz`, SCC of a mask operation set) or when every lane's does
    /// (`execz`, `vccz`), [`Structured::wave_any`]. Every lane of the wave
    /// is here, EXEC being a guard: a lane outside it says what it says on
    /// the console (no set bit for "some lane", no objection to "every
    /// lane"). Each host wave asks its lanes and the halves exchange the
    /// answers, so both go the same way and can meet again inside.
    fn wave_decision(&mut self, b: usize) -> Result<V, Why> {
        const WHY: Why = "a wave-wide branch where the halves cannot meet";
        let any = self.x.st.wave_any[b].ok_or("a wave-wide branch on a scalar condition")?;
        let &when = self
            .x
            .st
            .lane_when
            .get(&b)
            .ok_or("a wave-wide branch without its mask")?;
        let c = self.cur;
        let mine = self.boolean(when, c)?;
        let (i1t, i32t) = (self.s.t.i1, self.s.t.i32);
        self.s.flags |= flags::WAVE_OPS;
        let (name, code) = if any {
            ("waveAnyTrue", 113)
        } else {
            ("waveAllTrue", 114)
        };
        let host = self.s.op(c, name, Attr::NoUnwind, i1t, code, &[mine]);
        if self.half_alone() {
            return Ok(host);
        }
        if !self.halves_meet(b) || self.guarded != 0 {
            self.refuse(WHY);
            return Ok(host);
        }
        let v = self.s.m.cast(c, Cast::ZExt, host, i32t);
        // (a half with no lane left: no set bit, no objection)
        let absent = self.s.m.i32(u32::from(!any));
        let (lo, hi) = self.exchange(c, b, v, absent);
        let both = self
            .s
            .m
            .bin(c, if any { Bin::Or } else { Bin::And }, lo, hi);
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, pred::NE, both, z))
    }

    /// Halves: `v`, which every lane of a half holds alike, as the lower
    /// and the upper half of this thread's console wave hold it. Every lane
    /// stores its half's dword (each exchange has its own), the group
    /// meets, and each reads both. `at` is in IR block `b`, where the
    /// halves meet; where that is in a loop they meet again after reading,
    /// so that no half stores the next pass's before the other has read.
    ///
    /// Where threads may have ended before `b` a half may have none left
    /// to store: every thread first sets its half's dword to `absent` (a
    /// constant: no lane in a mask, a reduction's identity) in the entry
    /// block, which all run, so such a half reads as that. Its threads set
    /// it before they ended, and the barrier lets the rest go on only once
    /// they have. In a loop a half's threads may all end between two
    /// passes, with what they stored in the last one still there: after
    /// the second meeting every thread sets its half's dword to `absent`
    /// again, and all meet a third time, so that none does so after
    /// another of its half has stored the next pass's.
    fn exchange(&mut self, at: usize, b: usize, v: V, absent: V) -> (V, V) {
        let i32t = self.s.t.i32;
        let h = self.halves.as_mut().expect("halves");
        let (count, n, full) = (h.count, h.sites, h.full[b]);
        h.sites += 1;
        let arr = self.s.m.array(count, i32t);
        let g = self.s.m.global(&format!("halves{n}"), arr, 3);
        let f = self.halves_flat();
        let (five, one) = (self.s.m.i32(5), self.s.m.i32(1));
        // (Past a count of threads a half may have none even where every
        // thread that started is: `counted`.)
        if !full || self.counted.is_some() && matches!(absent, V::Const(_)) {
            let e = self.s.entry;
            let half = self.s.m.bin(e, Bin::LShr, f, five);
            let own = self.s.m.element(e, g, half);
            self.s.m.store(e, own, absent);
        }
        let half = self.s.m.bin(at, Bin::LShr, f, five);
        let own = self.s.m.element(at, g, half);
        self.s.m.store(at, own, v);
        let (mode, void) = (self.s.m.i32(1 | 8), self.s.t.void);
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        let lower = self.s.m.i32(!1);
        let lo = self.s.m.bin(at, Bin::And, half, lower);
        let hi = self.s.m.bin(at, Bin::Or, half, one);
        let (lo, hi) = (self.s.m.element(at, g, lo), self.s.m.element(at, g, hi));
        let r = (self.s.m.load(at, lo), self.s.m.load(at, hi));
        if !self.halves.as_ref().expect("halves").once[b] {
            self.s
                .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
            if !full {
                self.s.m.store(at, own, absent);
                self.s
                    .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
            }
        }
        r
    }

    /// Halves: whether this thread is in the upper half of its console wave.
    fn upper_half(&mut self, at: usize) -> V {
        let f = self.halves_flat();
        let (k32, z) = (self.s.m.i32(32), self.s.m.i32(0));
        let t = self.s.m.bin(at, Bin::And, f, k32);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    /// Whether every lane that ran the instruction writing VGPR value `v`
    /// holds the same `v`: made of SGPRs, constants and such values alone,
    /// by an instruction every lane reaching it runs.
    fn lane_uniform(&self, v: Value, depth: u32) -> bool {
        let p = self.x.p;
        let Some(&b) = self.def_block.get(&v) else {
            return false;
        };
        let Some((i, inst)) = p.blocks[b]
            .insts
            .iter()
            .enumerate()
            .find(|(_, i)| i.writes.iter().any(|w| w.1 == v))
        else {
            return false;
        };
        let name = inst.name();
        // A buffer dword at an offset no lane adds to, in a program that
        // writes no buffer: every lane loads the same.
        if let Fields::Buffer {
            soffset,
            offen: false,
            idxen: false,
            lds: false,
            ..
        } = &inst.fields
        {
            return name.starts_with("buffer_load_dword")
                && self.x.st.guards[b][i].is_none()
                && matches!(soffset, Operand::Const(_) | Operand::Scalar { .. })
                && !p.blocks.iter().flat_map(|b| &b.insts).any(|i| {
                    matches!(i.fields, Fields::Buffer { .. })
                        && (i.name().contains("store") || i.name().contains("atomic"))
                });
        }
        let Fields::Valu(valu) = &inst.fields else {
            return false;
        };
        if depth == 0
            || self.x.st.guards[b][i].is_some()
            || valu.sdwa.is_some()
            || valu.dpp.is_some()
            || valu.carry.is_some()
            || !matches!(valu.dst, Some(Operand::Vector { .. }))
            || ["lane", "mbcnt", "cndmask", "_co_", "_ci_", "addc", "subb"]
                .iter()
                .any(|n| name.contains(n))
        {
            return false;
        }
        valu.src.iter().all(|op| match *op {
            // (an SGPR joined where lanes arrive by their own branches is
            // what each lane's path brought)
            Operand::Scalar { code, dwords } => {
                code <= 105
                    && (0..dwords).all(|k| {
                        inst.read(Reg::S(code + k)).is_none_or(|a| {
                            self.phi_at
                                .get(&a)
                                .is_none_or(|&(j, _)| self.uniform_join(j))
                        })
                    })
            }
            Operand::Vector { reg, dwords } => (0..dwords).all(|k| {
                inst.read(Reg::V(reg + k))
                    .is_some_and(|a| self.lane_uniform(a, depth - 1))
            }),
            Operand::Const(c) => c <= 208 || (240..=248).contains(&c),
            Operand::Literal(_) => true,
            Operand::Attr { .. } => false,
        })
    }

    /// The console wave size.
    fn wave(&self) -> u32 {
        if self.x.native.wave64 { 64 } else { 32 }
    }

    fn lane_index(&mut self) -> V {
        if self.s.mesh.is_some() {
            // The console's lane: mesh code reads no other lanes, so host
            // waves need not match the console's.
            let f = self.mesh_thread();
            let m = self.s.m.i32(self.wave() - 1);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        if self.halves.is_some() {
            // The console's lane: the host's plus 32 in an upper half.
            let f = self.halves_flat();
            let m = self.s.m.i32(63);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        self.s.op(
            self.cur,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        )
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    /// Whether a quad-local read must reach helper lanes: in a pixel program
    /// every quad a triangle edge crosses has some (the console's whole-quad
    /// mode runs them; discarded pixels stay helpers too), and only quad
    /// reads see them on the host (wave reads give 0 from them on AMD: wrong
    /// derivatives of computed values along every triangle edge).
    fn helpers_read(&self, quad: bool) -> bool {
        quad && self.s.pixel
    }

    /// `v_readfirstlane_b32`'s result `first` of `own` in a pixel shader:
    /// a helper lane takes its own value instead.
    ///
    /// On the console, whole-quad mode puts helper lanes in EXEC, so a
    /// waterfall loop (read the first lane's value, run the lanes with that
    /// value, take them out of EXEC) serves helper lanes too. On the host,
    /// wave operations leave helper lanes out (DXIL before
    /// `WaveOpsIncludeHelperLanes`): once only helper lanes are left in such
    /// a loop, `WaveReadLaneFirst` reads no lane, no lane matches, and the
    /// loop runs to [`LOOP_LIMIT`] in every such lane: draws of 0.3 to 1.3 s
    /// each, then DEVICE_HUNG. With its own value a helper lane matches in
    /// the first pass and leaves; what a helper lane computes only feeds
    /// derivatives, and its stores are dropped.
    fn helper_reads_own(&mut self, own: V, first: V) -> V {
        if !self.s.pixel {
            return first;
        }
        let c = self.cur;
        let helper = self
            .s
            .op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        self.s.m.select(c, helper, own, first)
    }

    /// `x` of console lane `lane`, a lane of this lane's group of 32 (its
    /// host wave when the program runs as halves).
    fn read_near(&mut self, x: V, lane: V) -> V {
        if self.halves.is_none() {
            return self.read_lane(x, lane);
        }
        let m = self.s.m.i32(31);
        let lane = self.s.m.bin(self.cur, Bin::And, lane, m);
        self.read_lane(x, lane)
    }

    fn read_lane(&mut self, x: V, lane: V) -> V {
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// The float value of a register value.
    fn float_of(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&f) = self.floats.get(&v) {
            return Ok(f);
        }
        let x = self.i32_of(v)?;
        Ok(self.as_float(x))
    }

    /// The values of VGPRs `reg..reg + n` as `inst` reads them.
    fn vgprs(&mut self, inst: &Inst, op: &Operand, n: u8) -> Result<Vec<V>, Why> {
        let Operand::Vector { reg, .. } = *op else {
            return Err("a vector operand");
        };
        (0..n)
            .map(|k| {
                let v = inst.read(Reg::V(reg + k)).ok_or("an unread operand")?;
                self.i32_of(v)
            })
            .collect()
    }

    /// LDS (thread-group shared memory) and `ds_swizzle`.
    fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *gds {
            return Err("GDS");
        }
        let name = inst.name();
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        if name == "ds_swizzle_b32" {
            // Bit 15 set: each lane of a quad picks a lane of it; else lanes
            // of each group of 32 combine their index with and/or/xor masks.
            let quad = offset & 0x8000 != 0
                || offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4;
            // Mesh: host quads are threads 4n..4n+3 of a 1D group, as the
            // console's, so quad-local swizzles read through quad operations.
            let mesh_quad = self.s.mesh.is_some() && quad;
            if !mesh_quad {
                self.cross_lane(quad)?;
            }
            // The source sits in the address field.
            let x = self.vgprs(inst, addr.as_ref().ok_or("a swizzle source")?, 1)?[0];
            let lane = self.lane_index();
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let src = if offset & 0x8000 != 0 {
                let (three, one) = (k(self, 3), k(self, 1));
                let sel = self.s.m.bin(c, Bin::And, lane, three);
                let sh = self.s.m.bin(c, Bin::Shl, sel, one);
                let pattern = k(self, offset & 0xFF);
                let pick = self.s.m.bin(c, Bin::LShr, pattern, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, three);
                let not3 = k(self, !3);
                let base = self.s.m.bin(c, Bin::And, lane, not3);
                self.s.m.bin(c, Bin::Or, base, pick)
            } else {
                let and = k(self, offset & 0x1F | !0x1F);
                let or = k(self, offset >> 5 & 0x1F);
                let xor = k(self, offset >> 10 & 0x1F);
                let t = self.s.m.bin(c, Bin::And, lane, and);
                let t = self.s.m.bin(c, Bin::Or, t, or);
                self.s.m.bin(c, Bin::Xor, t, xor)
            };
            // (a quad or a group of 32: lanes of one half)
            let r = if mesh_quad || self.helpers_read(quad) {
                self.quad_pick(x, src)
            } else {
                self.read_near(x, src)
            };
            let reg = dst_reg.ok_or("a swizzle result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if name == "ds_bpermute_b32" {
            // Each lane reads `data` of the lane its address plus the offset
            // names in dwords (modulo the wave); 0 from a lane EXEC leaves out.
            self.cross_lane(false)?;
            if self.halves.as_ref().is_some_and(|h| !h.whole) {
                self.refuse("ds_bpermute across two host waves");
            }
            let a = self.vgprs(inst, addr.as_ref().ok_or("a permute address")?, 1)?[0];
            let x = self.vgprs(inst, data.first().ok_or("a permute source")?, 1)?[0];
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let (o, two, last) = (k(self, offset), k(self, 2), k(self, self.wave() - 1));
            let at = self.s.m.bin(c, Bin::Add, a, o);
            let lane = self.s.m.bin(c, Bin::LShr, at, two);
            let lane = self.s.m.bin(c, Bin::And, lane, last);
            let r = self.read_lane(x, lane);
            let on = guard.unwrap_or_else(|| self.s.m.bool(true));
            let i32t = self.s.t.i32;
            let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
            let ballot = self
                .s
                .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
            let (lo, hi) = (
                self.s.m.extract(c, ballot, 0),
                self.s.m.extract(c, ballot, 1),
            );
            let running = self.lane_bit_of(c, lane, lo, hi);
            let z = k(self, 0);
            let r = self.s.m.select(c, running, r, z);
            let reg = dst_reg.ok_or("a permute result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if self.s.lds_dwords == 0 {
            return Err("LDS the program does not allocate");
        }
        let a = self.vgprs(inst, addr.as_ref().ok_or("an LDS address")?, 1)?[0];
        // shortcut: LDS addresses are taken as aligned to what they access
        // and in bounds (out-of-bounds console accesses read 0 and drop
        // writes).
        let dword = |l: &mut Self, bytes: u32| -> V {
            let c = l.cur;
            let o = l.s.m.i32(bytes);
            let at = l.s.m.bin(c, Bin::Add, a, o);
            let two = l.s.m.i32(2);
            let index = l.s.m.bin(c, Bin::LShr, at, two);
            let lds = l.s.lds();
            l.s.m.element(c, lds, index)
        };
        let width = |n: &str| match n.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            _ => 1,
        };
        // (byte offset, dwords) per element; read2/write2 scale their two
        // offsets by the element size (×64 for st64).
        let pair = |n: &str| -> Vec<(u32, u32)> {
            let w = width(n);
            let scale = 4 * w * if n.contains("st64") { 64 } else { 1 };
            vec![
                (u32::from(*offset0) * scale, w),
                (u32::from(*offset1) * scale, w),
            ]
        };
        if let Some(rest) = name.strip_prefix("ds_read") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                // A byte or half, extended (`_d16`: into the low half,
                // `_d16_hi`: the high half, the other half kept).
                let (bits, signed) = match rest.trim_end_matches("_hi").trim_end_matches("_d16") {
                    "_u8" => (8, false),
                    "_i8" => (8, true),
                    "_u16" => (16, false),
                    "_i16" => (16, true),
                    _ => return Err("an LDS read of bytes or halves"),
                };
                let reg = dst_reg.ok_or("an LDS read result")?;
                let p = dword(self, offset);
                let x = self.s.m.load(c, p);
                let sh = self.lds_shift(a, offset);
                let x = self.s.m.bin(c, Bin::LShr, x, sh);
                let left = self.s.m.i32(32 - bits);
                let x = self.s.m.bin(c, Bin::Shl, x, left);
                let x = self
                    .s
                    .m
                    .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left);
                let x = if rest.contains("_d16") {
                    let old = match inst.read(Reg::V(reg)) {
                        Some(o) => self.i32_of(o)?,
                        None => self.s.m.i32(0),
                    };
                    let hi = rest.ends_with("_hi");
                    let keep = self.s.m.i32(if hi { 0xFFFF } else { 0xFFFF_0000 });
                    let old = self.s.m.bin(c, Bin::And, old, keep);
                    let x = if hi {
                        let k16 = self.s.m.i32(16);
                        self.s.m.bin(c, Bin::Shl, x, k16)
                    } else {
                        let low = self.s.m.i32(0xFFFF);
                        self.s.m.bin(c, Bin::And, x, low)
                    };
                    self.s.m.bin(c, Bin::Or, old, x)
                } else {
                    x
                };
                self.write(inst, Reg::V(reg), x, guard);
                return Ok(());
            };
            let reg = dst_reg.ok_or("an LDS read result")?;
            let mut n = 0;
            for (off, w) in elements {
                for k in 0..w {
                    let p = dword(self, off + 4 * k);
                    let x = self.s.m.load(self.cur, p);
                    self.write(inst, Reg::V(reg + n), x, guard);
                    n += 1;
                }
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_write") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                // A byte or half (`_d16_hi`: the high half of the data) into
                // its dword through atomics, so lanes writing the other parts
                // of the dword keep theirs.
                // shortcut: lanes writing the same part in one instruction
                // leave the OR of theirs (the console keeps one).
                let bits = match rest.trim_end_matches("_d16_hi") {
                    "_b8" => 8,
                    "_b16" => 16,
                    _ => return Err("an LDS write of bytes or halves"),
                };
                let x = self.vgprs(inst, data.first().ok_or("an LDS write")?, 1)?[0];
                let x = if rest.ends_with("_d16_hi") {
                    let k16 = self.s.m.i32(16);
                    self.s.m.bin(c, Bin::LShr, x, k16)
                } else {
                    x
                };
                let mask = self.s.m.i32((1 << bits) - 1);
                let x = self.s.m.bin(c, Bin::And, x, mask);
                let sh = self.lds_shift(a, offset);
                let set = self.s.m.bin(c, Bin::Shl, x, sh);
                let part = self.s.m.bin(c, Bin::Shl, mask, sh);
                let ones = self.s.m.i32(u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, part, ones);
                let p = dword(self, offset);
                let at = self.begin_guarded(guard);
                use super::bitcode::Rmw;
                self.s.m.atomic(self.cur, Rmw::And, p, keep);
                self.s.m.atomic(self.cur, Rmw::Or, p, set);
                self.end_guarded(at, &[]);
                return Ok(());
            };
            let mut stores = Vec::new();
            for (e, (off, w)) in elements.into_iter().enumerate() {
                let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
                for (k, x) in vals.into_iter().enumerate() {
                    stores.push((off + 4 * k as u32, x));
                }
            }
            let at = self.begin_guarded(guard);
            for (off, x) in stores {
                let p = dword(self, off);
                self.s.m.store(self.cur, p, x);
            }
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Atomics: the old value returns with `_rtn`.
        let op = name
            .strip_prefix("ds_")
            .map(|n| n.replace("_rtn", ""))
            .ok_or("an LDS operation")?;
        use super::bitcode::Rmw;
        let rmw = match op.as_str() {
            "add_u32" => Some(Rmw::Add),
            "sub_u32" => Some(Rmw::Sub),
            "and_b32" => Some(Rmw::And),
            "or_b32" => Some(Rmw::Or),
            "xor_b32" => Some(Rmw::Xor),
            "min_i32" => Some(Rmw::Min),
            "max_i32" => Some(Rmw::Max),
            "min_u32" => Some(Rmw::UMin),
            "max_u32" => Some(Rmw::UMax),
            "wrxchg_b32" => Some(Rmw::Xchg),
            "cmpst_b32" => None,
            _ => return Err("an LDS operation"),
        };
        let v0 = self.vgprs(inst, data.first().ok_or("an LDS atomic")?, 1)?[0];
        let v1 = match data.get(1) {
            Some(d) => Some(self.vgprs(inst, d, 1)?[0]),
            None => None,
        };
        let old_reg = dst_reg.and_then(|r| inst.read(Reg::V(r)).map(|v| (r, v)));
        let outside = match old_reg {
            Some((_, v)) => self.i32_of(v).unwrap_or(self.s.m.i32(0)),
            None => self.s.m.i32(0),
        };
        let p = dword(self, offset);
        let at = self.begin_guarded(guard);
        let r = match rmw {
            Some(op) => self.s.m.atomic(self.cur, op, p, v0),
            // Compare with data0, store data1.
            None => self
                .s
                .m
                .cmpxchg(self.cur, p, v0, v1.ok_or("an LDS compare")?),
        };
        let r = self.end_guarded(at, &[(r, outside)])[0];
        if let Some(reg) = dst_reg {
            self.write(inst, Reg::V(reg), r, None);
        }
        Ok(())
    }

    /// Thread-group sync with a shared-memory fence, and a UAV fence when
    /// the program writes UAVs (the waits before an s_barrier make those
    /// writes visible on the console).
    fn group_barrier(&mut self) {
        let mode = if self.writes_uavs { 1 | 2 | 8 } else { 1 | 8 };
        let mode = self.s.m.i32(mode);
        let void = self.s.t.void;
        self.s
            .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    // ---- instructions ----

    fn inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let offset = inst.machine.offset;
        let name = inst.name();
        self.mask_ok = match self.mask_data {
            MaskData::Console => true,
            MaskData::Host => {
                name.starts_with("s_bcnt1_i32_")
                    || name.starts_with("s_ff1_i32_")
                    || name.starts_with("v_mbcnt_")
                    || matches!(
                        name,
                        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
                    )
            }
            MaskData::None => name == "s_bcnt1_i32_b64",
        };
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read")
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute"));
        let fenced = self.fenced(name);
        if fenced {
            self.memory_fence();
        }
        let r = match &inst.fields {
            Fields::Sop { .. } => self.sop(inst),
            Fields::Sopp { .. } if name == "s_barrier" && self.no_barrier.contains(&b) => Ok(()),
            Fields::Sopp { .. } if name == "s_barrier" => {
                self.group_barrier();
                Ok(())
            }
            Fields::Sopp { .. } => Ok(()),
            Fields::Smem { .. } => self.smem(inst, (b, i)),
            Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
            Fields::Valu(_) => self.valu(inst, b, i),
            Fields::Vintrp { .. } => self.interp(inst, (b, i)),
            Fields::Image { .. } => self.image(inst, b, i),
            Fields::Buffer { .. } => self.buffer(inst, b, i),
            Fields::Exp { .. } => self.export(b, i),
            Fields::Ds { .. } => self.ds(inst, b, i),
            Fields::Flat { .. } if name.starts_with("global_") => self.flat(inst, b, i),
            Fields::Flat { .. } => Err("flat memory"),
        };
        if fenced && r.is_ok() {
            self.memory_fence();
        }
        match r {
            Ok(()) => Ok(()),
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                for &(_, v) in &inst.writes {
                    self.vals.insert(v, Val::Missing(why));
                }
                Ok(())
            }
        }
    }

    fn is_mask(&self, v: Option<Value>) -> bool {
        v.is_some_and(|v| self.x.st.masks.contains_key(&v))
    }
}

impl<'a> Lower<'a> {
    fn lane(&mut self, inst: &Inst, value: Option<bool>) {
        let b = self.s.m.bool(value == Some(true));
        self.lane_bits_for(inst, b);
    }

    fn lane_bits_for(&mut self, inst: &Inst, r: V) {
        let Fields::Valu(v) = &inst.fields else {
            return;
        };
        let code = if inst.name().starts_with("v_cmpx") {
            Some(EXEC)
        } else {
            match v.dst {
                Some(Operand::Scalar { code, .. }) => Some(code),
                _ => None,
            }
        };
        if let Some(lo) = code.and_then(|c| inst.write(Reg::S(c))) {
            self.lane_bits.insert(lo, r);
        }
        for &(_, w) in &inst.writes {
            self.vals
                .entry(w)
                .or_insert(Val::Missing("a lane mask read as data"));
        }
    }

    fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Some(&interp) = self.interps.get(&site) else {
            return Err("the first half of an interpolation");
        };
        let (attribute, mode) = match interp {
            Interp::Interpolated { unless, .. } if !self.avoids(site, unless) => {
                return Err("an interpolation with computed barycentrics");
            }
            Interp::Interpolated {
                attribute,
                barycentric,
                ..
            } => (
                attribute,
                interp_mode(*barycentric).ok_or("pull-model interpolation")?,
            ),
            // P0, the provoking vertex: a flat (nointerpolation) input.
            Interp::Vertex {
                attribute,
                vertex: 2,
            } => (attribute, 1),
            Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
        };
        let dst = match &inst.fields {
            Fields::Vintrp { dst, .. } => *dst,
            Fields::Valu(v) => v.dst.ok_or("an interpolation")?,
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else {
            return Err("an interpolation");
        };
        // A channel of an input element, interpolated as `mode` says.
        let channel = |l: &mut Self, row: u8, ch: u8| -> Result<V, Why> {
            let (id, col) = l
                .find_input(row, ch)
                .ok_or("an input the signature lacks")?;
            let e = &mut l.s.inputs[id];
            if e.interp != mode && e.usage != 0 {
                return Err("one input interpolated two ways");
            }
            e.interp = mode;
            match l.load_any_input(id, col) {
                Val::I32(x) => Ok(x),
                _ => Err("an input"),
            }
        };
        let packing = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => inputs
                .iter()
                .find(|i| i.row == attribute.row && i.fp16)
                .map(|i| i.pack2),
            _ => None,
        };
        let f16 = inst.name().ends_with("_f16");
        let x = match packing {
            None if f16 || attribute.high => return Err("a 16-bit read of a 32-bit input"),
            None => channel(self, attribute.row, attribute.channel)?,
            Some(pack2) => {
                // 16-bit values: one in the low half of each dword, or two
                // per dword (`packed_half`); the host's are 32-bit floats.
                let half = |l: &mut Self, h: u8| -> Result<Option<V>, Why> {
                    let src = if pack2 {
                        packed_half(&l.s.inputs, attribute.row, h)
                    } else {
                        h.is_multiple_of(2).then_some((attribute.row, h / 2))
                    };
                    let Some((row, ch)) = src else {
                        return Ok(None);
                    };
                    let x = channel(l, row, ch)?;
                    let f = l.as_float(x);
                    Ok(Some(l.narrow16(f)))
                };
                let h = 2 * attribute.channel;
                let c = self.cur;
                if f16 {
                    let lo = half(self, h + u8::from(attribute.high))?
                        .ok_or("a packed input half no element holds")?;
                    // omod and clamp are part of the f16 instruction; op_sel[3]
                    // picks the destination half.
                    let lo = self.half_modifiers(inst, lo);
                    return self.write_half(inst, lo, None);
                } else if mode == 1 {
                    // The whole dword of a flat input: both halves.
                    let z = self.s.m.i32(0);
                    let lo = half(self, h)?.unwrap_or(z);
                    let hi = half(self, h + 1)?.unwrap_or(z);
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        // A pixel interpolates in every lane; guards only hide inactive ones.
        self.write(inst, Reg::V(reg), x, None);
        Ok(())
    }

    /// Whether no lane that runs the instruction at `site` has been where
    /// `unless` says: a lane that came along such an edge or ran (skipped)
    /// such an instruction meets its condition, which the instruction's must
    /// contradict. Those lane bits must not change in between, so none of
    /// those blocks is on a cycle. A lane mask joined where such an edge
    /// enters holds, for the lanes that came along it, the edge's mask.
    fn avoids(&self, (b, i): Site, unless: &[Elsewhere]) -> bool {
        let st = self.x.st;
        let runs = [
            (st.reach[b], true),
            (st.guards[b][i].unwrap_or(Bool::TRUE), true),
        ];
        let mut atoms = st.bools.support(runs[0].0);
        atoms.extend(st.bools.support(runs[1].0));
        unless.iter().all(|&u| {
            let (mut all, at, mut links) = match u {
                Elsewhere::Edge(q, to) => {
                    let all = self.lane_edge(q, to);
                    let block = &self.x.p.blocks[to];
                    let n = block.preds.iter().position(|&p| p == q).expect("an edge");
                    let links: Vec<(Bool, Bool)> = block
                        .phis
                        .iter()
                        .filter(|ph| atoms.contains(&Atom::Phi(ph.value)))
                        .filter_map(|ph| {
                            Some((*st.masks.get(&ph.value)?, *st.masks.get(&ph.args[n])?))
                        })
                        .collect();
                    (all, to, links)
                }
                Elsewhere::Write((w, k)) | Elsewhere::Skip((w, k)) => {
                    let guard = st.guards[w][k].unwrap_or(Bool::TRUE);
                    let ran = matches!(u, Elsewhere::Write(_));
                    (vec![(st.reach[w], true), (guard, ran)], w, vec![])
                }
            };
            all.extend(runs);
            // Each joined mask equals the incoming one: try both values.
            links.truncate(6);
            let some_lane = (0..1u32 << links.len()).any(|bits| {
                let mut all = all.clone();
                for (k, &(joined, incoming)) in links.iter().enumerate() {
                    let v = bits >> k & 1 != 0;
                    all.extend([(joined, v), (incoming, v)]);
                }
                st.bools.meet(&all)
            });
            !self.on_cycle(at) && !some_lane
        })
    }

    /// What holds in a lane that goes from IR block `q` to `b`.
    fn lane_edge(&self, q: usize, b: usize) -> Vec<(Bool, bool)> {
        let st = self.x.st;
        let mut all = vec![(st.reach[q], true)];
        if let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = st.exits[q]
            && taken != not_taken
        {
            all.push((when, taken == b));
        }
        all
    }

    /// Whether every lane enters IR block `b` along the edge the console's
    /// wave takes, so that an SGPR joined there holds what that edge
    /// brought: every branch into it is uniform (SCC values). The wave runs
    /// one path; a per-lane branch into the join (an EXEC skip, a loop's
    /// per-lane exit) is where lanes would take edges the wave did not, and
    /// the console compiler's if/else always skips into its join that way.
    /// Lanes that left EXEC inside a uniform loop (the BVH walks) take no
    /// edge into its header; the others take the wave's.
    fn uniform_join(&self, b: usize) -> bool {
        let st = self.x.st;
        let preds = &self.x.p.blocks[b].preds;
        preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .all(|&q| match st.exits[q] {
                LaneExit::Branch {
                    when,
                    taken,
                    not_taken,
                } if taken != not_taken => st
                    .bools
                    .support(when)
                    .iter()
                    .all(|a| matches!(a, Atom::Uniform(_) | Atom::Wave(_))),
                _ => true,
            })
    }

    /// Whether no lane goes from IR block `q` to `b`.
    fn never_taken(&self, q: usize, b: usize) -> bool {
        !self.x.st.bools.meet(&self.lane_edge(q, b))
    }

    /// Whether, at a per-lane join `b` of a lane mask whose register holds
    /// data along edge `n`, some edge brings a mask and no lane taking edge
    /// `n` is in any of them (a compare inside a region those lanes skip).
    /// The console wave runs one path: on a mask edge, the lanes of edge `n`
    /// read their bit of that mask, 0. The compiler merges no value for them,
    /// so it holds none for them either where the whole wave skips (the data
    /// there is a reused register): 0 is theirs.
    fn no_mask_lane_on_edge(&self, b: usize, args: &[Value], n: usize) -> bool {
        let st = self.x.st;
        let preds = &self.x.p.blocks[b].preds;
        let on_n = self.lane_edge(preds[n], b);
        let mut some = false;
        for (k, arg) in args.iter().enumerate() {
            let Some(&m) = st.masks.get(arg).filter(|_| k != n) else {
                continue;
            };
            some = true;
            let mut lanes = on_n.clone();
            lanes.push((m, true));
            if st.bools.meet(&lanes) {
                return false;
            }
        }
        some
    }

    /// Whether IR block `b` can reach itself.
    fn on_cycle(&self, b: usize) -> bool {
        let blocks = &self.x.p.blocks;
        let mut seen = vec![false; blocks.len()];
        let mut stack = blocks[b].exit.successors();
        while let Some(x) = stack.pop() {
            if x == b {
                return true;
            }
            if !std::mem::replace(&mut seen[x], true) {
                stack.extend(blocks[x].exit.successors());
            }
        }
        false
    }

    /// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit
    /// results keep the register's high half (GFX10).
    fn low_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let old = self.s.m.bin(c, Bin::And, old, high);
        let x = self.s.m.bin(c, Bin::And, x, low);
        self.s.m.bin(c, Bin::Or, old, x)
    }

    /// VOP3 f16 output modifiers: omod, then clamp, applied in f32 so the
    /// half conversion rounds once (the order `write_f` uses).
    fn half_modifiers(&mut self, inst: &Inst, h: V) -> V {
        let Fields::Valu(v) = &inst.fields else {
            return h;
        };
        if v.omod == 0 && !v.clamp {
            return h;
        }
        let c = self.cur;
        let f32t = self.s.t.f32;
        let mut f = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        if v.omod != 0 {
            let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(v.omod)];
            let k = self.s.m.f32(factor.to_bits());
            f = self.s.m.bin(c, Bin::Mul, f, k);
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        self.s
            .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[f])
    }
}

#[cfg(test)]
mod tests;
