//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it.
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host. One
//!   of two immediate samplers chosen by uniform control flow takes two
//!   entries; the sampling runs with both and the result is chosen.
//! * **Root CBVs** as root descriptors `b<i>` in [`ROOT_CBV_SPACE`], `i`
//!   the title's root parameter index ([`HostShader::root_cbvs`]); the
//!   address the title sets is the buffer's.
//! * **Root SRVs and root UAVs** (raw buffers; changed 2026-11-24) are
//!   resource-heap views the host makes: a raw SRV (UAV) of the whole host
//!   copy holding the root address, at the heap index the host writes into
//!   the [`HostShader::user_data`] entry marked [`ROOT_SRV_VIEW`]
//!   ([`ROOT_UAV_VIEW`]) (dword `index` = the root descriptor's own first
//!   user-data dword, which such a program never reads as data), and the
//!   root address's byte offset in that copy into the entry marked
//!   [`ROOT_VIEW_OFFSET`] (its second); the shader adds that offset to
//!   every offset it reads at. Host root descriptors have no bounds, so a
//!   shader reading past the copy read other memory or faulted; through
//!   the view it reads 0 and its writes there are dropped. One view per
//!   host copy (and kind) serves every draw of a list. In programs that
//!   use title memory (below) root SRVs and UAVs are title memory instead;
//!   elsewhere [`HostShader::root_srvs`] and [`HostShader::root_uavs`] now
//!   hold only the host's own tables (`VIEW_STRIDES`).
//! * **Vertex fetch** (added 2026-11-23). the title's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the shader reads it as the root SRV view above. The
//!   vertex index, the stride and the attribute offset are the shader's
//!   own arithmetic on data in that buffer (per-stream headers), not
//!   descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way. Buffers read by element through
//!   the resource heap (no used-resource entry, so no stride) are raw too,
//!   addressed at element × the view's stride + offset, the stride from
//!   the view-stride table: root SRV `t`[`VIEW_STRIDES`] in
//!   [`ROOT_CBV_SPACE`], listed in [`HostShader::root_srvs`], whose dword
//!   `i` the host keeps at the byte stride of the buffer view at heap
//!   index `i`.
//! * **Constant buffers in the heap** (added 2026-11-24): scalar buffer
//!   loads through a descriptor taken from the resource heap itself (root
//!   type 5, so no used-resource entry) read a raw buffer view, not a
//!   constant buffer view: the title's constant buffers are 16-byte
//!   aligned, the host's views 256. The host makes a raw view at the
//!   constant buffer's address for each constant buffer view in the heap.
//!   Constant buffers in descriptor tables are read the same way (scalar
//!   loads and per-lane rows), so tables are never copied per draw.
//! * **Rectangle lists** ([`rectangle_gs`]): a geometry shader made from a
//!   vertex shader's outputs completes each rectangle with its fourth
//!   corner opposite the right angle ([`right_angle`]), as the console's
//!   hardware does.
//! * **Heap indices** (added 2026-11-24) are clamped to [`HEAP_VIEWS`] − 1
//!   (samplers [`HEAP_SAMPLERS`] − 1): the host's heaps have at least that
//!   many slots, those past its own null, so an index past the heap (data
//!   still wrong; the console would read whatever memory follows its heap)
//!   reads nothing instead of faulting the device.
//! * **Buffer views at computed heap indices** (added 2026-11-24): a
//!   buffer read through a descriptor the shader takes from the resource
//!   heap itself first looks up the view-stride table (root SRV
//!   `t`[`VIEW_STRIDES`], which then covers every index below
//!   [`HEAP_VIEWS`], dynamic slots too); where it holds 0 (no buffer view
//!   there) the shader reads heap index [`NULL_BUFFER_VIEW`] instead (for
//!   unordered access the index below), which the host keeps a null raw
//!   buffer SRV (UAV). The index comes from data, and a texture descriptor
//!   read as a buffer one is a wild address (the texture's shifted 8 bits
//!   down: GPU faults at low addresses).
//! * **Heap access** ([`HostShader::heap`], [`HostShader::heap_writes`],
//!   added 2026-11-24): whether the shader takes descriptors from the
//!   resource heap itself, and unordered access views among them; the
//!   host then keeps everything the heap's views reach current, and
//!   copies back what such views may have written.
//!
//! * **Texture atomics** (added 2026-11-23): a texture UAV the shader uses
//!   atomically is declared a 32-bit unsigned single-channel view; the
//!   host's view at that index must be an R32 integer view. Global loads
//!   from a root SRV read it raw. 64-bit texture atomics (two dwords per
//!   value; added 2026-11-24) are SM 6.6's on typed heap resources: the
//!   host's view must be `R32G32_UINT` (low dword in R) and the device must
//!   report 64-bit atomics on typed resources (options 9) and on descriptor
//!   heap resources (options 11), else the pipeline is refused.
//!
//! * **Root parameters past the user SGPRs** (added 2026-11-24): a
//!   compute program whose root signature does not fit its 16 user SGPRs
//!   loads the rest from memory whose 40-bit address is in user data 0-1;
//!   the table starts with user-data dword 16. They are user data as any
//!   other: the host fills dword `index` as above.
//! * What shaders write through a root UAV is the title's to see.
//! * **Title memory** (added 2026-11-24, decision 0005's unified-memory
//!   item): accesses at addresses the program computes (pointers in root
//!   constants or loaded from memory, BVH nodes) go through a page table,
//!   the root SRV `t0` in [`MEMORY_SPACE`] ([`HostShader::memory`]), dwords:
//!   4096 level-1 entries for address bits 28-39 (a level-2 block number,
//!   0 for none), then blocks of 4096 entries for bits 16-27 (64 KiB pages),
//!   two dwords each: the resource-heap index of a raw SRV of the host
//!   memory holding the page, whose raw UAV is the next index, and the low
//!   dword of the title address that memory starts at (byte offset =
//!   address low dword - it). Block 0 names null views (its entry 0 is
//!   how the shader tells a miss); addresses at or above 2^40 use it. The
//!   root UAV `u0` in [`MEMORY_SPACE`] ([`HostShader::memory_writes`]) takes
//!   bit `block × 4096 + page` for each page a shader writes (pages of an
//!   access's first and last byte: what the host copies back) and for each
//!   page with no copy an access reached, or for an address whose 256 MiB
//!   region has no block, bit `level-1 index` (block 0's bits). Reads there
//!   give 0, writes are dropped. In such a program (and one with root
//!   parameters past its user SGPRs) root SRVs and UAVs are title memory
//!   too: no root views; the shader reads their addresses as user data
//!   (root type [`RootType::Descriptor`] in [`HostShader::user_data`],
//!   dword 0 the address's low dword, 1 the high), and the host must have
//!   a copy holding each such address.
//! * **View addresses** ([`HostShader::view_addresses`]): a buffer
//!   descriptor's dwords 0-1 read as numbers (the BVH SRV's base address)
//!   and an image descriptor's dwords 1-3 read only for sizes come from the
//!   root SRV `t1` in [`MEMORY_SPACE`]: per resource-heap index, four
//!   dwords, a console descriptor's first four as far as the host knows
//!   them: a buffer view's title address (high dword bits 0-15; stride bits
//!   0), a texture view's width and height - 1 (dword 1 bits 30-31, dword 2
//!   bits 0-13 and 14-29), first and last level (dword 3 bits 12-15, 16-19)
//!   and type (bits 28-31) with an identity swizzle; zeros for anything
//!   else.
//! * **Write marks** ([`HostShader::view_marks`], with
//!   [`Options::view_marks`]; added 2026-11-25): every store or atomic
//!   through an unordered buffer view from the heap (tables, the heap
//!   itself, root UAVs as their heap views) also sets bits in the root UAV
//!   `u`[`VIEW_WRITES`] in [`MEMORY_SPACE`], a bitmap of
//!   [`VIEW_WRITE_BITS`] bits, so the host copies back only what shaders
//!   wrote. The root SRV `t`[`VIEW_MARKS`] there holds two dwords per heap
//!   index: a base bit and a phase (0-65535); a write of byte `x` of the
//!   view (its offset in the view, an element times the stride plus the
//!   offset) sets bit `base + ((phase + x) >> 16)`, for the access's first
//!   and last byte, at most [`VIEW_WRITE_BITS`] − 1. Base
//!   [`VIEW_MARK_NONE`]: nothing is marked. The host gives each copy of
//!   title memory a run of bits, one per 64 KiB title page it touches, and
//!   a view the bit of its first byte's page as base with that byte's
//!   offset in its page as phase. Writes at computed addresses mark the
//!   page-table bitmap instead ("Title memory").
//! * **Atomic locks** ([`Options::atomic_locks`]; added 2026-11-25): a
//!   64-bit image atomic (an R32G32_UINT view) takes a lock word, reads
//!   the texel, writes the result and frees the lock, in a loop each lane
//!   leaves once it has had the lock. The [`ATOMIC_LOCKS`] lock words are
//!   dwords of the root UAV `u`[`VIEW_WRITES`] in [`MEMORY_SPACE`] from
//!   byte [`ATOMIC_LOCKS_AT`], past the written pages and the read marks,
//!   chosen by a hash of the coordinates (texels that share one only wait
//!   for each other), so such a shader sets [`HostShader::view_marks`] and
//!   the host makes that buffer [`VIEW_WRITES_BYTES`] long, zeroed. Locks
//!   are free again when the shader ends.
//! * **Read marks** (with [`Options::view_marks`]; added 2026-11-25):
//!   every image instruction that reads a texture shader view the program
//!   takes from the resource heap itself (not through a table, whose slots
//!   the host makes for the draw) also sets bit
//!   [`VIEW_READS`] + the view's heap index in the same bitmap, after the
//!   written pages, where still clear: the host makes views it left null
//!   until a shader reads them (textures the title placed itself).
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).

mod bitcode;
#[cfg(all(test, windows))]
mod bvh_tests;
mod container;
#[cfg(all(test, windows))]
mod diff_tests;
#[cfg(all(test, windows))]
mod flow_tests;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
mod lower;
mod rect;
pub use lower::kept_wide;
pub use rect::{rectangle_gs, right_angle};
#[cfg(all(test, windows))]
mod memory_tests;
mod sampler;
#[cfg(all(test, windows))]
mod semantics_tests;
mod shader;
#[cfg(all(test, windows))]
mod stride_tests;
#[cfg(all(test, windows))]
mod synthetic_tests;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod valu_tests;
#[cfg(all(test, windows))]
mod waterfall_tests;
mod wave;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use sampler::StaticSampler;
pub use shader::{
    HEAP_SAMPLERS, HEAP_VIEWS, MEMORY_SPACE, NULL_BUFFER_VIEW, ROOT_CBV_SPACE, SAMPLER_SPACE,
    ATOMIC_LOCKS, ATOMIC_LOCKS_AT, USER_DATA_SPACE, VIEW_INTEGER, VIEW_MARK_NONE, VIEW_MARKS, VIEW_READS, VIEW_SIGNED,
    VIEW_STRIDE_BITS, VIEW_STRIDES, VIEW_WRITE_BITS, VIEW_WRITES, VIEW_WRITES_BYTES,
};

/// The loop limit without an override: 2^14.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// The iteration count after which a translated loop is left, per lane and
/// per entry into the loop: [`DEFAULT_LOOP_LIMIT`].
///
/// With the `diagnostics` feature, `VICTORY_SHADER_LOOP_LIMIT` (decimal or
/// `0x` hex, at least 1) sets it instead. A loop that hits the limit in
/// every lane of a full-screen pass is still slow (2^14 times a 4K screen
/// is about 10^11 iterations); a lower limit shows whether such loops are
/// what makes a frame slow, a higher one whether the limit cuts a genuine
/// loop short. The value is part of the translation, so the shader cache
/// must key on it (`victory-gpu`'s does).
pub fn loop_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        if cfg!(feature = "diagnostics") {
            parse_loop_limit(std::env::var("VICTORY_SHADER_LOOP_LIMIT").ok().as_deref())
        } else {
            lower::LOOP_LIMIT
        }
    })
}

/// `VICTORY_SHADER_LOOP_LIMIT`'s value; the default when it is unset,
/// unreadable or 0. Pure, for testing.
fn parse_loop_limit(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => lower::LOOP_LIMIT,
    }
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the host binds as root descriptors: `t<index>`
    /// in [`ROOT_CBV_SPACE`] (the host's own tables, and title root SRVs in
    /// programs using title memory; others are views, module notes).
    pub root_srvs: Vec<u8>,
    /// Root UAVs (raw buffers) the host binds as root descriptors:
    /// `u<index>` in [`ROOT_CBV_SPACE`] (likewise).
    pub root_uavs: Vec<u8>,
    /// Whether the shader reads title memory at addresses it computes: the
    /// host binds the page table `t0` in [`MEMORY_SPACE`] (module notes,
    /// "Title memory").
    pub memory: bool,
    /// Whether the host binds the page bitmap `u0` in [`MEMORY_SPACE`]
    /// (pages written and pages missed): whenever [`HostShader::memory`].
    pub memory_writes: bool,
    /// Whether it reads buffer views' title addresses or texture views'
    /// sizes from their descriptors: the host binds the view addresses `t1`
    /// in [`MEMORY_SPACE`] (module notes, "View addresses").
    pub view_addresses: bool,
    /// Whether it takes descriptors from the resource heap at indices it
    /// computes (root type 5, not through a table): the host cannot tell
    /// which views it reads, so every view in the heap must be current.
    pub heap: bool,
    /// Whether some of those are unordered access views: it may write
    /// through any of the heap's.
    pub heap_writes: bool,
    /// Whether it marks the pages it writes through buffer views: the host
    /// binds the view-mark table `t`[`VIEW_MARKS`] and the bitmap
    /// `u`[`VIEW_WRITES`] in [`MEMORY_SPACE`] (module notes, "Write
    /// marks").
    pub view_marks: bool,
}

impl HostShader {
    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }
}

/// [`UserData::dword`] of a root descriptor's entry that takes the heap
/// index of the host's raw SRV of it (module notes, "Root SRVs and root
/// UAVs"); dwords 0 and 1 are its address (programs using title memory).
pub const ROOT_SRV_VIEW: u32 = 2;
/// As [`ROOT_SRV_VIEW`], for the raw UAV of a root UAV.
pub const ROOT_UAV_VIEW: u32 = 3;
/// [`UserData::dword`] of a root descriptor's entry that takes its
/// address's byte offset in the view [`ROOT_SRV_VIEW`] or [`ROOT_UAV_VIEW`]
/// names.
pub const ROOT_VIEW_OFFSET: u32 = 4;

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants`, `Table` or `Descriptor` (see the module notes for the
    /// value).
    pub root: RootType,
    /// Dword within the parameter; for a root descriptor
    /// [`ROOT_SRV_VIEW`], [`ROOT_UAV_VIEW`] or [`ROOT_VIEW_OFFSET`] name
    /// its view's heap index or its offset in that view instead.
    pub dword: u32,
}

/// How a translation is made.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Options {
    /// Writes through buffer views mark the pages they write (module notes,
    /// "Write marks"); the host must then bind the view-mark table and the
    /// bitmap ([`HostShader::view_marks`]).
    pub view_marks: bool,
    /// 64-bit image atomics run as 32-bit locks around plain loads and
    /// stores (module notes, "Atomic locks"), for hosts that claim the
    /// 64-bit atomics and leave the texels untouched (D3DMetal).
    pub atomic_locks: bool,
    /// Compute-stage biased samples run as gradient samples with the quad
    /// gradients scaled by `2^bias`, for hosts that ignore the bias of a
    /// compute shader's sample (D3DMetal, #30).
    pub bias_as_gradients: bool,
    /// Trial linked layout for the 16-row mesh/pixel interface.
    pub compact_mesh_interface: bool,
    /// Mesh: each export's output stores come after all the values they
    /// store, not each after its own value. Metal shader converter builds a
    /// row's vector where the row's first channel is stored, from values
    /// made after it when they are widened halves or bitcasts; Apple's GPU
    /// compiler then crashes (register allocation) on the mesh pipelines
    /// whose pixel shader reads that row (D3DMetal: "XPC_ERROR_CONNECTION_
    /// INTERRUPTED", the pipeline a no-op).
    pub mesh_stores_last: bool,
    /// Mesh: when Metal would count more output than it allows
    /// (`container::metal_mesh_bytes` over `METAL_MESH_BYTES`), the vertex
    /// outputs are ordered for the least alignment padding in the struct
    /// D3DMetal makes of them (`container::metal_packed_order`). D3DMetal
    /// fails such pipelines ("Total mesh size (N) exceeds the maximum mesh
    /// size allowed (32768)"); the order of the elements is not what links
    /// the stages, so the pixel shaders stay as they are. Meshes within
    /// the limit are made as without it.
    pub mesh_pack_outputs: bool,
    /// The host runs wave64 programs in 32-lane waves whatever `WaveSize`
    /// says (D3DMetal): a compute group that is one console wave of more
    /// than 32 lanes gets group barriers where its lanes read LDS other
    /// lanes may have written, which the console's lockstep made safe.
    pub narrow_waves: bool,
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container, without write marks.
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    translate_with(container, Options::default())
}

/// [`translate`] made as `options` say.
pub fn translate_with(container: &[u8], options: Options) -> Result<HostShader, ShaderError> {
    translate_keeping(container, options, None)
}

/// [`translate_with`]; a mesh program with `kept` leaves out the vertex
/// outputs it does not name (system values stay): a pipeline's pixel shader
/// reads no others ([`pixel_reads`]). For the meshes Metal counts too
/// much output of even packed ([`mesh_over_metal_limit`]).
pub fn translate_keeping(
    container: &[u8],
    options: Options,
    kept: Option<&Kept>,
) -> Result<HostShader, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .or(c.part(FourCC::ACCC))
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
            )
        })
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a container without a vertex, pixel, compute or mesh program",
        })?;
    // Structurisation first: tracing per-lane descriptors needs it (its
    // failure is reported after the earlier stages').
    let code = native.patched_code();
    let (p, st) = crate::range::build(&code, native.wave64, native.vgprs)?;
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, &native)?;
    let st = st?;
    lower::take_wide();
    let made = emit_keeping(&c, &native, &p, &st, &io, &res, options, kept);
    let Some(why) = lower::take_wide() else {
        return made;
    };
    // A wave64 program whose wave operations sit where the halves of a
    // wave cannot meet: again without the skips no lane sees, so that every
    // lane runs that code under EXEC. Kept only if it then runs in halves.
    if let Some(again) = without_idle_skips(&c, &native, &code, &p, &st, options, kept) {
        return Ok(again);
    }
    lower::count_wide(why);
    made
}

/// The translation of `p` without its [`lower::idle_skips`], if that one
/// runs in 32-lane host waves ([`Options::narrow_waves`]).
fn without_idle_skips(
    c: &Container,
    native: &NativeProgram,
    code: &[u8],
    p: &ir::Program,
    st: &structure::Structured,
    options: Options,
    kept: Option<&Kept>,
) -> Option<HostShader> {
    const S_NOP: u32 = 0xBF80_0000;
    // First with the skips around loops left in, then without those too.
    let mut tried = Vec::new();
    for loops in [false, true] {
        let skips = lower::idle_skips(p, st, loops);
        if skips.is_empty() || skips == tried {
            continue;
        }
        let mut code = code.to_vec();
        for &at in &skips {
            code[at..at + 4].copy_from_slice(&S_NOP.to_le_bytes());
        }
        tried = skips;
        let Ok((p, st)) = crate::range::build(&code, native.wave64, native.vgprs) else {
            continue;
        };
        let Ok(res) = resource::track(&p, &native.layout, native.stage, st.as_ref().ok()) else {
            continue;
        };
        let (Ok(io), Ok(st)) = (io::raise(&p, native), st) else {
            continue;
        };
        let made = emit_keeping(c, native, &p, &st, &io, &res, options, kept);
        if lower::take_wide().is_none()
            && let Ok(made) = made
        {
            return Some(made);
        }
    }
    None
}

/// The vertex outputs a mesh translation keeps besides its system values:
/// the semantics (name in upper case, index) a pixel shader reads, sorted.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Kept(Vec<(String, u32)>);

impl Kept {
    pub fn new(semantics: impl IntoIterator<Item = (String, u32)>) -> Kept {
        let mut v: Vec<(String, u32)> = semantics
            .into_iter()
            .map(|(n, i)| (n.to_ascii_uppercase(), i))
            .collect();
        v.sort_unstable();
        v.dedup();
        Kept(v)
    }

    fn keeps(&self, e: &container::SigElement) -> bool {
        e.kind != 0
            || self
                .0
                .binary_search(&(e.name.to_ascii_uppercase(), e.index))
                .is_ok()
    }

    /// One line per semantic, for cache keys.
    pub fn describe(&self) -> String {
        self.0.iter().map(|(n, i)| format!("{n}{i}\n")).collect()
    }
}

/// The input semantics a translated pixel shader reads (its `ISG1`).
pub fn pixel_reads(dxil: &[u8]) -> Result<Kept, ShaderError> {
    let c = Container::parse(dxil)?;
    let part = c
        .part(FourCC::ISG1)
        .ok_or(ShaderError::Malformed("no input signature"))?
        .data;
    let read = |i: usize| part.get(crate::dxbc::u32_at(part, 4).ok()? as usize + 32 * i + 25);
    let elements = signature(part)?;
    let mut out = Vec::new();
    for (i, e) in elements.into_iter().enumerate() {
        if read(i).is_some_and(|&rw| rw != 0) {
            out.push((e.name, e.index));
        }
    }
    Ok(Kept::new(out))
}

/// A translated mesh shader's vertex and primitive outputs and its maximum
/// vertex and primitive counts.
type MeshOutputs = (Vec<container::SigElement>, Vec<container::SigElement>, u32, u32);

fn mesh_outputs(dxil: &[u8]) -> Option<MeshOutputs> {
    let c = Container::parse(dxil).ok()?;
    let sig = |f| c.part(f).map_or(Ok(Vec::new()), |p| signature(p.data)).ok();
    let (outputs, prims) = (sig(FourCC::OSG1)?, sig(FourCC::PSG1)?);
    let psv = c.part(FourCC::PSV0)?.data;
    let u16_at = |o: usize| {
        psv.get(o..o + 2)
            .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]])))
    };
    Some((outputs, prims, u16_at(4 + 12)?, u16_at(4 + 14)?))
}

/// Whether Metal counts more output than it allows of a translated mesh
/// shader even with its outputs packed (`container::metal_packed_order`):
/// D3DMetal fails the pipelines of such a mesh shader with any pixel shader
/// that reads them all.
pub fn mesh_over_metal_limit(dxil: &[u8]) -> bool {
    let Some((outputs, prims, vertices, primitives)) = mesh_outputs(dxil) else {
        return false;
    };
    let packed: Vec<container::SigElement> = container::metal_packed_order(&outputs)
        .into_iter()
        .map(|i| outputs[i].clone())
        .collect();
    container::metal_mesh_bytes(&packed, &prims, vertices, primitives)
        > container::METAL_MESH_BYTES
}

/// Whether a translated mesh shader has vertex outputs (other than system
/// values) that `kept` does not name.
pub fn mesh_outputs_unread(dxil: &[u8], kept: &Kept) -> bool {
    mesh_outputs(dxil).is_some_and(|(outputs, ..)| outputs.iter().any(|e| !kept.keeps(e)))
}

/// Emits a program whose earlier stages already ran, without write marks.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    emit_with(c, native, p, st, io, res, Options::default())
}

/// [`emit`] made as `options` say.
pub fn emit_with(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    options: Options,
) -> Result<HostShader, ShaderError> {
    emit_keeping(c, native, p, st, io, res, options, None)
}

/// [`emit_with`], a mesh program keeping only the vertex outputs `kept`
/// names ([`translate_keeping`]).
#[allow(clippy::too_many_arguments)]
fn emit_keeping(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    options: Options,
    kept: Option<&Kept>,
) -> Result<HostShader, ShaderError> {
    let sig = |fourcc| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1)?;
    let mut outputs = sig(FourCC::OSG1)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say.
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (13, _) => 1,
            // Until a read says which mode (`lower`).
            (28, _) => 2,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    // With a conservative depth output (SV_DepthLessEqual, GreaterEqual) DXIL
    // wants the position centroid (or per sample), as DXC declares it then.
    if pixel && outputs.iter().any(|e| matches!(e.kind, 18 | 19)) {
        for e in inputs.iter_mut().filter(|e| e.kind == 3) {
            e.interp = 5;
        }
    }
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // Rows of the vertex outputs left out: their exports store nothing.
    let mut dropped_rows = Vec::new();
    if let (Some(kept), crate::native::StageIo::Ms(_)) = (kept, &native.io) {
        outputs.retain(|e| {
            let keep = kept.keeps(e);
            if !keep {
                dropped_rows.extend(e.row);
            }
            keep
        });
    }
    if let (true, crate::native::StageIo::Ms(ms)) = (options.mesh_pack_outputs, &native.io) {
        let bytes =
            container::metal_mesh_bytes(&outputs, &prims, ms.max_vertices, ms.max_primitives);
        if bytes > container::METAL_MESH_BYTES {
            let order = container::metal_packed_order(&outputs);
            outputs = order.into_iter().map(|i| outputs[i].clone()).collect();
        }
    }
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (the title's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    // The structure that kills discarded lanes, made when first needed.
    let killing = std::cell::OnceCell::<(structure::Structured, resource::Resources)>::new();
    let parts = |st, res| lower::Parts {
        native,
        psv: c.part(FourCC::PSV0).map(|p| p.data),
        p,
        st,
        io,
        res,
        inputs: inputs.clone(),
        outputs: outputs.clone(),
        prims: prims.clone(),
        dropped_rows: dropped_rows.clone(),
        mesh_stride,
        view_marks: options.view_marks,
        atomic_locks: options.atomic_locks,
        bias_as_gradients: options.bias_as_gradients,
        narrow_waves: options.narrow_waves,
    };
    // Most programs only interpolate with the barycentric VGPRs, or join
    // them in a former use no lane reads; one that cannot do without them
    // as values is made again with `SV_Barycentrics`.
    let mut made = None;
    for barycentrics in [false, true] {
        let l = match lower::lower_with_layout(
            &parts(st, res),
            options.compact_mesh_interface,
            options.mesh_stores_last,
            barycentrics,
        ) {
            // Discarded lanes kept on as helpers can read what no surviving
            // lane does (a register only killed lanes skipped writing); such
            // a program kills them as the console does (`structure::killing`).
            Err(_) if !st.discards.is_empty() => {
                if killing.get().is_none() {
                    let st = structure::killing(p)?;
                    let res = resource::track(p, &native.layout, native.stage, Some(&st))?;
                    let _ = killing.set((st, res));
                }
                let (st, res) = killing.get().expect("made above");
                lower::lower_with_layout(
                    &parts(st, res),
                    options.compact_mesh_interface,
                    options.mesh_stores_last,
                    barycentrics,
                )
            }
            l => l,
        };
        match l {
            Err(ShaderError::Unsupported { what, .. })
                if what == lower::BARYCENTRICS_READ && !barycentrics => {}
            l => {
                made = Some(l?);
                break;
            }
        }
    }
    let l = made.expect("the second pass returns");
    Ok(HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.filter |= 0x80;
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    })
}

/// Whether the container's `PSV0` part declares the resource at
/// `register` structured: the native binding map gives a register but no
/// space, so every declaration of that register class covering it must be
/// a structured buffer (SRV kind 5, UAV kind 8).
fn declares_structured_buffer(part: &[u8], register: u32, uav: bool) -> bool {
    use crate::dxbc::u32_at;
    let table = || {
        let info = u32_at(part, 0).ok()? as usize;
        let header = part.get(4..)?.get(info..)?;
        let count = u32_at(header, 0).ok()? as usize;
        let stride = u32_at(header, 4).ok()? as usize;
        if stride < 16 {
            return None;
        }
        Some(
            header
                .get(8..)?
                .get(..count.checked_mul(stride)?)?
                .chunks_exact(stride),
        )
    };
    let Some(entries) = table() else { return false };
    let mut found = false;
    for entry in entries {
        let (Ok(kind), Ok(lo), Ok(hi)) = (u32_at(entry, 0), u32_at(entry, 8), u32_at(entry, 12))
        else {
            return false;
        };
        let class = if uav { 6..=9 } else { 3..=5 };
        if class.contains(&kind) && (lo..=hi).contains(&register) {
            if kind != if uav { 8 } else { 5 } {
                return false;
            }
            found = true;
        }
    }
    found
}

/// The rows of an `ISG1`/`OSG1` part.
fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            23 => 28, // barycentrics
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod barycentric_tests {
    use super::container::SigElement;
    use super::*;
    use crate::native::{PsInput, StageIo, Usage};

    /// `SPI_PS_INPUT_ADDR` bits: PERSP_CENTER, LINEAR_CENTER.
    const PERSP_CENTER: u32 = 1 << 1;
    const LINEAR_CENTER: u32 = 1 << 5;

    /// `exp mrt0, v<r>, v<r>, v<r>, v<r> done vm`.
    fn exp_mrt0(r: u32) -> [u32; 2] {
        [0xF800_180F, r * 0x0101_0101]
    }

    /// VOP2 `v_add_f32 v<dst>, v<a>, v<b>`.
    fn v_add_f32(dst: u32, a: u32, b: u32) -> u32 {
        3 << 25 | dst << 17 | b << 9 | (256 + a)
    }

    fn element(name: &str, kind: u8, row: u32) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }
    }

    /// Translates a pixel program with the input VGPRs `input_addr`
    /// enables and one float4 TEXCOORD0 attribute (attr 0, row 0).
    fn pixel(words: &[u32], input_addr: u32) -> Result<HostShader, ShaderError> {
        pixel_with_inputs(words, input_addr, &[element("TEXCOORD", 0, 0)])
    }

    /// [`pixel`] with the input signature `inputs` (attr 0, row 0).
    fn pixel_with_inputs(
        words: &[u32],
        input_addr: u32,
        inputs: &[SigElement],
    ) -> Result<HostShader, ShaderError> {
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 0b1111,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mut target = element("SV_Target", 16, 0);
        target.usage = 15;
        let isg1 = container::signature(inputs, true);
        let osg1 = container::signature(&[target], false);
        let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
        let st = structure::structure(&p)?;
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native)?;
        emit(&c, &native, &p, &st, &io, &res)
    }

    /// The translation's `SFI0` features and `ISG1` rows.
    fn parts(shader: &HostShader) -> (u64, Vec<SigElement>) {
        let c = Container::parse(&shader.dxil).unwrap();
        let sfi0 = c.part(FourCC::SFI0).unwrap().data.try_into().unwrap();
        (
            u64::from_le_bytes(sfi0),
            signature(c.part(FourCC::ISG1).unwrap().data).unwrap(),
        )
    }

    fn validate(shader: &HostShader) {
        #[cfg(windows)]
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil.clone()).unwrap();
        }
        let _ = shader;
    }

    /// Custom interpolation reads I and J (v0, v1 with PERSP_CENTER) as
    /// plain floats: they are channels y and z of `SV_Barycentrics`, the
    /// weights of vertices 1 and 2.
    #[test]
    fn barycentrics_read_as_values_are_sv_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [v_add_f32(2, 0, 1), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        let bary: Vec<_> = inputs.iter().filter(|e| e.kind == 28).collect();
        assert_eq!(bary.len(), 1, "{inputs:?}");
        assert_eq!((bary[0].index, bary[0].col, bary[0].cols), (0, 0, 3));
        // Not packed: the validator wants packing location -1.
        assert_eq!(bary[0].row, None);
        // Only y and z (I and J) are read.
        let isg1 = Container::parse(&shader.dxil).unwrap();
        let isg1 = isg1.part(FourCC::ISG1).unwrap().data;
        let rw: Vec<u8> = (0..2).map(|k| isg1[8 + 32 * k + 25]).collect();
        assert!(rw.contains(&0b110), "{rw:?}");
        validate(&shader);
    }

    /// Two kinds of barycentrics read directly: one element per mode.
    #[test]
    fn two_barycentric_modes_get_two_elements() {
        // v0-v1 perspective i, j; v2-v3 linear i, j.
        let [e0, e1] = exp_mrt0(4);
        let words = [v_add_f32(4, 0, 3), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER | LINEAR_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        let mut indices: Vec<u32> = inputs
            .iter()
            .filter(|e| e.kind == 28)
            .map(|e| e.index)
            .collect();
        indices.sort();
        assert_eq!(indices, [0, 1], "{inputs:?}");
        validate(&shader);
    }

    /// I and J only fed to `v_interp_*` stay host interpolation: no
    /// barycentrics are declared.
    #[test]
    fn interpolation_alone_declares_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_eq!(features & 0x2_0000, 0);
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        validate(&shader);
    }

    /// An unread float input packed beside an input read at a vertex
    /// (`v_interp_mov_f32`, constant) takes that constant mode: D3D wants
    /// one mode per row, including the components the shader never reads
    /// (DexNom/VIctory 5f4c3df).
    #[test]
    fn an_unused_packed_input_matches_its_provoking_vertex_sibling() {
        let [e0, e1] = exp_mrt0(3);
        // v_interp_mov_f32 v3, p0, attr0.z followed by exp mrt0.
        let words = [0xC80E_0202, e0, e1, 0xBF81_0000];
        let xy = SigElement { cols: 2, ..element("TEXCOORD", 0, 0) };
        let zw = SigElement { col: 2, cols: 2, ..element("NORMAL", 0, 0) };
        let shader = pixel_with_inputs(&words, PERSP_CENTER, &[xy, zw]).unwrap();
        let c = Container::parse(&shader.dxil).unwrap();
        let psv = c.part(FourCC::PSV0).unwrap().data;
        // PSV0 records: one row from row 0, (columns, start), kind 0, f32,
        // constant.
        for columns in [0x42, 0x62] {
            let record = [1, 0, columns, 0, 3, 1];
            assert!(psv.windows(record.len()).any(|w| w == record));
        }
        let inputs = c.part(FourCC::ISG1).unwrap().data;
        assert_eq!((inputs[8 + 25], inputs[8 + 32 + 25]), (0, 4));
        validate(&shader);
    }

    /// Both: the attribute is interpolated and J is used as a value.
    #[test]
    fn interpolation_and_a_barycentric_value_together() {
        let [e0, e1] = exp_mrt0(3);
        let words = [
            0xC808_0000,        // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,        // v_interp_p2_f32 v2, v1, attr0.x
            v_add_f32(3, 2, 1), // v3 = attribute + j
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        validate(&shader);
    }
}

#[cfg(test)]
mod tests {
    use super::bitcode::Attr;
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
    }
    use super::container::SigElement;
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    #[test]
    fn indexed_move_initializes_an_input_before_its_first_use() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Case {
            Complete,
            NonzeroBase,
            WrongStart,
            SkippedSlot,
            Decrement,
            Wrap,
            UnknownStart,
            ShortLoop,
            ClobberedScc,
            EarlyExit,
            DivergentExit,
            MaskedWrite,
            SourceIsOld,
            ReadBeforeWrite,
            ReadOnBackedge,
        }
        use Case::*;

        // A two-block do-while loop writes v[M0] from v2, then exports v1.
        // On the first two passes M0 selects v0 and v1 respectively. Both
        // registers initially contain raw perspective-center barycentrics.
        let words = |case| {
            let init = match case {
                WrongStart => 0xbe80_0382,   // s_mov_b32 s0, 2
                Wrap => 0xbe80_03c1,         // s_mov_b32 s0, -1
                UnknownStart => 0x8000_8080, // s_add_u32 s0, 0, 0
                _ => 0xbe80_0380,            // s_mov_b32 s0, 0
            };
            let mut w = vec![init, 0x7e04_0281, 0xbf82_0000]; // v_mov_b32 v2, 1; s_branch header
            let latch = w.len();
            if case == EarlyExit {
                w.extend([0xbf06_8000, 0xbf85_0000]); // s_cmp_eq_u32 s0, 0; s_cbranch_scc1 exit
            }
            w.push(0xbe80_0302); // s_mov_b32 s0, s2
            let header = w.len();
            w[2] |= (header - 3) as u32;
            w.push(0xbefc_0300); // s_mov_b32 m0, s0
            w.push(match case {
                SkippedSlot => 0x8102_8200, // s_add_i32 s2, s0, 2
                Decrement => 0x8102_c100,   // s_add_i32 s2, s0, -1
                _ => 0x8102_8100,           // s_add_i32 s2, s0, 1
            });
            if case == ReadBeforeWrite {
                w.push(0x7e06_0301);
            } // v_mov_b32 v3, v1
            if case == MaskedWrite {
                w.extend([0x7d84_0480, 0xbe84_3c6a]); // v_cmp_eq_u32 vcc, 0, v2; s_and_saveexec_b32 s4, vcc_lo
            }
            w.push(match case {
                SourceIsOld => 0x7e00_8501, // v_movreld_b32 v0, v1
                NonzeroBase => 0x7e02_8502, // v_movreld_b32 v1, v2
                _ => 0x7e00_8502,           // v_movreld_b32 v0, v2
            });
            if case == MaskedWrite {
                w.push(0x887e_047e);
            } // s_or_b32 exec_lo, exec_lo, s4
            if case == ReadOnBackedge {
                w.push(0x7e06_0301);
            } // v_mov_b32 v3, v1
            let branch = if case == DivergentExit {
                w.push(0x7d84_0480); // v_cmp_eq_u32 vcc, 0, v2
                0xbf87_0000 // s_cbranch_vccnz latch
            } else {
                w.push(if matches!(case, ShortLoop | NonzeroBase) {
                    0xbf06_8102
                } else {
                    0xbf06_8402
                }); // s_cmp_eq_u32 s2, 1/4
                if case == ClobberedScc {
                    w.push(0xbf06_8000); // s_cmp_eq_u32 s0, 0 exits on the first pass
                }
                0xbf84_0000 // s_cbranch_scc0 latch
            };
            let rel = latch as i32 - (w.len() as i32 + 1);
            w.push(branch | u32::from(rel as u16));
            let exit = w.len();
            if case == EarlyExit {
                w[latch + 1] |= (exit - latch - 2) as u32;
            }
            let output = if matches!(case, ReadBeforeWrite | ReadOnBackedge) {
                3
            } else {
                1
            };
            w.extend([0xf800_1801, output, 0xbf81_0000]); // exp mrt0.x done vm; s_endpgm
            w
        };
        let target = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let sig = super::container::signature(&[target], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &sig)]);
        let c = Container::parse(&bytes).unwrap();
        for case in [
            Complete,
            NonzeroBase,
            WrongStart,
            SkippedSlot,
            Decrement,
            Wrap,
            UnknownStart,
            ShortLoop,
            ClobberedScc,
            EarlyExit,
            DivergentExit,
            MaskedWrite,
            SourceIsOld,
            ReadBeforeWrite,
            ReadOnBackedge,
        ] {
            let native = NativeProgram {
                stage: HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: StageIo::Ps {
                    inputs: vec![],
                    input_addr: 2,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words(case).iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = structure::structure(&p).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let shader = emit(&c, &native, &p, &st, &io, &res);
            if matches!(case, Complete | NonzeroBase) {
                let shader = shader.unwrap();
                assert!(
                    Container::parse(&shader.dxil)
                        .unwrap()
                        .part(FourCC::DXIL)
                        .is_some()
                );
                #[cfg(windows)]
                if let Ok(mut validator) =
                    validator::Validator::load(validator::Validator::default_path().as_deref())
                {
                    let mut dxil = shader.dxil;
                    validator.validate(&mut dxil).unwrap();
                    eprintln!("synthetic {case:?} SDK valid bytes={}", dxil.len());
                }
            } else {
                // The loop does not overwrite the input first, so the
                // program reads the barycentrics it was given: as values
                // through SV_Barycentrics (this fork; upstream refuses
                // "barycentrics read directly").
                eprintln!("synthetic {case:?} {:?}", shader.as_ref().err());
                assert!(shader.is_ok(), "{case:?}");
            }
        }
    }

    /// A pixel shader writing SV_StencilRef (channel y of `exp mrtz`).
    /// DXC sets ShaderFlags.StencilRef (bit 11) for a pixel shader whose
    /// output signature has SV_StencilRef, and the feature bit
    /// D3D_SHADER_FEATURE_STENCIL_REF (0x200) in SFI0; the validator checks
    /// both against the module.
    #[test]
    fn a_stencil_export_declares_the_stencil_ref_feature() {
        let words = [
            0x7E00_02F2u32, // v_mov_b32 v0, 1.0
            0x7E02_0285,    // v_mov_b32 v1, 5
            0xF800_1883,
            0x0000_0100, // exp mrtz v0, v1, off, off done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, comp| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let osg1 =
            container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
        // dxil.dll, where there is one, checks the flags against the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        let host = Container::parse(&out.dxil).unwrap();
        let sfi0 = host.part(FourCC::SFI0).unwrap().data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_ne!(features & 0x200, 0, "SFI0 {features:#x}: no StencilRef");
    }

    #[test]
    fn immediate_sampler_slots_distinguish_comparison_uses() {
        let words = [
            0xBE81_0380u32, // s_mov_b32 s1, 0
            0xF40C_0000,
            0xFA00_0000, // s_load_dwordx8 s[0:7], s[0:1], null
            0xBE88_0380, // s_mov_b32 s8, 0
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
            0x7E00_02F0, // v_mov_b32 v0, 0.5 (comparison reference)
            0x7E02_02F0, // v_mov_b32 v1, 0.5 (u)
            0x7E04_02F0, // v_mov_b32 v2, 0.5 (v)
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0xF080_0108,
            0x0040_0301, // image_sample v3, v[1:2], s[0:7], s[8:11]
            0xF0A0_0108,
            0x0040_0400, // image_sample_c v4, v[0:2], s[0:7], s[8:11]
            0xF0BC_0108,
            0x0040_0500, // image_sample_c_lz v5, v[0:2], s[0:7], s[8:11]
            0xF080_0108,
            0x0040_0601, // image_sample v6, v[1:2], s[0:7], s[8:11]
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xF800_180F,
            0x0605_0403, // exp mrt0, v3, v4, v5, v6 done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 7,
            layout: vec![crate::native::RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![crate::native::UsedResource {
                kind: crate::native::ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let output = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        };
        let osg1 = container::signature(&[output], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();

        let sampler = StaticSampler::from_words([0; 4]);
        let comparison = StaticSampler {
            filter: sampler.filter | 0x80,
            ..sampler
        };
        assert_eq!(out.samplers, [[0; 4]; 2]);
        assert_eq!(out.static_samplers, [sampler, comparison]);
        #[cfg(windows)]
        if let Some(path) = super::validator::Validator::default_path() {
            let mut v = super::validator::Validator::load(Some(&path)).unwrap();
            v.validate(&mut out.dxil.clone()).unwrap();
        }
    }

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask is
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A pixel shader writing a constant colour.
    #[cfg(windows)]
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let host = match super::host::Host::new(false) {
            Ok(host) => host,
            Err(e) => {
                eprintln!("no SM 6.6 device ({e}); skipped");
                return;
            }
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            memory: false,
            memory_writes: false,
            view_addresses: false,
            heap: false,
            heap_writes: false,
            view_marks: false,
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                memory: false,
                memory_writes: false,
                view_addresses: false,
                heap: false,
                heap_writes: false,
                view_marks: false,
            })
        };
        let ps = file("VICTORY_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("VICTORY_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        // D3DMetal under Wine does not check the hash, so it accepts both.
        // SAFETY: ntdll is always loaded; both names end with a NUL.
        let wine = unsafe {
            use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
            GetProcAddress(
                GetModuleHandleA(c"ntdll.dll".as_ptr().cast()),
                c"wine_get_version".as_ptr().cast(),
            )
        }
        .is_some();
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        let digests: &[&[u8]] = if wine { &[] } else { &[&[0u8; 16], &wrong] };
        for &digest in digests {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// Triangle (0, 1, 2), each corner's 9-bit index in its 10-bit field.
    const TRIANGLE: u32 = 1 << 10 | 2 << 20;
    /// The edge flags of the three corners (bits 9, 19 and 29).
    const EDGE_FLAGS: u32 = 1 << 9 | 1 << 19 | 1 << 29;
    /// The culling bit.
    const CULL: u32 = 1 << 31;

    /// A one-wave (32-lane) triangle mesh program whose lanes all export
    /// primitive `prim` and, as vertex `lane`, the position
    /// (lane & 1 ? 3 : -1, lane & 2 ? 3 : -1, 0.5, 1): triangle (0, 1, 2)
    /// covers the whole target.
    fn triangle_mesh(prim: u32) -> Result<HostShader, ShaderError> {
        let words = [
            0xD765_0001u32, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0x0001_00C1,
            0x3604_0281, // v_and_b32 v2, 1, v1
            0x3606_0282, // v_and_b32 v3, 2, v1
            0x7E04_0D02, // v_cvt_f32_u32 v2, v2
            0x7E06_0D03, // v_cvt_f32_u32 v3, v3
            0x1008_04F6, // v_mul_f32 v4, 4.0, v2
            0x0608_08F3, // v_add_f32 v4, -1.0, v4
            0x100A_06F4, // v_mul_f32 v5, 2.0, v3
            0x060A_0AF3, // v_add_f32 v5, -1.0, v5
            0x7E0C_02F0, // v_mov_b32 v6, 0.5
            0x7E0E_02F2, // v_mov_b32 v7, 1.0
            0x7E12_02FF, // v_mov_b32 v9, prim
            prim,
            0xF800_0941, // exp prim v9, off, off, off done
            0x0000_0009,
            0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
            0x0706_0504,
            0xBF81_0000, // s_endpgm
        ];
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64: false,
            vgprs: 10,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(crate::native::MeshIo {
                outputs: vec![],
                clip_cntl: 0,
                threads: 32,
                max_vertices: 32,
                max_primitives: 32,
                topology: 2,
                lds_bytes: 0,
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let position = SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xF,
            half: false,
        };
        let osg1 = container::signature(&[position], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes)?;
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
        let st = structure::structure(&p)?;
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native)?;
        emit(&c, &native, &p, &st, &io, &res)
    }

    /// A primitive export's corners are 9-bit indices: the edge flag above
    /// each (bits 9, 19, 29) is no part of it. Masking ten bits made a
    /// flagged corner index 512 or more, past any mesh program's vertices
    /// (the draw test below shows the triangle). Flagged, unflagged and
    /// culled exports translate, and dxil.dll, where there is one, takes
    /// each module.
    #[test]
    fn primitive_export_indices_drop_the_edge_flags() {
        #[cfg(windows)]
        let mut validator = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        )
        .ok();
        for prim in [
            TRIANGLE,
            TRIANGLE | EDGE_FLAGS,
            TRIANGLE | EDGE_FLAGS | CULL,
        ] {
            let ms = triangle_mesh(prim).unwrap_or_else(|e| panic!("{prim:#x}: {e}"));
            #[cfg(windows)]
            if let Some(v) = validator.as_mut() {
                v.validate(&mut ms.dxil.clone())
                    .unwrap_or_else(|e| panic!("{prim:#x}: {e}"));
            }
        }
    }

    /// A triangle mesh program on the host GPU and on WARP (where they have
    /// mesh shaders): triangle (0, 1, 2) covers the target with its corners'
    /// edge flags set as without them; with the culling bit, nothing is lit.
    #[test]
    #[cfg(windows)]
    fn edge_flagged_triangles_draw_as_unflagged_ones() {
        use super::host::{Draw, Host, Root};
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_FEATURE_D3D12_OPTIONS7, D3D12_FEATURE_DATA_D3D12_OPTIONS7,
            D3D12_MESH_SHADER_TIER_NOT_SUPPORTED, D3D12_SHADER_VISIBILITY_MESH,
        };
        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ps = constant_color();
        let mut ran = 0;
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                eprintln!("no SM 6.6 {label} device; skipped");
                continue;
            };
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            // SAFETY: the query writes the structure of the given size.
            let queried = unsafe {
                host.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS7,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
            };
            if queried.is_err() || options.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED {
                eprintln!("no mesh shaders on the {label} device; skipped");
                continue;
            }
            for (prim, culled) in [
                (TRIANGLE, false),
                (TRIANGLE | EDGE_FLAGS, false),
                (TRIANGLE | EDGE_FLAGS | CULL, true),
            ] {
                let ms = triangle_mesh(prim).unwrap();
                let root = host
                    .root_signature_for(&[(&ms, D3D12_SHADER_VISIBILITY_MESH)])
                    .unwrap();
                let pso = host
                    .mesh_pipeline_with(&root, &ms.dxil, &ps)
                    .unwrap_or_else(|e| panic!("{label}: mesh pipeline: {e}"));
                let texels = host
                    .draw(
                        &root,
                        &pso,
                        &Draw {
                            roots: vec![(0, Root::Constants(vec![0]))],
                            textures: &[],
                            samplers: &[],
                            vertices: &[],
                            stride: 0,
                            size: (16, 16),
                            mesh_groups: 1,
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label}: draw: {e}"));
                let lit = texels.iter().filter(|t| t[3] != 0.0).count();
                if culled {
                    assert_eq!(lit, 0, "{label}, {prim:#x}: a culled triangle lit texels");
                    continue;
                }
                assert!(
                    texels.iter().all(|t| *t == [1.0, 0.5, 0.25, 1.0]),
                    "{label}, {prim:#x}: {lit} of 256 texels lit"
                );
            }
            eprintln!("{label}: edge-flagged triangles drew");
            ran += 1;
        }
        if ran == 0 {
            eprintln!("no device with mesh shaders; skipped");
        }
    }

    /// A pixel shader interpolating at sample positions (PERSP_SAMPLE
    /// barycentrics, `SPI_PS_INPUT_ADDR` bit 0) runs at sample frequency:
    /// DXC's PSV0 says so in PSInfo.SampleFrequency (runtime info byte 1)
    /// whenever an input is interpolated per sample.
    #[test]
    fn sample_interpolated_inputs_set_psv_sample_frequency() {
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let psv_of = |input_addr: u32| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half: false,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let out = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
            // dxil.dll, where there is one, checks PSV0 against the module.
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = super::Container::parse(&out.dxil).unwrap();
            host.part(super::FourCC::PSV0).unwrap().data.to_vec()
        };
        // Runtime info starts after its size; PSInfo is {DepthOutput, SampleFrequency}.
        let center = psv_of(1 << 1);
        assert_eq!(center[4 + 1], 0, "center interpolation is per pixel");
        let sample = psv_of(1 << 0);
        assert_eq!(sample[4 + 1], 1, "PSInfo.SampleFrequency");
    }

    /// Emits a pixel program of `words` with these inputs and signatures.
    fn emit_pixel(
        words: &[u32],
        inputs: Vec<crate::native::PsInput>,
        input_addr: u32,
        isg1: &[u8],
        osg1: &[u8],
    ) -> Result<HostShader, ShaderError> {
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs,
                input_addr,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let bytes = crate::dxbc::build(&[(FourCC::ISG1, isg1), (FourCC::OSG1, osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native)?;
        let out = emit(&c, &native, &p, &st, &io, &res)?;
        // dxil.dll, where there is one, checks the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        Ok(out)
    }

    /// A float4 element per row (16-bit ones declared so in the part).
    fn rows(elements: &[(&str, u8, bool)], input: bool) -> Vec<u8> {
        let els: Vec<SigElement> = elements
            .iter()
            .enumerate()
            .map(|(row, &(name, kind, _))| SigElement {
                name: name.into(),
                index: if kind == 3 { 0 } else { row as u32 },
                kind,
                comp: 9,
                interp: 0,
                row: Some(row as u32),
                col: 0,
                cols: 4,
                usage: 0,
                half: false,
            })
            .collect();
        let mut part = container::signature(&els, input);
        for (row, &(.., half)) in elements.iter().enumerate() {
            if half {
                part[8 + 32 * row + 16] = 6;
            }
        }
        part
    }

    /// Attribute 0 in row 0, 16-bit (one value per dword) or not.
    fn texcoord(fp16: bool) -> crate::native::PsInput {
        crate::native::PsInput {
            usage: crate::native::Usage::Generic,
            row: 0,
            mask: 1,
            attr: 0,
            flat: false,
            fp16,
            pack2: false,
            row2: 0,
        }
    }

    /// Rows of [`param_mesh`]'s output signature after the position: each
    /// a scalar (`TEXCOORD<2r>`) and a 3-vector (`TEXCOORD<2r+1>`), which
    /// D3DMetal lays out with padding (`container::metal_record`).
    const PARAM_ROWS: u32 = 14;

    /// The output (or input) elements of [`param_mesh`]: the position, then
    /// per row r of 1..=rows a scalar `TEXCOORD<2r>` in x and a 3-vector
    /// `TEXCOORD<2r+1>` in yzw.
    fn param_elements(rows: u32) -> Vec<SigElement> {
        let el = |name: &str, index, kind, row, col, cols| SigElement {
            name: name.into(),
            index,
            kind,
            comp: 9,
            interp: 0,
            row: Some(row),
            col,
            cols,
            usage: ((1u8 << cols) - 1) << col,
            half: false,
        };
        let mut els = vec![el("SV_Position", 0, 3, 0, 0, 4)];
        for r in 1..=rows {
            els.push(el("TEXCOORD", 2 * r, 0, r, 0, 1));
            els.push(el("TEXCOORD", 2 * r + 1, 0, r, 1, 3));
        }
        els
    }

    /// A one-wave triangle mesh program (128 vertices declared) whose lanes
    /// export, besides [`triangle_mesh`]'s triangle and positions, the
    /// constants (r, r + 0.25, r + 0.5, r + 0.75) to parameter row r of
    /// `rows` ([`param_elements`]).
    fn param_mesh(
        rows: u32,
        options: Options,
        kept: Option<&Kept>,
    ) -> Result<HostShader, ShaderError> {
        let mut words = vec![
            0xD765_0001u32, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0x0001_00C1,
            0x3604_0281, // v_and_b32 v2, 1, v1
            0x3606_0282, // v_and_b32 v3, 2, v1
            0x7E04_0D02, // v_cvt_f32_u32 v2, v2
            0x7E06_0D03, // v_cvt_f32_u32 v3, v3
            0x1008_04F6, // v_mul_f32 v4, 4.0, v2
            0x0608_08F3, // v_add_f32 v4, -1.0, v4
            0x100A_06F4, // v_mul_f32 v5, 2.0, v3
            0x060A_0AF3, // v_add_f32 v5, -1.0, v5
            0x7E0C_02F0, // v_mov_b32 v6, 0.5
            0x7E0E_02F2, // v_mov_b32 v7, 1.0
        ];
        for r in 1..=rows {
            for (k, add) in [0.0f32, 0.25, 0.5, 0.75].into_iter().enumerate() {
                // v_mov_b32 v(10 + k), r + add
                words.extend([0x7E14_02FF + ((k as u32) << 17), (r as f32 + add).to_bits()]);
            }
            // exp param(r - 1) v10, v11, v12, v13
            words.extend([0xF800_020F + ((r - 1) << 4), 0x0D0C_0B0A]);
        }
        words.extend([
            0x7E12_02FF, // v_mov_b32 v9, TRIANGLE
            TRIANGLE,
            0xF800_0941, // exp prim v9, off, off, off done
            0x0000_0009,
            0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
            0x0706_0504,
            0xBF81_0000, // s_endpgm
        ]);
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64: false,
            vgprs: 16,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(crate::native::MeshIo {
                outputs: (1..=rows as u8)
                    .map(|r| crate::native::VsOutput {
                        usage: crate::native::Usage::Generic,
                        row: r,
                        param: r - 1,
                        fp16: false,
                        pack2: false,
                    })
                    .collect(),
                clip_cntl: 0,
                threads: 32,
                max_vertices: 128,
                max_primitives: 128,
                topology: 2,
                lds_bytes: 0,
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let osg1 = container::signature(&param_elements(rows), false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes)?;
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
        let st = structure::structure(&p)?;
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native)?;
        emit_keeping(&c, &native, &p, &st, &io, &res, options, kept)
    }

    /// A pixel program that writes four of [`param_mesh`]'s outputs,
    /// interpolated, to target 0: TEXCOORD6 (row 3 x), TEXCOORD15's second
    /// channel (row 7 z), TEXCOORD25's third (row 12 w) and TEXCOORD28
    /// (row 14 x): 3, 7.5, 12.75 and 14. Its signature declares them all.
    fn param_pixel() -> HostShader {
        let reads: [(u32, u32); 4] = [(3, 0), (7, 2), (12, 3), (14, 0)];
        let mut words = Vec::new();
        for (k, &(row, ch)) in reads.iter().enumerate() {
            let (dst, attr) = (2 + k as u32, row - 1);
            // v_interp_p1_f32 v(dst), v0, attr.ch; v_interp_p2_f32 v(dst), v1, attr.ch
            words.push(0xC800_0000 | dst << 18 | attr << 10 | ch << 8);
            words.push(0xC801_0001 | dst << 18 | attr << 10 | ch << 8);
        }
        words.extend([
            0xF800_180F,
            0x0504_0302, // exp mrt0, v2, v3, v4, v5 done vm
            0xBF81_0000,
        ]);
        let inputs = (1..=PARAM_ROWS as u8)
            .map(|r| crate::native::PsInput {
                usage: crate::native::Usage::Generic,
                row: r,
                mask: 0b1111,
                attr: r - 1,
                flat: false,
                fp16: false,
                pack2: false,
                row2: 0,
            })
            .collect();
        let mut isg1 = param_elements(PARAM_ROWS);
        isg1.remove(0);
        for e in &mut isg1 {
            e.usage = 0;
        }
        let isg1 = container::signature(&isg1, true);
        let osg1 = rows(&[("SV_Target", 16, false)], false);
        emit_pixel(&words, inputs, 1 << 1, &isg1, &osg1).unwrap()
    }

    /// The vertex output semantics of a translated mesh, in `OSG1`.
    fn output_names(dxil: &[u8]) -> Vec<String> {
        let c = Container::parse(dxil).unwrap();
        let mut names: Vec<String> = signature(c.part(FourCC::OSG1).unwrap().data)
            .unwrap()
            .into_iter()
            .map(|e| format!("{}{}", e.name, e.index))
            .collect();
        names.sort();
        names
    }

    /// Metal counts more of [`param_mesh`]'s output than it allows as the
    /// console lays it out, and less packed: packing changes its
    /// translation; a mesh within the limit is made as without it.
    #[test]
    fn mesh_outputs_are_packed_only_over_metals_limit() {
        let els = param_elements(PARAM_ROWS);
        assert!(container::metal_mesh_bytes(&els, &[], 128, 128) > container::METAL_MESH_BYTES);
        let packed: Vec<SigElement> = container::metal_packed_order(&els)
            .into_iter()
            .map(|i| els[i].clone())
            .collect();
        assert_eq!(container::metal_mesh_bytes(&packed, &[], 128, 128), 128 * 240);
        let pack = Options {
            mesh_pack_outputs: true,
            ..Options::default()
        };
        let plain = param_mesh(PARAM_ROWS, Options::default(), None).unwrap();
        let ours = param_mesh(PARAM_ROWS, pack, None).unwrap();
        assert_ne!(plain.dxil, ours.dxil);
        assert_eq!(output_names(&plain.dxil), output_names(&ours.dxil));
        assert!(!mesh_over_metal_limit(&ours.dxil));
        // Ten rows: 16 + 10 × 24 bytes a vertex, within the limit.
        let small = param_mesh(10, pack, None).unwrap();
        assert_eq!(small.dxil, param_mesh(10, Options::default(), None).unwrap().dxil);
        assert!(!mesh_over_metal_limit(&small.dxil));
    }

    /// With what a pixel shader reads kept, a mesh translation keeps those
    /// outputs and its system values and leaves the rest out; one keeping
    /// everything is the plain translation.
    #[test]
    fn a_kept_mesh_translation_keeps_what_the_pixel_shader_reads() {
        let ps = param_pixel();
        let kept = pixel_reads(&ps.dxil).unwrap();
        assert_eq!(
            kept.describe(),
            "TEXCOORD6\nTEXCOORD15\nTEXCOORD25\nTEXCOORD28\n"
        );
        let ms = param_mesh(PARAM_ROWS, Options::default(), Some(&kept)).unwrap();
        assert_eq!(
            output_names(&ms.dxil),
            ["SV_Position0", "TEXCOORD15", "TEXCOORD25", "TEXCOORD28", "TEXCOORD6"]
        );
        let plain = param_mesh(PARAM_ROWS, Options::default(), None).unwrap();
        assert!(mesh_outputs_unread(&plain.dxil, &kept));
        assert!(!mesh_outputs_unread(&ms.dxil, &kept));
        let all = Kept::new(
            param_elements(PARAM_ROWS)
                .into_iter()
                .filter(|e| e.kind == 0)
                .map(|e| (e.name.to_lowercase(), e.index)),
        );
        let everything = param_mesh(PARAM_ROWS, Options::default(), Some(&all)).unwrap();
        assert_eq!(everything.dxil, plain.dxil);
    }

    /// [`param_mesh`] with [`param_pixel`] on the host GPU (and WARP),
    /// packed, kept to what the pixel shader reads, both, and neither: the
    /// pixel shader reads the values the mesh stored under their semantics.
    /// Under Wine, D3DMetal makes the plain one a no-op (Metal: "Total mesh
    /// size (N) exceeds the maximum mesh size allowed (32768)"), so it only
    /// has to draw where Metal is not underneath.
    #[test]
    #[cfg(windows)]
    fn packed_and_kept_mesh_outputs_reach_the_pixel_shader() {
        use super::host::{Draw, Host, Root};
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_FEATURE_D3D12_OPTIONS7, D3D12_FEATURE_DATA_D3D12_OPTIONS7,
            D3D12_MESH_SHADER_TIER_NOT_SUPPORTED, D3D12_SHADER_VISIBILITY_MESH,
        };
        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ps = param_pixel();
        let kept = pixel_reads(&ps.dxil).unwrap();
        let pack = Options {
            mesh_pack_outputs: true,
            ..Options::default()
        };
        let wine = super::host::under_wine();
        let mut ran = 0;
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                eprintln!("no SM 6.6 {label} device; skipped");
                continue;
            };
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            // SAFETY: the query writes the structure of the given size.
            let queried = unsafe {
                host.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS7,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
            };
            if queried.is_err() || options.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED {
                eprintln!("no mesh shaders on the {label} device; skipped");
                continue;
            }
            for (case, options, kept, must_draw) in [
                ("packed", pack, None, true),
                ("kept", Options::default(), Some(&kept), true),
                ("packed and kept", pack, Some(&kept), true),
                ("plain", Options::default(), None, !wine),
            ] {
                let ms = param_mesh(PARAM_ROWS, options, kept).unwrap();
                if let Ok(mut v) = super::validator::Validator::load(
                    super::validator::Validator::default_path().as_deref(),
                ) {
                    v.validate(&mut ms.dxil.clone())
                        .unwrap_or_else(|e| panic!("{case}: {e}"));
                }
                let root = host
                    .root_signature_for(&[(&ms, D3D12_SHADER_VISIBILITY_MESH)])
                    .unwrap();
                let pso = host
                    .mesh_pipeline_with(&root, &ms.dxil, &ps.dxil)
                    .unwrap_or_else(|e| panic!("{label}, {case}: mesh pipeline: {e}"));
                let texels = host
                    .draw(
                        &root,
                        &pso,
                        &Draw {
                            roots: vec![(0, Root::Constants(vec![0]))],
                            textures: &[],
                            samplers: &[],
                            vertices: &[],
                            stride: 0,
                            size: (16, 16),
                            mesh_groups: 1,
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label}, {case}: draw: {e}"));
                let want = [3.0f32, 7.5, 12.75, 14.0];
                let right = texels
                    .iter()
                    .filter(|t| t.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-3))
                    .count();
                eprintln!("{label}, {case}: {right} of 256 texels right (first {:?})", texels[0]);
                if must_draw {
                    assert_eq!(right, 256, "{label}, {case}: first texel {:?}", texels[0]);
                }
            }
            ran += 1;
        }
        if ran == 0 {
            eprintln!("no device with mesh shaders; skipped");
        }
    }

    /// A 16-bit result written over a first interpolation half keeps that
    /// half's high 16 bits, which have no host form: what reads only the low
    /// half translates, a read of the whole dword does not.
    #[test]
    fn a_half_over_a_first_interpolation_half_reads_as_a_half() {
        let program = |read: u32| {
            let words = [
                0xD742_0002u32,
                0x0002_0000, // v_interp_p1ll_f16 v2, v0, attr0.x
                0xD75A_0003,
                0x040A_0200, // v_interp_p2_f16 v3, v1, attr0.x, v2
                0xCC21_0002,
                0x0A01_E503, // v_fma_mixlo_f16 v2, sel_lo(v3), 1.0, 0
                read,        // v4 from v2
                0xF800_180F,
                0x0404_0404, // exp mrt0, v4, v4, v4, v4 done vm
                0xBF81_0000,
            ];
            let isg1 = rows(&[("TEXCOORD", 0, true)], true);
            let osg1 = rows(&[("SV_Target", 16, false)], false);
            emit_pixel(&words, vec![texcoord(true)], 1 << 1, &isg1, &osg1)
        };
        program(0x7E08_1702).unwrap(); // v_cvt_f32_f16 v4, v2
        program(0x3408_0490).unwrap(); // v_lshlrev_b32 v4, 16, v2
        let whole = program(0x7E08_0302).unwrap_err(); // v_mov_b32 v4, v2
        assert!(whole.to_string().contains("first half"), "{whole}");
        let part = program(0x3408_0488).unwrap_err(); // v_lshlrev_b32 v4, 8, v2
        assert!(part.to_string().contains("first half"), "{part}");
    }

    /// A pixel export before the `done` one, both under a lane guard (EXEC
    /// narrowed without a branch): the lanes outside it are discarded at the
    /// `done` export, so the first one stores in every lane. With EXEC
    /// restored before the `done` export those lanes would write only the
    /// second target, which stays refused.
    #[test]
    fn an_export_before_a_discarding_done_export_stores_unguarded() {
        let program = |restore: &[u32]| {
            let mut words = vec![
                0xBE80_037Eu32, // s_mov_b32 s0, exec_lo
                0x7C22_0080,    // v_cmpx_lt_f32 exec, 0, v0
                0xF800_000F,
                0x0000_0000, // exp mrt0, v0, v0, v0, v0
            ];
            words.extend(restore);
            words.extend([
                0xF800_181F,
                0x0000_0000, // exp mrt1, v0, v0, v0, v0 done vm
                0xBF81_0000,
            ]);
            let isg1 = rows(&[("SV_Position", 3, false)], true);
            let osg1 = rows(&[("SV_Target", 16, false), ("SV_Target", 16, false)], false);
            emit_pixel(&words, vec![], 1 << 8, &isg1, &osg1)
        };
        program(&[]).unwrap();
        let e = program(&[0xBEFE_0300]).unwrap_err(); // s_mov_b32 exec_lo, s0
        assert!(e.to_string().contains("a conditional export"), "{e}");
    }

    /// The compiler reuses the j register (`v_mov` under a kill's EXEC) and
    /// interpolates in the other lanes: those still hold j, which
    /// structurisation's lane conditions prove. Interpolating in the lanes
    /// the copy ran in stays refused.
    #[test]
    fn an_interpolation_after_j_is_reused_in_other_lanes_translates() {
        let program = |other_lanes: &[u32]| {
            let mut words = vec![
                0xBE80_037Eu32, // s_mov_b32 s0, exec_lo
                0x7C22_0480,    // v_cmpx_lt_f32 exec, 0, v2
                0x7E02_0280,    // v_mov_b32 v1, 0
            ];
            words.extend(other_lanes);
            words.extend([
                0xC80C_0000, // v_interp_p1_f32 v3, v0, attr0.x
                0xC80D_0001, // v_interp_p2_f32 v3, v1, attr0.x
                0xBEFE_0300, // s_mov_b32 exec_lo, s0
                0xF800_180F,
                0x0303_0303, // exp mrt0, v3, v3, v3, v3 done vm
                0xBF81_0000,
            ]);
            let isg1 = rows(&[("TEXCOORD", 0, false), ("SV_Position", 3, false)], true);
            let osg1 = rows(&[("SV_Target", 16, false)], false);
            emit_pixel(&words, vec![texcoord(false)], 1 << 1 | 1 << 8, &isg1, &osg1)
        };
        program(&[0x8A7E_7E00]).unwrap(); // s_andn2_b32 exec_lo, s0, exec_lo
        let e = program(&[]).unwrap_err();
        assert!(e.to_string().contains("computed barycentrics"), "{e}");
    }

    /// Two elements sharing a row, the one at the higher channels listed
    /// first (as console signatures have them): the signature part keeps
    /// them in element order within the row, as the validator does.
    #[test]
    fn elements_sharing_a_row_keep_their_order() {
        let el = |index, col, cols| SigElement {
            name: "TEXCOORD".into(),
            index,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col,
            cols,
            usage: 0,
            half: false,
        };
        let isg1 = container::signature(&[el(1, 1, 2), el(0, 0, 1)], true);
        let osg1 = rows(&[("SV_Target", 16, false)], false);
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        emit_pixel(&words, vec![texcoord(false)], 1 << 1, &isg1, &osg1).unwrap();
    }

    /// An unread float input packed beside the front face (as console
    /// signatures have it when the float was `nointerpolation`) takes the
    /// row's constant mode: DXIL wants one mode per row.
    #[test]
    fn a_row_keeps_one_interpolation_mode() {
        let el = |name: &str, kind, comp, row, col| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: Some(row),
            col,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = container::signature(
            &[
                el("TEXCOORD", 0, 9, 0, 0),
                el("SV_IsFrontFace", 13, 5, 0, 1),
                el("COLOR", 0, 9, 1, 0),
            ],
            true,
        );
        let osg1 = rows(&[("SV_Target", 16, false)], false);
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let color = crate::native::PsInput {
            row: 1,
            ..texcoord(false)
        };
        emit_pixel(&words, vec![color], 1 << 1, &isg1, &osg1).unwrap();
    }

    /// The LLVM text `dxc -dumpbin` prints for a DXIL container, when dxc
    /// is installed.
    #[cfg(windows)]
    fn dump_dxil(container: &[u8]) -> Option<String> {
        let dxc = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => std::path::PathBuf::from(p),
            None => super::validator::Validator::default_path()?.with_file_name("dxc.exe"),
        };
        if !dxc.exists() {
            return None;
        }
        let file = std::env::temp_dir().join(format!("victory-interp-{}.dxil", std::process::id()));
        std::fs::write(&file, container).ok()?;
        let out = std::process::Command::new(dxc)
            .arg("-dumpbin")
            .arg(&file)
            .output()
            .ok()?;
        let _ = std::fs::remove_file(&file);
        assert!(
            out.status.success(),
            "dxc -dumpbin failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        Some(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    #[cfg(windows)]
    #[test]
    fn interpolated_f16_writes_the_op_sel_half_with_modifiers() {
        // p1_f32 v2, v0, attr0.x; then a p2_f16 completing it into v2.
        let p2 = |dst_high: bool, omod: u32, clamp: bool| {
            let vgpr = |n: u32| 256 + n;
            let mut w0 = 0x35u32 << 26 | 0x35au32 << 16 | 2;
            if dst_high {
                w0 |= 1 << 14; // op_sel[3]: the high destination half.
            }
            if clamp {
                w0 |= 1 << 15;
            }
            let w1 = vgpr(1) << 9 | vgpr(2) << 18;
            [w0, w1 | omod << 27]
        };
        let dump = |dst_high: bool, omod: u32, clamp: bool| {
            let [w0, w1] = p2(dst_high, omod, clamp);
            let words = [
                0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
                w0,
                w1, // v_interp_p2_f16 v2, attr0, v1, v0
                0xF800_180F,
                0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
                0xBF81_0000,
            ];
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: true,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr: 2,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage, half| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0, true)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15, false)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            super::emit(&c, &native, &p, &st, &io, &res).unwrap().dxil
        };
        let (Some(low), Some(high)) = (
            dump_dxil(&dump(false, 0, false)),
            dump_dxil(&dump(true, 1, true)),
        ) else {
            eprintln!("no dxc.exe; DXIL check skipped");
            return;
        };
        // The low-half write keeps the destination's high half.
        assert!(low.contains("-65536"), "low write keeps the high half");
        // The high-half write shifts the result up by 16.
        assert!(high.contains("shl i32"), "op_sel[3] writes the high half");
        // omod mul:2 and clamp (dx.op.unary.f32, opcode 7) are applied.
        assert!(high.contains("2.000000e+00"), "omod multiplies by two");
        assert!(
            high.contains("dx.op.unary.f32(i32 7"),
            "clamp saturates the result"
        );
        assert!(!low.contains("dx.op.unary.f32(i32 7"));
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res);
    }
}
