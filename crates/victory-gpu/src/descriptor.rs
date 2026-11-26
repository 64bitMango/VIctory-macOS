//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, DESCRIPTOR_HEAP, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid};
use crate::resource::{Memory, PAGEABLE_IIDS};

#[cfg(windows)]
pub(crate) mod host;
// Also built for tests, which run where `host` cannot.
#[cfg(any(windows, test))]
mod paged;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    kind: u32,
    count: u32,
    flags: u32,
    nodes: u32,
}

/// Newer titles (the title) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    memory: Option<Memory>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (decision 0007, step
    /// 3); slots written before wait in `pending`.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<host::HostHeap>>,
    #[cfg(windows)]
    pending: std::sync::Mutex<Vec<u32>>,
    /// A shader-visible heap in the title's memory, tracked: the title also
    /// copies descriptors into it itself, as bytes (`sync_heaps`).
    #[cfg(windows)]
    written: Option<crate::watch::Tracked>,
}

pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of the title's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    #[cfg(windows)]
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS)
        {
            return None;
        }
        // Submissions held off while a view heap is made and filled
        // (`Host::quiet`), from before `pending` is taken: `mirror` waits
        // for that with `SHADER_HEAPS` held, which a submission may wait
        // for (behind a thread waiting to hold it alone).
        let _quiet = (self.desc.kind == VIEWS_KIND)
            .then(|| crate::host::host()?.quiet())
            .flatten();
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Publish only after replaying the queued writes.
        self.host
            .get_or_init(|| {
                let step = u64::from(increment(self.desc.kind)?);
                let h = host::HostHeap::new(self.samplers(), self.desc.count)?;
                for i in pending.drain(..) {
                    h.mirror(i, self.start + u64::from(i) * step, false);
                }
                Some(h)
            })
            .as_ref()
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap.
///
/// A read-write lock. As a mutex it was held by every title thread for the
/// whole of each view it wrote into a heap, and by the submitting thread
/// for the whole of each heap sync: the title's recording threads waited
/// 200-390 ms a frame on it (run ab-all1), taking turns at work that is
/// each heap's own. Held shared ([`heaps`]) by what looks a heap up and
/// then works under that heap's locks: [`mirror`], [`sync_heaps`],
/// [`read_textures`], [`release_dynamic`], [`with_host`]. Held alone
/// ([`heaps_alone`]) to register and drop a heap, and by the passes whose
/// steps are one against every slot write: [`make_read`], [`evict`],
/// [`with_slots_still`]. `VICTORY_HEAPS_SHARED=0` holds it alone
/// everywhere, as the mutex was.
///
/// What the mutex gave, and what gives it now:
/// - A registered heap is live while the lock is held: `Heap::drop` holds
///   it alone.
/// - The list does not change under a walk of it: registering holds it
///   alone.
/// - One thread at a time makes a slot: the slot lock
///   (`host::HostHeap::made`), held from reading the slot's record to
///   entering what it views (`host::HostHeap::mirror`). Draws recorded
///   while a title thread wrote or a sync ran always relied on that alone;
///   now title threads and heap syncs do too.
/// - A sync leaves every slot as the title's record of it was when the
///   sync looked: it looks at the records of the pages written, and makes
///   each slot that differs through `mirror`, which reads the record again
///   under the slot lock. A title thread writing a slot meanwhile makes it
///   itself after its write (its own `mirror`), or dirties the page for
///   the next sync (bytes copied without a call); whichever thread takes
///   the slot lock last read the record last. A write that had not
///   returned when the submission began was never part of it: with the
///   mutex its `mirror` could as well wait out the whole sync.
/// - The passes held alone see no slot written meanwhile by a title thread
///   or a sync, as before (draws being recorded were never held off).
///
/// Lock order (each only with those before it held or none): a
/// submission's turn (`Host::submitted`, also `Host::quiet`), this lock,
/// a heap's `pending`, a host heap's slot lock `made`, then the leaves:
/// [`VIEWS`] (shared), `kept`, `crate::bindless::Viewed`, the copies'
/// registries (`crate::shadow`). [`with_slots_still`] takes every heap's
/// slot lock in the list's order, under this lock held alone. Nothing takes
/// this lock while it holds it (a second shared hold behind a waiting
/// exclusive one would not return).
#[cfg(windows)]
static SHADER_HEAPS: std::sync::RwLock<Vec<Registration>> = std::sync::RwLock::new(Vec::new());

/// [`SHADER_HEAPS`] held shared, or alone with `VICTORY_HEAPS_SHARED=0`.
#[cfg(windows)]
enum Heaps {
    Shared(std::sync::RwLockReadGuard<'static, Vec<Registration>>),
    Alone(std::sync::RwLockWriteGuard<'static, Vec<Registration>>),
}

#[cfg(windows)]
impl std::ops::Deref for Heaps {
    type Target = Vec<Registration>;
    fn deref(&self) -> &Vec<Registration> {
        match self {
            Heaps::Shared(heaps) => heaps,
            Heaps::Alone(heaps) => heaps,
        }
    }
}

/// Whether [`heaps`] shares the lock: unless `VICTORY_HEAPS_SHARED=0`.
#[cfg(windows)]
fn heaps_shared() -> bool {
    #[cfg(test)]
    if tests::HEAPS_ALONE.with(|alone| alone.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_HEAPS_SHARED"))
}

/// The registered heaps, live and the same ones until the guard is dropped;
/// other threads may hold them too.
#[cfg(windows)]
fn heaps() -> Heaps {
    if heaps_shared() {
        Heaps::Shared(SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner()))
    } else {
        Heaps::Alone(heaps_alone())
    }
}

/// The registered heaps with no other thread holding them.
#[cfg(windows)]
fn heaps_alone() -> std::sync::RwLockWriteGuard<'static, Vec<Registration>> {
    SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner())
}

/// [`heaps_alone`] for a pass over the heaps' slots: how long it holds them
/// is counted (diagnostics), every title thread writing a view waits it out.
#[cfg(windows)]
fn heaps_for_pass() -> Pass {
    Pass {
        heaps: heaps_alone(),
        #[cfg(feature = "diagnostics")]
        _held: crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::HeapsAloneUs),
    }
}

/// The heaps held alone by a pass (fields drop in order: the time held
/// ends when the lock is given back).
#[cfg(windows)]
struct Pass {
    #[cfg(feature = "diagnostics")]
    _held: crate::watch::Timed,
    heaps: std::sync::RwLockWriteGuard<'static, Vec<Registration>>,
}

#[cfg(windows)]
impl std::ops::Deref for Pass {
    type Target = Vec<Registration>;
    fn deref(&self) -> &Vec<Registration> {
        &self.heaps
    }
}

#[cfg(any(windows, test))]
type Registration = (u64, u64, usize);

/// Registers a heap, retiring any range it overlaps. Descriptor handles carry
/// no heap identity, so two live heaps cannot unambiguously own one address.
#[cfg(any(windows, test))]
fn register_heap(heaps: &mut Vec<Registration>, (start, end): (u64, u64), at: usize) {
    heaps.retain(|&(old_start, old_end, _)| old_end <= start || end <= old_start);
    heaps.push((start, end, at));
}

/// Removes only this heap's registration.
#[cfg(any(windows, test))]
fn unregister_heap(heaps: &mut Vec<Registration>, at: usize) {
    heaps.retain(|&(_, _, heap)| heap != at);
}

#[cfg(any(windows, test))]
static AMBIGUOUS_RANGE: std::sync::Once = std::sync::Once::new();

/// The heap owning `handle`, if exactly one registered range contains it.
#[cfg(any(windows, test))]
fn heap_at(heaps: &[Registration], handle: u64) -> Option<usize> {
    let mut found = None;
    for &(start, end, at) in heaps {
        if (start..end).contains(&handle) && found.replace(at).is_some() {
            AMBIGUOUS_RANGE.call_once(|| {
                tracing::warn!(
                    handle,
                    "overlapping descriptor heaps: descriptor write refused"
                )
            });
            return None;
        }
    }
    found
}

/// Makes the host descriptor for what the title wrote at `handle`, if it
/// is in a shader-visible heap. Keeps `SHADER_HEAPS` held (shared) for the
/// duration so `Heap::drop` cannot free the heap concurrently.
#[cfg(windows)]
fn mirror(handle: u64) {
    // Its wait is counted (diagnostics: "title threads waiting to write
    // views").
    let heaps = if heaps_shared() {
        Heaps::Shared(crate::contend::read(&SHADER_HEAPS, crate::contend::Lock::Heaps))
    } else {
        Heaps::Alone(crate::contend::write(&SHADER_HEAPS, crate::contend::Lock::Heaps))
    };
    let Some(at) = heap_at(&heaps, handle) else {
        return;
    };
    // SAFETY: registered heaps stay alive until `Heap::drop`, which takes
    // the `SHADER_HEAPS` lock we hold here.
    let heap = unsafe { &*(at as *const Heap) };
    let Some(index) = heap.index(handle) else {
        return;
    };
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                host.mirror(index, handle, false);
            }
        }
    }
}

/// Mirrors what the title wrote into its shader-visible heaps without a
/// call: the title copies descriptors (our records) into its heap
/// as bytes, so only a change of its memory shows them. Each slot of a page
/// written since the last sync is made again if its record changed (in the
/// world 1,300-14,400 slots a frame, against a few hundred views made by
/// calls). Called before each submission: shaders take descriptors from the
/// heap at indices in their data, where a slot never mirrored reads nothing.
#[cfg(windows)]
pub(crate) fn sync_heaps() {
    // Held throughout: a heap is not dropped while its slots are made.
    // Shared: title threads write views meanwhile (`SHADER_HEAPS`).
    let heaps = heaps();
    let _syncing = crate::contend::syncing();
    for &(start, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        let (Some(Some(host)), Some(written), Some(step)) =
            (heap.host.get(), &heap.written, increment(heap.desc.kind))
        else {
            continue;
        };
        let step = u64::from(step);
        // (Pages the title has just written are left open and looked at
        // again at the next sync: `crate::watch::rest_ms`. A slot is made
        // again only if its record changed, so a page told twice costs the
        // look.)
        #[cfg(feature = "diagnostics")]
        let taking = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncTakeUs);
        let changed = written.take_resting(start as usize, usize::MAX);
        #[cfg(feature = "diagnostics")]
        drop(taking);
        // The slots to make of every page, then made together: on several
        // threads where there are enough (`host::HostHeap::make_runs`).
        let mut runs = Vec::new();
        for (from, to) in changed {
            // By page: one with nothing mapped (the title unmaps heap pages
            // too) holds no records.
            let page_size = crate::watch::PAGE;
            for page in (from & !(page_size - 1)..to).step_by(page_size) {
                let (page, end) = (page.max(from), (page + page_size).min(to));
                #[cfg(feature = "diagnostics")]
                let probing = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncProbeUs);
                if !crate::host::read_into(page as *const u8, &mut [0u8; 4]) {
                    continue;
                }
                #[cfg(feature = "diagnostics")]
                drop(probing);
                let (first, last) = ((page as u64 - start) / step, (end as u64 - start).div_ceil(step));
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(
                    crate::copystat::C::SyncSlots,
                    last.min(u64::from(heap.desc.count)).saturating_sub(first),
                );
                host.due_runs(first as u32, last.min(u64::from(heap.desc.count)) as u32, start, step, &mut runs);
            }
        }
        host.make_runs(&runs, start, step);
    }
}

/// Makes the views of slots `slots` that were left null (textures the title
/// placed itself, textures given up) in every shader-visible view heap, now
/// that shaders read them (`crate::shadow::take_reads`), and stamps what
/// they view as used (`crate::resource::evict`).
#[cfg(windows)]
pub(crate) fn make_read(slots: &[u32]) {
    // The placed textures those slots view are made first, without the
    // heaps' lock, which every title thread writing a view waits on (making
    // a texture takes up to milliseconds; title threads waited 1-6 s a frame,
    // runs evict2, noevict1). They are never released, so they outlive it;
    // the lock is held only to look the slots up and to make the views.
    let placed: Vec<View> = {
        let heaps = heaps_for_pass();
        heaps
            .iter()
            // SAFETY: registered heaps are live (removed when dropped).
            .map(|&(_, _, at)| unsafe { &*(at as *const Heap) })
            .filter(|heap| heap.desc.kind == VIEWS_KIND)
            .filter_map(|heap| heap.host.get()?.as_ref().map(|host| (heap, host)))
            .flat_map(|(heap, host)| slots.iter().filter(|&&i| i < heap.desc.count).filter_map(|&i| host.unmade_placed(i)))
            .collect()
    };
    #[cfg(test)]
    tests::while_making(slots);
    // Counted with the submitting thread's work on slots (diagnostics).
    let _syncing = crate::contend::syncing();
    // Several at a time (each a committed texture; hundreds in a new shot,
    // 7 % of the submitting thread when made one by one, run evict5).
    crate::texture::in_parallel(placed.len(), |i| {
        let _ = host::placed_texture(&placed[i]);
    });
    let heaps = heaps_for_pass();
    for &(start, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        let (Some(Some(host)), Some(step)) = (heap.host.get(), increment(heap.desc.kind)) else {
            continue;
        };
        if heap.desc.kind != VIEWS_KIND {
            continue;
        }
        for &i in slots.iter().filter(|&&i| i < heap.desc.count) {
            host.mirror(i, start + u64::from(i) * u64::from(step), true);
            host.stamp_read(i);
        }
    }
}

/// The host textures (raw pointers) that slots `slots` of the
/// shader-visible view heaps view.
#[cfg(windows)]
pub(crate) fn read_textures(slots: &[u32]) -> std::collections::HashSet<usize> {
    let heaps = heaps();
    let mut textures = std::collections::HashSet::new();
    for &(_, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        if let (Some(Some(host)), VIEWS_KIND) = (heap.host.get(), heap.desc.kind) {
            textures.extend(host.textures_at(slots));
        }
    }
    textures
}

/// Every shader-visible view heap's slots viewing one of the host textures
/// `textures` (raw pointers) are left null, made again when read
/// (`crate::resource::evict` gives those textures up).
#[cfg(windows)]
pub(crate) fn evict(textures: &std::collections::HashSet<usize>) {
    let heaps = heaps_for_pass();
    for &(_, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        if let (Some(Some(host)), VIEWS_KIND) = (heap.host.get(), heap.desc.kind) {
            host.evict(textures);
        }
    }
}

/// Runs `f` while no title heap slot is being made: every shader-visible
/// heap's slot record lock is held, which [`host::HostHeap::mirror`] holds
/// from making a slot's view to entering what it views in
/// `crate::bindless`. What the heaps' slots view, as `crate::bindless` has
/// it, stays so until `f` returns (`f` must not make slots itself). `f` is
/// given a look at the slots themselves: of host resources (raw pointers),
/// those some title slot views.
#[cfg(windows)]
pub(crate) fn with_slots_still<R>(
    f: impl FnOnce(&dyn Fn(&std::collections::HashSet<usize>) -> std::collections::HashSet<usize>) -> R,
) -> R {
    let heaps = heaps_for_pass();
    let hosts: Vec<&host::HostHeap> = heaps
        .iter()
        // SAFETY: registered heaps are live (removed when dropped, which
        // waits for the lock we hold).
        .filter_map(|&(_, _, at)| unsafe { &*(at as *const Heap) }.host.get()?.as_ref())
        .collect();
    let _held: Vec<_> = hosts.iter().map(|h| h.hold_slots()).collect();
    f(&|resources| hosts.iter().flat_map(|h| h.viewing(resources)).collect())
}

/// Gives back dynamic slots of the host heap behind the title heap at
/// `heap`, if that heap still exists.
#[cfg(windows)]
pub(crate) fn release_dynamic(heap: usize, first: u32, n: u32) {
    let heaps = heaps();
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.release(first, n);
    }
}

/// Runs `f` on the host heap behind the title heap at `heap` while that
/// heap cannot be released; None if it is gone or has no host heap.
#[cfg(windows)]
pub(crate) fn with_host<R>(heap: usize, f: impl FnOnce(&host::HostHeap) -> R) -> Option<R> {
    let heaps = heaps();
    if !heaps.iter().any(|&(_, _, h)| h == heap) {
        return None;
    }
    // SAFETY: registered heaps are live (removed when dropped, which waits
    // for the lock we hold).
    let host = unsafe { &*(heap as *const Heap) }.host.get()?.as_ref()?;
    Some(f(host))
}

#[cfg(windows)]
impl Drop for Heap {
    fn drop(&mut self) {
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        let mut heaps = heaps_alone();
        unregister_heap(&mut heaps, at);
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

impl RtvRecord {
    /// The view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn depth_desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        const TEXTURE_2D: u32 = 3;
        self.host_desc(TEXTURE_2D)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    #[cfg(windows)]
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = if self.described != 0 {
            self.desc
        } else {
            [
                crate::resource::view_format(self.resource)?,
                dimension,
                0,
                0,
                0,
                0,
            ]
        };
        words[0] = crate::resource::pc_format(words[0]);
        // SAFETY: both host view descriptions are 24 bytes in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }
}

/// Device method: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: the title passes a description or null.
    let described = unsafe { desc.as_ref() };
    let record = RtvRecord {
        tag: RTV_TAG,
        described: u32::from(described.is_some()),
        resource,
        desc: described.copied().unwrap_or_default(),
    };
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// Device method: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(
    this: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    create_rtv(this, resource, desc, handle);
    if handle != 0 {
        // SAFETY: just written by `create_rtv`.
        unsafe { (handle as *mut u32).write_unaligned(DSV_TAG) };
    }
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
///
/// Views without a resource object have nothing that releases them, and a
/// title copies descriptors as bytes, so nothing says when the last
/// descriptor naming an entry is gone: such entries are kept for the run.
/// The title writes thousands of constant buffer views a frame, each at
/// the address its constants have that frame, so as entries they grew the
/// table without end: 544 MB of entries after seven minutes, about
/// 0.28 GB more a minute. So the two kinds that carry an address are not
/// entries ([`inline_words`]): their records name them whole, and the
/// table holds only what is bounded: views of resource objects (freed with
/// the resource), samplers (as many as there are sampler states), and the
/// shapes of placed views ([`SHAPES`]). `VICTORY_VIEW_INLINE=0`: every
/// view is an entry, as before.
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    /// Each entry's generation again, alone: heap syncs ask about every
    /// record of the pages written ([`views_current`]), and an entry of
    /// `list` is 136 bytes, a cache line fetched per slot asked about.
    generations: Vec<u32>,
}

/// Where each view's entry is, the entries naming each resource, and the
/// entries free: what recording and releasing a view look up and change.
struct Index {
    index: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
}

// Read by every list recorder for every descriptor it binds: readers share
// it. As a mutex it made the title's recording threads take turns (Belobug
// ff920f2: 6,600 context switches a second each in a PIX timing capture).
//
// The entries alone: with the index beside them under one lock, each of the
// 9,500 views a frame the title records for the first time held every
// reader off for its search and insert in a tree of 136-byte keys, and the
// readers (200,000 looks a frame, the submitting thread's among them)
// waited 75 ms a frame, 10,000 times (run with what slots view in parts).
// The index has its own lock ([`VIEW_INDEX`]); the entries are held alone
// only to write one entry.
//
// What the one lock gave, and what gives it now:
// - An entry's view and generation are read together: both are written
//   with this lock held alone.
// - A record is written with its entry's own generation: `record` holds
//   the index (shared for a view it finds, alone for a new one) from
//   finding the entry to writing the record, and `forget`, which moves
//   generations on, holds the index alone.
// - An entry is freed and given to another view under the index held
//   alone; its generation moved on when it was freed, so a record of the
//   view it held before names no view from then on.
//
// Locked in the order: the index, then the entries.
// `VICTORY_VIEWS_SPLIT=0`: readers hold the index too (shared), so a view
// recorded or released holds them off for all of it, as under the one lock.
static VIEWS: std::sync::RwLock<Views> = std::sync::RwLock::new(Views { list: Vec::new(), generations: Vec::new() });

static VIEW_INDEX: std::sync::RwLock<Index> = std::sync::RwLock::new(Index {
    index: std::collections::BTreeMap::new(),
    by_resource: std::collections::BTreeMap::new(),
    free: Vec::new(),
});

/// Whether readers of the entries leave the index alone ([`VIEWS`]): unless
/// `VICTORY_VIEWS_SPLIT=0`.
fn views_split() -> bool {
    #[cfg(test)]
    if tests::VIEWS_ONE.with(|one| one.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_VIEWS_SPLIT"))
}

/// The entries, to read; with the index held too (dropped after them) if
/// the two are not split.
struct Entries {
    views: std::sync::RwLockReadGuard<'static, Views>,
    _index: Option<std::sync::RwLockReadGuard<'static, Index>>,
}

impl std::ops::Deref for Entries {
    type Target = Views;
    fn deref(&self) -> &Views {
        &self.views
    }
}

/// The entries, to read. Not for a thread that holds the index.
fn entries() -> Entries {
    let _index = (!views_split()).then(|| crate::contend::read(&VIEW_INDEX, crate::contend::Lock::Views));
    Entries { views: crate::contend::read(&VIEWS, crate::contend::Lock::Views), _index }
}

const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

// A record is the tag and two words, `i` and `generation` (the host keeps
// them as one 64-bit key per slot). The top two bits of `generation` say
// what the other 62 bits are: 0, entry `i` of the table at that
// generation; [`CONSTANT`], a constant buffer view, whole; [`PLACED`], a
// view of placed memory: its address and the number of its shape.
const KIND_SHIFT: u32 = 30;
/// Table generations stay below the kind bits (and above 0).
const GENERATION_MASK: u32 = (1 << KIND_SHIFT) - 1;
const CONSTANT: u32 = 1;
const PLACED: u32 = 2;
/// Bits of a constant buffer view's address in its record: any address a
/// program can have. The 15 bits above them are its size.
const CONSTANT_ADDRESS_BITS: u32 = 47;
/// Bits of a placed view's address in its record: placed memory is title
/// memory, which lies below 1 TiB. The 22 bits above them number its shape.
const PLACED_ADDRESS_BITS: u32 = 40;
const SHAPE_LIMIT: usize = 1 << (62 - PLACED_ADDRESS_BITS);

/// A view of placed memory without its address: whether it is unordered
/// access, the resource description, and the view's words but the two that
/// hold the address (10 and 11).
type Shape = (bool, [u32; 14], [u32; 14]);

/// The shapes of placed views named whole in their records, by number, and
/// where each is. Only ever added to: as many as the title has kinds of
/// textures and buffers, however often it moves them. A lock of its own,
/// apart from the entries and their index: nearly every placed view has a
/// shape seen before, which readers share.
struct Shapes {
    list: Vec<Shape>,
    index: std::collections::BTreeMap<Shape, u32>,
}

static SHAPES: std::sync::RwLock<Shapes> =
    std::sync::RwLock::new(Shapes { list: Vec::new(), index: std::collections::BTreeMap::new() });

/// The number of shapes in [`SHAPES`], for [`views_current`] to know a
/// record's shape without the lock (stored after the shape is there).
static SHAPES_KNOWN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Views named whole in their records since the start: constant buffer
/// views, views of placed memory (for the views line).
static NAMED_WHOLE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Table entries made for constant buffer views and views of placed
/// memory (never freed: they name no resource), counted as made so the
/// views line need not go through the table.
static ADDRESS_ENTRIES: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether constant buffer views and views of placed memory are named whole
/// in their records ([`Views`]). On unless `VICTORY_VIEW_INLINE=0`, which
/// makes every view a table entry, as before.
fn view_inline() -> bool {
    #[cfg(test)]
    if tests::ENTRIES_ONLY.with(|only| only.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_VIEW_INLINE"))
}

/// The two record words of the 62 bits `payload` of kind `kind`.
fn kind_words(kind: u32, payload: u64) -> [u32; 2] {
    [payload as u32, kind << KIND_SHIFT | (payload >> 32) as u32]
}

/// The 62 bits two record words carry beside their kind.
fn payload(i: u32, generation: u32) -> u64 {
    u64::from(generation & GENERATION_MASK) << 32 | u64::from(i)
}

/// The record words that name the constant buffer view of `size` bytes at
/// `address` whole; `None` if they cannot: an address past 47 bits, or a
/// size of 16 KiB or more that is no multiple of 256 or reaches 4 MiB (the
/// size has 15 bits: bytes below 16 KiB, else units of 256 bytes). Such a
/// view is a table entry.
fn constant_words(address: u64, size: u32) -> Option<[u32; 2]> {
    if address >> CONSTANT_ADDRESS_BITS != 0 {
        return None;
    }
    let code = if size < 1 << 14 {
        u64::from(size)
    } else if size.is_multiple_of(256) && size >> 8 < 1 << 14 {
        1 << 14 | u64::from(size >> 8)
    } else {
        return None;
    };
    Some(kind_words(CONSTANT, code << CONSTANT_ADDRESS_BITS | address))
}

/// The constant buffer view [`constant_words`] named.
fn constant_of(i: u32, generation: u32) -> View {
    let payload = payload(i, generation);
    let code = (payload >> CONSTANT_ADDRESS_BITS) as u32;
    let size = if code >> 14 == 0 { code } else { (code & 0x3fff) << 8 };
    View::Constant(payload & ((1 << CONSTANT_ADDRESS_BITS) - 1), size)
}

/// A placed view's shape and its address.
fn shape_of(unordered: bool, resource: &[u32; 14], desc: &[u32; 16]) -> (Shape, u64) {
    let mut rest = [0u32; 14];
    rest[..10].copy_from_slice(&desc[..10]);
    rest[10..].copy_from_slice(&desc[12..]);
    ((unordered, *resource, rest), u64::from(desc[10]) | u64::from(desc[11]) << 32)
}

/// The number of `shape`, a new one if it is not there yet; `None` when the
/// shapes are full (the view is then a table entry).
fn shape_number(shape: Shape) -> Option<u32> {
    if let Some(&at) = SHAPES.read().unwrap_or_else(|e| e.into_inner()).index.get(&shape) {
        return Some(at);
    }
    let mut shapes = SHAPES.write().unwrap_or_else(|e| e.into_inner());
    // Looked up again: another thread may have entered it meanwhile.
    if let Some(&at) = shapes.index.get(&shape) {
        return Some(at);
    }
    if shapes.list.len() >= SHAPE_LIMIT {
        return None;
    }
    let at = shapes.list.len() as u32;
    shapes.list.push(shape);
    shapes.index.insert(shape, at);
    SHAPES_KNOWN.store(shapes.list.len(), std::sync::atomic::Ordering::Release);
    Some(at)
}

/// The record words that name the view of placed memory at `address` with
/// shape number `shape`.
fn placed_words(shape: u32, address: u64) -> [u32; 2] {
    kind_words(PLACED, u64::from(shape) << PLACED_ADDRESS_BITS | address)
}

/// The view of placed memory [`placed_words`] named, if its shape is known.
fn placed_of(i: u32, generation: u32) -> Option<View> {
    let payload = payload(i, generation);
    let address = payload & ((1 << PLACED_ADDRESS_BITS) - 1);
    let shapes = SHAPES.read().unwrap_or_else(|e| e.into_inner());
    let &(unordered, resource, rest) = shapes.list.get((payload >> PLACED_ADDRESS_BITS) as usize)?;
    let mut desc = [0u32; 16];
    desc[..10].copy_from_slice(&rest[..10]);
    (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
    desc[12..].copy_from_slice(&rest[10..]);
    Some(View::Placed(unordered, resource, desc))
}

/// The two words that name `view` whole in a record, where its kind allows
/// and `VICTORY_VIEW_INLINE` is not `0` ([`Views`]); else `None`, and the
/// view is a table entry.
fn inline_words(view: &View) -> Option<[u32; 2]> {
    use std::sync::atomic::Ordering::Relaxed;
    if !view_inline() {
        return None;
    }
    match view {
        View::Constant(address, size) => {
            let words = constant_words(*address, *size)?;
            NAMED_WHOLE[0].fetch_add(1, Relaxed);
            Some(words)
        }
        View::Placed(unordered, resource, desc) => {
            let (shape, address) = shape_of(*unordered, resource, desc);
            if address >> PLACED_ADDRESS_BITS != 0 {
                return None;
            }
            let words = placed_words(shape_number(shape)?, address);
            NAMED_WHOLE[1].fetch_add(1, Relaxed);
            Some(words)
        }
        _ => None,
    }
}

/// Whether the record words `i`, `generation` of a kind other than a table
/// entry name a view: a constant buffer view always does, a placed view if
/// its shape is known.
fn inline_current(i: u32, generation: u32, shapes: usize) -> bool {
    match generation >> KIND_SHIFT {
        CONSTANT => true,
        PLACED => ((payload(i, generation) >> PLACED_ADDRESS_BITS) as usize) < shapes,
        _ => false,
    }
}

impl View {
    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }
}

fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    // A view its record names whole needs neither the index nor an entry.
    if let Some([i, generation]) = inline_words(&view) {
        write_record(handle, i, generation);
        #[cfg(all(windows, feature = "diagnostics"))]
        crate::copystat::add(crate::copystat::C::RecordInline, 1);
        #[cfg(windows)]
        mirror(handle);
        return;
    }
    // A view recorded before (the title records the same views again and
    // again, into other slots) is only looked up: recorders share the
    // index, and took turns at it. Its record is written with the index
    // held, as below: its generation is the entry's own then (`forget`
    // holds the index alone).
    if record_shared() {
        let index = crate::contend::read(&VIEW_INDEX, crate::contend::Lock::ViewIndex);
        if let Some(&i) = index.index.get(&view) {
            let generation = VIEWS.read().unwrap_or_else(|e| e.into_inner()).generations[i as usize];
            write_record(handle, i, generation);
            drop(index);
            #[cfg(all(windows, feature = "diagnostics"))]
            crate::copystat::add(crate::copystat::C::RecordFound, 1);
            #[cfg(windows)]
            mirror(handle);
            return;
        }
    }
    let mut index = crate::contend::write(&VIEW_INDEX, crate::contend::Lock::ViewIndex);
    // Looked up again: another thread may have entered it meanwhile.
    let i = match index.index.get(&view) {
        Some(&i) => {
            #[cfg(all(windows, feature = "diagnostics"))]
            crate::copystat::add(crate::copystat::C::RecordFound, 1);
            i
        }
        None => {
            let free = index.free.pop();
            let i = {
                // The entries held alone for the one entry written.
                let mut views = crate::contend::write(&VIEWS, crate::contend::Lock::ViewIndex);
                match free {
                    Some(i) => {
                        views.list[i as usize].0 = view.clone();
                        i
                    }
                    None => {
                        views.list.push((view.clone(), 1));
                        views.generations.push(1);
                        views.list.len() as u32 - 1
                    }
                }
            };
            for r in view.resources().into_iter().filter(|&r| r != 0) {
                index.by_resource.entry(r).or_default().push(i);
            }
            match view {
                View::Constant(..) => ADDRESS_ENTRIES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                View::Placed(..) => ADDRESS_ENTRIES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                _ => 0,
            };
            index.index.insert(view, i);
            #[cfg(all(windows, feature = "diagnostics"))]
            crate::copystat::add(crate::copystat::C::RecordNew, 1);
            i
        }
    };
    let generation = VIEWS.read().unwrap_or_else(|e| e.into_inner()).generations[i as usize];
    write_record(handle, i, generation);
    drop(index);
    #[cfg(windows)]
    mirror(handle);
}

/// Writes the record of view entry `i`, generation `generation`, at `handle`:
/// the tag, then the entry and its generation in one store of eight bytes.
/// Threads recording views share the views table, so two may write one slot
/// at once (a title's race, but under the table's lock one record or the
/// other came out whole): the tag is the same from both, and a store of
/// eight bytes within a cache line lands whole on the processors this runs
/// on, so the slot holds one thread's entry with its own generation, not
/// one's entry with the other's generation, which names no view.
fn write_record(handle: u64, i: u32, generation: u32) {
    // SAFETY: the handle is a descriptor slot, at least 16 bytes.
    unsafe {
        (handle as *mut u32).write_unaligned(VIEW_TAG);
        ((handle + 4) as *mut u64).write_unaligned(u64::from(generation) << 32 | u64::from(i));
    }
}

/// Times a thread waited to enter what a host heap's slot holds, and the
/// microseconds, since last asked (`paged::WAITS`; diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn kept_waits() -> [u64; 2] {
    [0, 1].map(|k| paged::WAITS[k].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// Whether [`record`] looks a view up with the table shared before it
/// takes it alone to enter a new one: unless `VICTORY_RECORD_SHARED=0`.
fn record_shared() -> bool {
    #[cfg(test)]
    if tests::RECORD_ALONE.with(|alone| alone.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_RECORD_SHARED"))
}

/// The title released `resource`: the views naming it go stale.
pub(crate) fn forget(resource: usize) {
    let mut index = VIEW_INDEX.write().unwrap_or_else(|e| e.into_inner());
    let Some(entries) = index.by_resource.remove(&resource) else {
        return;
    };
    // Found with the entries shared, their generations moved on with them
    // held alone, at once: no recorder looks meanwhile (the index is held).
    let mut freed = Vec::new();
    {
        let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
        for i in entries {
            let (view, _) = &views.list[i as usize];
            // Entries freed through a view's other resource may hold another
            // view by now.
            if index.index.get(view) != Some(&i) || !view.resources().contains(&resource) {
                continue;
            }
            index.index.remove(view);
            freed.push(i);
        }
    }
    let mut views = VIEWS.write().unwrap_or_else(|e| e.into_inner());
    for i in freed {
        // (Below the bits that say a record is no table entry.)
        let generation = (views.list[i as usize].1.wrapping_add(1) & GENERATION_MASK).max(1);
        views.list[i as usize].1 = generation;
        views.generations[i as usize] = generation;
        // The view's other resource (a counter) keeps a stale entry, which
        // `forget` of that resource skips.
        index.free.push(i);
    }
}

/// The view the record words `i` and `generation` name: the one they hold
/// whole ([`inline_words`]), or the view of entry `i` if `generation` is
/// still its own.
fn view_at(i: u32, generation: u32) -> Option<View> {
    match generation >> KIND_SHIFT {
        0 => {
            let views = entries();
            let (view, current) = views.list.get(i as usize)?;
            (*current == generation).then(|| view.clone())
        }
        CONSTANT => Some(constant_of(i, generation)),
        PLACED => placed_of(i, generation),
        _ => None,
    }
}

/// For each record (tag, index, generation): whether it names a view that
/// is current, as [`view_at`] would find it (one lock for them all). Asked
/// of the generations kept alone, which are the entries' own (written with
/// them, under the lock); of the entries themselves with
/// `VICTORY_SLOT_GENERATIONS=0`.
#[cfg_attr(not(windows), allow(dead_code))]
fn views_current(records: &[[u32; 3]]) -> Vec<bool> {
    let shapes = SHAPES_KNOWN.load(std::sync::atomic::Ordering::Acquire);
    let inline = |i: u32, generation: u32| (generation >> KIND_SHIFT != 0).then(|| inline_current(i, generation, shapes));
    let views = entries();
    if generations_alone() {
        let generations = &views.generations;
        return records
            .iter()
            .map(|&[tag, i, generation]| {
                tag == VIEW_TAG
                    && inline(i, generation).unwrap_or_else(|| generations.get(i as usize) == Some(&generation))
            })
            .collect();
    }
    records
        .iter()
        .map(|&[tag, i, generation]| {
            tag == VIEW_TAG
                && inline(i, generation).unwrap_or_else(|| {
                    views.list.get(i as usize).is_some_and(|(_, current)| *current == generation)
                })
        })
        .collect()
}

/// Whether [`views_current`] reads the generations kept alone: on unless
/// `VICTORY_SLOT_GENERATIONS=0`.
#[cfg_attr(not(windows), allow(dead_code))]
fn generations_alone() -> bool {
    #[cfg(test)]
    if tests::FROM_ENTRIES.with(|off| off.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_SLOT_GENERATIONS"))
}

/// The view recorded at `handle`, if one is (and its resource is alive).
pub(crate) fn view(handle: u64) -> Option<View> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    if tag != VIEW_TAG {
        return None;
    }
    view_at(i, generation)
}

/// For the views line: the entries of the table ([`Views`]), those of them
/// free for reuse, and the bytes the table, its index and the shapes hold
/// (an estimate: an entry of the list and of the generations, and one of
/// the index with its share of a tree node; the shapes likewise); then the
/// entries that are constant buffer views and views of placed memory (those
/// that did not fit a record, or all of them with `VICTORY_VIEW_INLINE=0`),
/// the shapes of placed views kept, and the views named whole in their
/// records since the start: constant buffer views, views of placed memory.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn views_kept() -> [u64; 8] {
    use std::sync::atomic::Ordering::Relaxed;
    let (index, free) = {
        let index = VIEW_INDEX.read().unwrap_or_else(|e| e.into_inner());
        (index.index.len() as u64, index.free.len() as u64)
    };
    let shapes = {
        let shapes = SHAPES.read().unwrap_or_else(|e| e.into_inner());
        (shapes.list.len() as u64, shapes.list.capacity() as u64)
    };
    let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
    let each = size_of::<(View, u32)>() as u64;
    let shape = size_of::<Shape>() as u64;
    let [constants, placed] = [0, 1].map(|k| ADDRESS_ENTRIES[k].load(Relaxed));
    [
        views.list.len() as u64,
        free,
        views.list.capacity() as u64 * each
            + views.generations.capacity() as u64 * 4
            + index * each * 3 / 2
            + shapes.1 * shape
            + shapes.0 * shape * 3 / 2,
        constants,
        placed,
        shapes.0,
        NAMED_WHOLE[0].load(Relaxed),
        NAMED_WHOLE[1].load(Relaxed),
    ]
}

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

/// Device method.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    if let Some([lo, hi, size, _]) = words(desc) {
        let address = u64::from(lo) | u64::from(hi) << 32;
        record(handle, View::Constant(address, size));
    }
}

/// Device method.
pub(crate) extern "system" fn create_srv(
    _: This,
    resource: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    record(handle, View::Shader(resource as usize, words(desc)));
}

/// Device method.
pub(crate) extern "system" fn create_uav(
    _: This,
    resource: This,
    counter: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    let view = View::Unordered(resource as usize, counter as usize, words(desc));
    record(handle, view);
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`.
pub(crate) extern "system" fn create_placed_srv(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(false, resource, desc));
    }
}

pub(crate) extern "system" fn create_placed_uav(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(true, resource, desc));
    }
}

/// Device method.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let Some(step) = increment(kind).map(u64::from) else {
        return;
    };
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        // SAFETY: both ranges are descriptor slots the title owns.
        unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len as usize) };
        #[cfg(windows)]
        for i in 0..u64::from(count) {
            mirror(dst + i * step);
        }
    }
}

/// Device method: ranges of descriptors (start and count each) gathered
/// from the sources and scattered to the destinations, in order.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            .map(|i| {
                // SAFETY: the title passes `n` starts and sizes (or null sizes).
                unsafe {
                    (
                        *starts.add(i),
                        if sizes.is_null() { 1 } else { *sizes.add(i) },
                    )
                }
            })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const RtvRecord).read_unaligned() };
    (record.tag == tag).then_some(record)
}

/// Device method.
pub(crate) extern "system" fn create_heap(
    _: This,
    desc_ptr: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1)))
    else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    tracing::debug!(?desc, placed, "descriptor heap");
    let (start, memory) = if placed {
        if placement.address.checked_add(size).is_none() {
            return E_INVALIDARG;
        }
        (placement.address, None)
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        memory,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        pending: Default::default(),
        #[cfg(windows)]
        written: (placed && visible)
            .then(|| crate::watch::track(start as usize, size as usize, true))
            .flatten(),
    });
    #[cfg(windows)]
    let at = std::ptr::from_ref::<Heap>(&heap) as usize;
    #[cfg(not(windows))]
    let _ = visible;
    let result = object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out);
    #[cfg(windows)]
    if visible && result == crate::object::S_OK {
        let mut heaps = heaps_alone();
        register_heap(&mut heaps, (start, start + size), at);
    }
    result
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &HEAP_IIDS).then_some(this);
    object::answer(DESCRIPTOR_HEAP, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
}
/// The same address for the CPU and the GPU start.
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<DESCRIPTOR_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = desc as *const (); // 0x40
    f[9] = start as *const (); // 0x48 CPU
    f[10] = start as *const (); // 0x50 GPU
    Table(f)
};

#[cfg(test)]
pub(crate) mod tests {
    /// Work to run while `make_read` makes the textures of a read slot (the
    /// slot), outside the heaps' lock.
    pub(crate) static WHILE_MAKING: std::sync::Mutex<Option<(u32, Work)>> = std::sync::Mutex::new(None);

    /// The work [`WHILE_MAKING`] runs.
    pub(crate) type Work = Box<dyn FnOnce() + Send>;

    pub(crate) fn while_making(slots: &[u32]) {
        let mut hook = WHILE_MAKING.lock().unwrap_or_else(|e| e.into_inner());
        if hook.as_ref().is_some_and(|(slot, _)| slots.contains(slot))
            && let Some((_, work)) = hook.take()
        {
            drop(hook);
            work();
        }
    }

    use super::*;
    use crate::object::{S_OK, slot};

    thread_local! {
        /// This thread asks the view entries themselves whether records are
        /// current (as `VICTORY_SLOT_GENERATIONS=0`).
        pub(crate) static FROM_ENTRIES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// This thread takes the views table alone for every record (as
        /// `VICTORY_RECORD_SHARED=0`).
        pub(crate) static RECORD_ALONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// This thread's reads of the views' entries hold their index too (as
        /// `VICTORY_VIEWS_SPLIT=0`).
        pub(crate) static VIEWS_ONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// This thread holds the heaps' lock alone wherever it takes it (as
        /// `VICTORY_HEAPS_SHARED=0`).
        pub(crate) static HEAPS_ALONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// This thread makes every view a table entry, none named whole in
        /// its record (as `VICTORY_VIEW_INLINE=0`).
        pub(crate) static ENTRIES_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Held by tests that count on entries' generations: one releases and
    /// makes views thousands of times, which moves freed entries'
    /// generations on by any number while another looks at a generation
    /// "never given".
    static GENERATIONS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Whether the generations kept alone are the entries' own.
    fn generations_match() -> bool {
        let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
        views.generations.len() == views.list.len()
            && views.list.iter().zip(&views.generations).all(|((_, own), alone)| own == alone)
    }

    /// The generations kept alone follow the entries' through views made,
    /// released and made again at freed entries, and records are found
    /// current or stale the same from either: records of live views, of
    /// released ones, of an entry's next generation, with another tag, and
    /// past the entries.
    #[test]
    fn records_are_current_by_the_generations_kept_alone_as_by_the_entries() {
        let _alone = GENERATIONS.lock().unwrap_or_else(|e| e.into_inner());
        let null = std::ptr::null();
        let slots = [0u64; 4 * 64];
        let at = |i: usize| slots.as_ptr() as u64 + 32 * i as u64;
        // SAFETY: slots of `slots`.
        let record = |i: usize| unsafe { (at(i) as *const [u32; 3]).read_unaligned() };
        let both = |records: &[[u32; 3]]| {
            let alone = views_current(records);
            FROM_ENTRIES.with(|f| f.set(true));
            let entries = views_current(records);
            FROM_ENTRIES.with(|f| f.set(false));
            assert_eq!(alone, entries);
            alone
        };
        // Resources no other test names: other tests' views come and go
        // meanwhile, these only here.
        let resource = |k: usize| 0x5eed_0000_0000 + 0x100 * k;
        for k in 0..32 {
            create_srv(null, resource(k) as This, std::ptr::null(), at(k));
        }
        assert!(generations_match());
        let mut records: Vec<[u32; 3]> = (0..32).map(record).collect();
        assert_eq!(both(&records), [true; 32]);
        // Every other resource released: its records are stale.
        (0..32).step_by(2).for_each(|k| forget(resource(k)));
        assert!(generations_match());
        let stale: Vec<bool> = (0..32).map(|k| k % 2 == 1).collect();
        assert_eq!(both(&records), stale);
        // Made again (freed entries are reused): new records are current,
        // the old ones stay stale.
        for k in (0..32).step_by(2) {
            create_srv(null, resource(k) as This, std::ptr::null(), at(32 + k));
        }
        assert!(generations_match());
        assert_eq!(both(&records), stale);
        records.extend((0..32).step_by(2).map(|k| record(32 + k)));
        // Not records of ours: another tag, an entry that does not exist, a
        // generation never given.
        let [tag, i, generation] = records[1];
        records.extend([[tag ^ 1, i, generation], [tag, u32::MAX, 1], [tag, i, generation.wrapping_add(7)], [0; 3]]);
        let mut expected = stale;
        expected.extend([true; 16]);
        expected.extend([false; 4]);
        assert_eq!(both(&records), expected);
        (0..32).for_each(|k| forget(resource(k)));
        assert!(generations_match());
        assert_eq!(both(&records), [false; 52]);
    }

    /// Threads record views of the same few resources into slots of their
    /// own while others release those resources: a slot then reads back as
    /// the view its thread recorded there or as nothing (its resource
    /// released since), never as another view, and the generations kept
    /// alone stay the entries' own. With the table shared for views
    /// recorded before, and taken alone for each.
    #[test]
    fn views_recorded_and_released_from_several_threads_read_back_as_recorded() {
        let _alone = GENERATIONS.lock().unwrap_or_else(|e| e.into_inner());
        const THREADS: usize = 6;
        const ROUNDS: usize = if cfg!(debug_assertions) { 20_000 } else { 200_000 };
        // Resources no other test names.
        let resource = |k: usize| 0x5eed_7000_0000 + 0x100 * k;
        std::thread::scope(|scope| {
            for t in 0..THREADS {
                scope.spawn(move || {
                    RECORD_ALONE.with(|alone| alone.set(t % 3 == 2));
                    VIEWS_ONE.with(|one| one.set(t % 2 == 1));
                    let slots = [0u64; 4 * 8];
                    let at = |i: usize| slots.as_ptr() as u64 + 32 * i as u64;
                    let mut seed = 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(t as u64 + 1);
                    for _ in 0..ROUNDS {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let (k, i, what) = ((seed >> 33) as usize % 8, (seed >> 40) as usize % 8, seed >> 60);
                        if what < 3 {
                            forget(resource(k));
                            continue;
                        }
                        // With a description or without: two views a resource.
                        let desc = (what & 1 == 1).then_some([k as u32; 10]);
                        let desc_ptr = desc.as_ref().map_or(std::ptr::null(), std::ptr::from_ref);
                        create_srv(std::ptr::null(), resource(k) as This, desc_ptr, at(i));
                        match view(at(i)) {
                            None => {}
                            Some(found) => assert_eq!(found, View::Shader(resource(k), desc)),
                        }
                        // SAFETY: a slot of `slots`.
                        let record = unsafe { (at(i) as *const [u32; 3]).read_unaligned() };
                        assert_eq!(record[0], VIEW_TAG);
                        assert!(record[2] != 0, "generations start at 1");
                    }
                });
            }
        });
        assert!(generations_match());
        (0..8).for_each(|k| forget(resource(k)));
        assert!(generations_match());
    }

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    #[test]
    fn registering_a_heap_retires_overlapping_ranges_but_keeps_adjacent_ones() {
        let mut heaps = vec![(10, 20, 1), (20, 30, 2), (30, 40, 3), (40, 50, 4)];
        register_heap(&mut heaps, (15, 35), 5);
        assert_eq!(heaps, [(40, 50, 4), (15, 35, 5)]);
        register_heap(&mut heaps, (35, 40), 6);
        assert_eq!(heaps, [(40, 50, 4), (15, 35, 5), (35, 40, 6)]);
    }

    #[test]
    fn lookup_is_half_open_and_refuses_ambiguous_ranges() {
        let heaps = [(10, 20, 1), (20, 30, 2)];
        assert_eq!(heap_at(&heaps, 10), Some(1));
        assert_eq!(heap_at(&heaps, 19), Some(1));
        assert_eq!(heap_at(&heaps, 20), Some(2));
        assert_eq!(heap_at(&heaps, 30), None);
        assert_eq!(heap_at(&[(10, 30, 1), (20, 40, 2)], 25), None);
    }

    #[test]
    fn unregistering_a_retired_heap_keeps_its_replacement() {
        let mut heaps = vec![(10, 20, 1)];
        register_heap(&mut heaps, (10, 20), 2);
        unregister_heap(&mut heaps, 1);
        assert_eq!(heaps, [(10, 20, 2)]);
    }

    #[cfg(windows)]
    struct HeapOwner(This);

    #[cfg(windows)]
    impl Drop for HeapOwner {
        fn drop(&mut self) {
            if !self.0.is_null() {
                slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
            }
        }
    }

    #[cfg(windows)]
    fn owner(desc: HeapDesc, placement: Placement) -> HeapOwner {
        let mut this = std::ptr::null();
        assert_eq!(create(desc, placement, &mut this), S_OK);
        HeapOwner(this)
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc {
            kind: 0,
            count: 1000,
            flags: 1,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 4 * 8]; // four 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3) = (at(0), at(1), at(2), at(3));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0), Some(View::Constant(0x1_0000_1000, 256)));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1), Some(View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 3]) }, unsafe {
            *(s2 as *const [u32; 3])
        });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(
            null,
            1,
            dst.as_ptr(),
            dst_n.as_ptr(),
            2,
            src.as_ptr(),
            src_n.as_ptr(),
            0,
        );
        assert_eq!(view(s2), view(s0));
        assert_eq!(view(s3), view(s1));
        assert_eq!(view(at(0) + 8), None);
        heap[0] = 0;
    }

    /// A released resource's views go stale; the same view of a new
    /// resource at the same address is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = 0x7777_0000usize as This;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Shader(0x7777_0000, None)));
        forget(0x7777_0000);
        assert_eq!(view(a), None, "stale after release");
        create_srv(null, resource, std::ptr::null(), b);
        assert_eq!(view(b), Some(View::Shader(0x7777_0000, None)));
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 3]), *(b as *const [u32; 3])) };
        assert_ne!(old, new, "a new generation");
    }

    /// Constant buffer views are named whole in their records: any address
    /// a program can have, any size below 16 KiB, multiples of 256 bytes
    /// up to 4 MiB; what does not fit says so.
    #[test]
    fn constant_views_fit_their_records() {
        let top = (1u64 << CONSTANT_ADDRESS_BITS) - 1;
        for address in [0u64, 0x100, 0x2602_0000, 0xd_6b60_0000, 0x7ffd_a34f_0000, top] {
            for size in [0u32, 1, 255, 256, 4000, (1 << 14) - 1, 1 << 14, 65536, (1 << 22) - 256] {
                let [i, generation] = constant_words(address, size).expect("fits");
                assert_eq!(generation >> KIND_SHIFT, CONSTANT);
                assert_eq!(view_at(i, generation), Some(View::Constant(address, size)), "{address:#x} {size}");
                assert_eq!(views_current(&[[VIEW_TAG, i, generation]]), [true]);
            }
        }
        assert_eq!(constant_words(top + 1, 256), None, "an address past 47 bits");
        assert_eq!(constant_words(0x1000, (1 << 14) + 1), None, "16 KiB or more, no multiple of 256");
        assert_eq!(constant_words(0x1000, 1 << 22), None, "4 MiB");
        // Different views never share their words.
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1000, 512));
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1100, 256));
        assert_ne!(constant_words(0x1000, 1 << 14), constant_words(0x1000, 64));
        // A kind no record has names nothing.
        assert_eq!(view_at(0, 3 << KIND_SHIFT), None);
        assert_eq!(views_current(&[[VIEW_TAG, 0, 3 << KIND_SHIFT]]), [false]);
    }

    /// The table's entries that are views of addresses in `addresses`, or
    /// placed views of `resource`; and the shapes of `resource`.
    fn address_entries(addresses: std::ops::Range<u64>, resource: [u32; 14]) -> (usize, usize) {
        let mine = |v: &View| match v {
            View::Constant(address, _) => addresses.contains(address),
            View::Placed(_, r, _) => *r == resource,
            _ => false,
        };
        let entries = VIEWS.read().unwrap().list.iter().filter(|(v, _)| mine(v)).count();
        let shapes = SHAPES.read().unwrap().list.iter().filter(|s| s.1 == resource).count();
        (entries, shapes)
    }

    /// The leak: a title that writes constant buffer views and views of
    /// placed memory at ever new addresses adds nothing to the table; one
    /// shape stands for all the placed views alike. With
    /// `VICTORY_VIEW_INLINE=0` each such view is an entry, kept for the run.
    #[test]
    fn views_at_new_addresses_add_no_entries() {
        if !view_inline() {
            return;
        }
        let null = std::ptr::null();
        let slots = [0u64; 4];
        let at = slots.as_ptr() as u64;
        // A resource description and a view no other test uses.
        let resource: [u32; 14] = std::array::from_fn(|k| 0x5ee0_0000 + k as u32);
        let mut desc: [u32; 16] = std::array::from_fn(|k| 0x7a70_0000 + k as u32);
        let addresses = 0x6_0000_0000..0x6_1000_0000;
        // SAFETY: a slot of `slots`.
        let record = || unsafe { (at as *const [u32; 3]).read_unaligned() };
        for n in 0..20_000u64 {
            let address = 0x6_0000_0000 + n * 0x100;
            create_cbv(null, &[address as u32, (address >> 32) as u32, 512, 0], at);
            assert_eq!(view(at), Some(View::Constant(address, 512)));
            assert_eq!(views_current(&[record()]), [true]);
            let placed = 0xd_4940_0000 + n * 0x1_0000;
            (desc[10], desc[11]) = (placed as u32, (placed >> 32) as u32);
            create_placed_srv(null, &resource, &desc, at);
            assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
            assert_eq!(views_current(&[record()]), [true]);
        }
        assert_eq!(address_entries(addresses.clone(), resource), (0, 1), "no entries, one shape for 20,000 placed views");
        // The same shape, unordered: another shape.
        create_placed_uav(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(true, resource, desc)));
        assert_eq!(address_entries(addresses.clone(), resource), (0, 2));
        // Placed memory past 1 TiB (not title memory) is a table entry.
        (desc[10], desc[11]) = (0, 0x100);
        create_placed_srv(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
        assert_eq!(address_entries(addresses.clone(), resource), (1, 2));
        // A placed record of a shape not made yet names nothing.
        let unknown = placed_words(SHAPE_LIMIT as u32 - 1, 0x1000);
        assert_eq!(view_at(unknown[0], unknown[1]), None);
        assert_eq!(views_current(&[[VIEW_TAG, unknown[0], unknown[1]]]), [false]);
        // A record's words are the same for the same view (the host knows
        // a slot's view by them), and differ with the address.
        // SAFETY: both handles are slots of 32 bytes.
        let words = |handle: u64| unsafe { *(handle as *const [u32; 3]) };
        let other = [0u64; 4];
        create_cbv(null, &[0x40, 6, 512, 0], at);
        create_cbv(null, &[0x40, 6, 512, 0], other.as_ptr() as u64);
        assert_eq!(words(at), words(other.as_ptr() as u64));
        create_cbv(null, &[0x140, 6, 512, 0], other.as_ptr() as u64);
        assert_ne!(words(at), words(other.as_ptr() as u64));

        // As `VICTORY_VIEW_INLINE=0`: every new address is a new entry.
        ENTRIES_ONLY.with(|only| only.set(true));
        let before = address_entries(addresses.clone(), resource).0;
        for n in 0..100u64 {
            let address = 0x6_0800_0000 + n * 0x100;
            create_cbv(null, &[address as u32, (address >> 32) as u32, 512, 0], at);
            assert_eq!(view(at), Some(View::Constant(address, 512)));
            assert_eq!(record()[2] >> KIND_SHIFT, 0, "a table entry");
        }
        ENTRIES_ONLY.with(|only| only.set(false));
        assert_eq!(address_entries(addresses, resource).0, before + 100);
    }

    /// A record copied as bytes names its constant buffer or placed view
    /// wherever it lands, as an entry's record does.
    #[test]
    fn views_named_whole_survive_copies() {
        let null = std::ptr::null();
        let heap = [0u64; 4 * 4];
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let resource: [u32; 14] = std::array::from_fn(|k| 0x5ee1_0000 + k as u32);
        let mut desc: [u32; 16] = std::array::from_fn(|k| 0x7a71_0000 + k as u32);
        (desc[10], desc[11]) = (0x4940_0000, 0xd);
        create_cbv(null, &[0x2000, 7, 1 << 16, 0], at(0));
        create_placed_uav(null, &resource, &desc, at(1));
        let (dst, dst_n, src, src_n) = ([at(2)], [2u32], [at(0), at(1)], [1u32, 1]);
        copy(null, 1, dst.as_ptr(), dst_n.as_ptr(), 2, src.as_ptr(), src_n.as_ptr(), 0);
        assert_eq!(view(at(2)), Some(View::Constant(0x7_0000_2000, 1 << 16)));
        assert_eq!(view(at(3)), Some(View::Placed(true, resource, desc)));
    }

    /// The hot path of constant buffer and placed views, named whole in
    /// their records against table entries (`VICTORY_VIEW_INLINE=0`): title
    /// threads recording them (four at once, each its own slots), then the
    /// host reading them back as a heap sync does (whether each record is
    /// current, then its view). Per "frame" each thread records views at
    /// addresses not seen before, a share of them twice (a view held
    /// already), placed views over a few hundred shapes; the table grows
    /// frame by frame with entries, as in a run.
    #[test]
    #[ignore = "synthetic release benchmark; run with --release --ignored --nocapture"]
    fn benchmark_views_named_whole() {
        use std::hint::black_box;
        use std::time::Instant;

        const THREADS: u64 = 4;
        const CONSTANTS: u64 = 2_000;
        const PLACED: u64 = 3_000;
        const SHAPES_USED: u64 = 300;
        const FRAMES: u64 = 40;
        let per_thread = (CONSTANTS + PLACED) as usize;
        for (mode, entries_only) in [("named whole", false), ("table entries", true), ("named whole", false)] {
            // Addresses this mode alone uses (each pass its own).
            static PASS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let pass = PASS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut heaps: Vec<Vec<[u64; 4]>> = (0..THREADS).map(|_| vec![[0u64; 4]; per_thread]).collect();
            let (mut record_ns, mut current_ns, mut view_ns) = (Vec::new(), Vec::new(), Vec::new());
            for frame in 0..FRAMES {
                let start = Instant::now();
                std::thread::scope(|s| {
                    for (t, heap) in heaps.iter_mut().enumerate() {
                        s.spawn(move || {
                            ENTRIES_ONLY.with(|only| only.set(entries_only));
                            let null = std::ptr::null();
                            let base = 0x10_0000_0000 * (pass + 1) + (frame * THREADS + t as u64) * 0x100_0000;
                            for (k, slot) in heap.iter_mut().enumerate() {
                                let k = k as u64;
                                let at = slot.as_mut_ptr() as u64;
                                // One in four the view of the slot before (held already).
                                let n = if k % 4 == 3 { k - 1 } else { k };
                                if k < CONSTANTS {
                                    let address = base + n * 0x100;
                                    create_cbv(null, &[address as u32, (address >> 32) as u32, 512, 0], at);
                                } else {
                                    let shape = n % SHAPES_USED;
                                    let resource: [u32; 14] = std::array::from_fn(|w| 0x5ee2_0000 + w as u32);
                                    let mut desc: [u32; 16] = std::array::from_fn(|w| 0x7a72_0000 + w as u32 + shape as u32 * 0x100);
                                    let address = (base + n * 0x1_0000) & ((1 << PLACED_ADDRESS_BITS) - 1);
                                    (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
                                    create_placed_srv(null, &resource, &desc, at);
                                }
                            }
                            ENTRIES_ONLY.with(|only| only.set(false));
                        });
                    }
                });
                record_ns.push(start.elapsed().as_secs_f64() * 1e9 / (THREADS as f64 * per_thread as f64));
                let records: Vec<[u32; 3]> = heaps
                    .iter()
                    .flatten()
                    // SAFETY: a slot of `heaps`.
                    .map(|slot| unsafe { (slot.as_ptr() as *const [u32; 3]).read_unaligned() })
                    .collect();
                let start = Instant::now();
                let current = black_box(views_current(black_box(&records)));
                current_ns.push(start.elapsed().as_secs_f64() * 1e9 / records.len() as f64);
                assert!(current.iter().all(|&c| c), "{mode}: every record current");
                let start = Instant::now();
                for slot in heaps.iter().flatten() {
                    black_box(view(slot.as_ptr() as u64).expect("a view"));
                }
                view_ns.push(start.elapsed().as_secs_f64() * 1e9 / records.len() as f64);
            }
            let median = |v: &mut Vec<f64>| {
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            let [first, last] = [record_ns[0], record_ns[record_ns.len() - 1]];
            println!(
                "{mode:13}: record {:6.0} ns/view (median; first frame {first:.0}, last {last:.0}), current {:5.1} ns/record, view {:5.1} ns/record; table {} entries",
                median(&mut record_ns),
                median(&mut current_ns),
                median(&mut view_ns),
                VIEWS.read().unwrap().list.len(),
            );
        }
    }

    /// A record the title copies into its shader-visible heap as bytes (no
    /// call) is made on the host at the next sync, and only then.
    #[cfg(windows)]
    #[test]
    fn records_copied_as_bytes_are_made_at_the_next_sync() {
        let memory = crate::watch::tests::pool(1);
        let desc = HeapDesc {
            kind: SAMPLERS,
            count: 64,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        let placement = Placement {
            address: memory as u64,
            size: 64 * 16,
        };
        assert_eq!(create(desc, placement, &mut h), S_OK);
        if let Some(host) = heap(h).host() {
            // Our record, written where no heap sees it, then copied in.
            let mut elsewhere = [0u64; 2];
            let at = elsewhere.as_mut_ptr() as u64;
            create_sampler(std::ptr::null(), &[21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0], at);
            let slot = memory as u64 + 5 * 16;
            // SAFETY: 16 bytes of each, ours; the heap's page is guarded
            // (our handler opens it).
            unsafe { std::ptr::copy_nonoverlapping(at as *const u8, slot as *mut u8, 16) };
            assert!(!host.made(5));
            sync_heaps();
            assert!(host.made(5), "made at the sync");
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        crate::watch::tests::free(memory);
    }

    /// Title threads write descriptors into two placed heaps (samplers, and
    /// views of placed textures) by calls, by copies through the device and
    /// as bytes, clear them, bind them as draws do and look them up, all at
    /// the same slots, while another thread runs heap syncs and the passes
    /// that hold the heaps alone. Once they are done and a sync has run,
    /// every host slot is made from the record its title slot holds (from
    /// none where that names no view), and what the view heap's slots view
    /// is entered for exactly those: as in a second pair of heaps given
    /// the same bytes by one thread and synced once. A third heap, in
    /// memory of ours, no sync looks at: placed texture views are written
    /// into it by calls only, and each slot is as the last call left it
    /// with no sync's help.
    #[cfg(windows)]
    fn slots_written_during_syncs_end_as_their_records(alone: bool, one_lock: bool) {
        use host::tests::ONE_LOCK;
        use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const SLOTS: u32 = 512;
        // Other tests' passes name low slots of every view heap.
        const FIRST: u32 = 64;
        const WRITERS: u64 = 4;
        const ROUNDS: u64 = 5_000;
        HEAPS_ALONE.with(|a| a.set(alone));
        ONE_LOCK.with(|one| one.set(one_lock));
        // What the slots view in one part too (as `VICTORY_VIEWED_PARTS=0`).
        crate::bindless::tests::ONE_PART.with(|one| one.set(one_lock));
        // The syncs' runs on helper threads, or all on the syncing thread
        // (as `VICTORY_SYNC_THREADS`).
        let helpers = Some(if one_lock { 1 } else { 4 });
        host::tests::SYNC_THREADS.with(|threads| threads.set(helpers));
        let page = crate::watch::PAGE;
        let pages = (SLOTS as usize * 32).div_ceil(page);
        // Each heap in pages of its own: samplers, views, and the pair one
        // thread fills afterwards.
        let memory = crate::watch::tests::pool(4 * pages);
        let base = |heap: usize| (memory + heap * pages * page) as u64;
        let make = |heap: usize, kind: u32| {
            let size = u64::from(SLOTS) * u64::from(increment(kind).unwrap_or(0));
            let desc = HeapDesc { kind, count: SLOTS, flags: SHADER_VISIBLE, nodes: 1 };
            owner(desc, Placement { address: base(heap), size })
        };
        let owners = [make(0, SAMPLERS), make(1, VIEWS_KIND), make(2, SAMPLERS), make(3, VIEWS_KIND)];
        let unwatched = owner(
            HeapDesc { kind: VIEWS_KIND, count: SLOTS, flags: SHADER_VISIBLE, nodes: 1 },
            Placement { address: 0, size: 0 },
        );
        let unwatched_start = heap(unwatched.0).start;
        if owners.iter().chain([&unwatched]).any(|o| heap(o.0).host().is_none()) {
            drop(unwatched);
            drop(owners);
            crate::watch::tests::free(memory);
            return;
        }
        // Memory the placed textures' views name (never read: such a view
        // is left null until a shader reads its slot, which none does).
        let textures = vec![0u8; 16 << 12];
        let textures_at = textures.as_ptr() as u64;
        // Memory of a title buffer that constant buffer views name: their
        // slots are made at a sync and hold the buffer's host copy.
        let constants = vec![0u8; 16 << 8];
        let constants_at = constants.as_ptr() as u64;
        crate::shadow::register(constants_at as usize, constants_at, 16 << 8);
        let sampler_heap = owners[0].0 as usize;
        let stop = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let stop = &stop;
            scope.spawn(move || {
                HEAPS_ALONE.with(|a| a.set(alone));
                ONE_LOCK.with(|one| one.set(one_lock));
                crate::bindless::tests::ONE_PART.with(|one| one.set(one_lock));
                host::tests::SYNC_THREADS.with(|threads| threads.set(helpers));
                let host = crate::host::host().expect("checked");
                let mut round = 0u32;
                while !stop.load(Relaxed) {
                    host.between_submissions(sync_heaps);
                    round += 1;
                    if round.is_multiple_of(4) {
                        with_slots_still(|_| ());
                        let _ = read_textures(&[FIRST, FIRST + 1]);
                        evict(&Default::default());
                        make_read(&[]);
                    }
                }
            });
            let writers: Vec<_> = (0..WRITERS)
                .map(|t| {
                    scope.spawn(move || {
                        HEAPS_ALONE.with(|a| a.set(alone));
                        ONE_LOCK.with(|one| one.set(one_lock));
                        crate::bindless::tests::ONE_PART.with(|one| one.set(one_lock));
                        let null = std::ptr::null();
                        // Records of 16 samplers and of 16 placed texture
                        // views, where no heap sees them.
                        let (sampler_records, view_records) = ([0u64; 2 * 16], [0u64; 4 * 16]);
                        let from = |views: bool, k: u64| match views {
                            false => sampler_records.as_ptr() as u64 + 16 * k,
                            true => view_records.as_ptr() as u64 + 32 * k,
                        };
                        let sampler = |k: u64| [21, 1 + k as u32 % 4, 1 + k as u32 / 4, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
                        // A 4x1 RGBA8 texture, linear, and a 2D view of it.
                        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
                        let placed = |k: u64| {
                            let mut view = [0u32; 16];
                            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                            let address = textures_at + (k << 12);
                            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
                            view
                        };
                        for k in 0..16 {
                            create_sampler(null, &sampler(k), from(false, k));
                            create_placed_srv(null, &words, &placed(k), from(true, k));
                        }
                        let mut seed = 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(t + 1);
                        for _ in 0..ROUNDS {
                            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                            let (what, k, wide) = (seed >> 60, seed >> 56 & 15, seed >> 52 & 15);
                            // Mostly the same few slots as the other threads.
                            let span = if wide == 0 { u64::from(SLOTS - FIRST) - 4 } else { 24 };
                            let slot = FIRST + ((seed >> 20) % span) as u32;
                            // A third of the writes into the heap no sync
                            // looks at: by calls only.
                            let own = (seed >> 48) % 3 == 0;
                            let views = own || what & 1 == 1;
                            let (kind, step) = if views { (VIEWS_KIND, 32u64) } else { (SAMPLERS, 16) };
                            let start = if own { unwatched_start } else { base(usize::from(views)) };
                            let at = start + u64::from(slot) * step;
                            match what >> 1 {
                                2..=4 if own => create_placed_srv(null, &words, &placed(k), at),
                                0 if views && !own && k % 2 == 1 => {
                                    let cbv = [(constants_at + (k << 8)) as u32, ((constants_at + (k << 8)) >> 32) as u32, 256, 0];
                                    create_cbv(null, &cbv, at);
                                }
                                0 if views => create_placed_srv(null, &words, &placed(k), at),
                                0 => create_sampler(null, &sampler(k), at),
                                1 => copy_simple(null, 1 + (k as u32 & 3), at, from(views, k.min(12)), kind),
                                // As bytes, with no call: the next sync finds it.
                                // SAFETY: a record of ours into a slot of the heap.
                                2 | 3 => unsafe {
                                    std::ptr::copy_nonoverlapping(from(views, k) as *const u8, at as *mut u8, 12)
                                },
                                // SAFETY: a slot of the heap.
                                4 => unsafe { std::ptr::write_bytes(at as *mut u8, 0, step as usize) },
                                // As a draw binding the slot through a table
                                // (sampler slots: a placed texture's would be
                                // made from memory that is no texture).
                                5 | 6 if !views => {
                                    let title = heap(sampler_heap as This);
                                    let _ = title.host().and_then(|h| h.mirror(slot, at, true));
                                }
                                _ => {
                                    let _ = view(at);
                                }
                            }
                        }
                    })
                })
                .collect();
            for writer in writers {
                writer.join().expect("a writing thread");
            }
            stop.store(true, Relaxed);
        });
        // One thread gives the second pair the same bytes.
        for h in 0..2 {
            let bytes = SLOTS as usize * if h == 0 { 16 } else { 32 };
            // SAFETY: both heaps' memory, of `bytes` each.
            unsafe { std::ptr::copy_nonoverlapping(base(h) as *const u8, base(h + 2) as *mut u8, bytes) };
        }
        host.between_submissions(sync_heaps);
        for h in 0..2 {
            let step = if h == 0 { 16 } else { 32 };
            let (raced, single) = (heap(owners[h].0).host().expect("checked"), heap(owners[h + 2].0).host().expect("checked"));
            let (mut made, mut holding) = (0, 0);
            for i in 0..SLOTS {
                // SAFETY: a slot of the heap.
                let record = unsafe { ((base(h) + u64::from(i) * step) as *const [u32; 3]).read_unaligned() };
                let key = match views_current(&[record])[0] {
                    true => u64::from(record[2]) << 32 | u64::from(record[1]),
                    false => 0,
                };
                assert_eq!(raced.made_key(i), key, "heap {h} slot {i}: made from its record");
                assert_eq!(raced.made_key(i), single.made_key(i), "heap {h} slot {i}: as one thread leaves it");
                assert_eq!(raced.kept(i), single.kept(i), "heap {h} slot {i}: holding what one thread leaves");
                assert!(raced.held_agrees(i) && single.held_agrees(i), "heap {h} slot {i}: its bit says what it holds");
                made += usize::from(key != 0);
                holding += usize::from(raced.kept(i));
            }
            if h == 1 {
                assert!(made > 0, "views were written");
                assert!(holding > 0, "constant buffer views hold their copies");
                assert_eq!(raced.viewed.slots(), made, "what the slots view is entered for the slots made");
                assert_eq!(single.viewed.slots(), made);
            }
        }
        let raced = heap(unwatched.0).host().expect("checked");
        let mut made = 0;
        for i in 0..SLOTS {
            // SAFETY: a slot of the heap.
            let record = unsafe { ((unwatched_start + u64::from(i) * 32) as *const [u32; 3]).read_unaligned() };
            let key = match views_current(&[record])[0] {
                true => u64::from(record[2]) << 32 | u64::from(record[1]),
                false => 0,
            };
            assert_eq!(raced.made_key(i), key, "the heap no sync looks at, slot {i}: made from its record");
            made += usize::from(key != 0);
        }
        assert!(made > 0, "views were written");
        assert_eq!(raced.viewed.slots(), made, "what the slots view is entered for the slots made");
        drop(unwatched);
        drop(owners);
        crate::shadow::unregister(constants_at as usize, constants_at);
        drop(constants);
        crate::watch::tests::free(memory);
        HEAPS_ALONE.with(|a| a.set(false));
        ONE_LOCK.with(|one| one.set(false));
        crate::bindless::tests::ONE_PART.with(|one| one.set(false));
        host::tests::SYNC_THREADS.with(|threads| threads.set(None));
    }

    /// Several threads write the same slots of a heap no sync looks at, at
    /// once, by calls (placed texture views, left null until read, and
    /// cleared slots): after each round every slot is made from the record
    /// it holds, whichever thread's write was the last.
    #[cfg(windows)]
    #[test]
    fn the_last_of_several_threads_writing_a_slot_at_once_makes_it() {
        last_of_several_threads_makes_a_slot(false);
    }

    /// The same with one lock for all of a heap's slots (as
    /// `VICTORY_SLOT_LOCKS=0`).
    #[cfg(windows)]
    #[test]
    fn the_last_of_several_threads_writing_a_slot_at_once_makes_it_under_one_lock() {
        last_of_several_threads_makes_a_slot(true);
    }

    #[cfg(windows)]
    fn last_of_several_threads_makes_a_slot(one_lock: bool) {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        const SLOTS: u32 = 96;
        const FIRST: u32 = 64;
        let title = owner(
            HeapDesc { kind: VIEWS_KIND, count: SLOTS, flags: SHADER_VISIBLE, nodes: 1 },
            Placement { address: 0, size: 0 },
        );
        let start = heap(title.0).start;
        // The heap's locks are as this thread has them when it is made.
        host::tests::ONE_LOCK.with(|one| one.set(one_lock));
        let host = heap(title.0).host();
        host::tests::ONE_LOCK.with(|one| one.set(false));
        let Some(host) = host else { return };
        let textures = vec![0u8; 16 << 12];
        let textures_at = textures.as_ptr() as u64;
        for round in 0..400u64 {
            let barrier = std::sync::Barrier::new(4);
            std::thread::scope(|scope| {
                for t in 0..4u64 {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        host::tests::ONE_LOCK.with(|one| one.set(one_lock));
                        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
                        let mut view = [0u32; 16];
                        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                        let address = textures_at + ((round + t) % 16 << 12);
                        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
                        barrier.wait();
                        for i in FIRST..SLOTS {
                            let at = start + u64::from(i) * 32;
                            if (round + t + u64::from(i)).is_multiple_of(5) {
                                // A cleared slot, written by a copy of zeros.
                                let zeros = [0u64; 4];
                                copy_simple(std::ptr::null(), 1, at, zeros.as_ptr() as u64, VIEWS_KIND);
                            } else {
                                create_placed_srv(std::ptr::null(), &words, &view, at);
                            }
                        }
                    });
                }
            });
            for i in FIRST..SLOTS {
                // SAFETY: a slot of the heap.
                let record = unsafe { ((start + u64::from(i) * 32) as *const [u32; 3]).read_unaligned() };
                let key = match views_current(&[record])[0] {
                    true => u64::from(record[2]) << 32 | u64::from(record[1]),
                    false => 0,
                };
                assert_eq!(host.made_key(i), key, "round {round}, slot {i}");
            }
        }
    }

    /// A slot made from one record is written, at once, another record by
    /// one thread and the record it is made from again by a second: it ends
    /// made from whichever record it holds. (The second thread finds the
    /// slot made from its record while the first is making it from the
    /// other: it must not take the slot for made.)
    #[cfg(windows)]
    #[test]
    fn a_slot_written_back_to_what_it_is_made_from_while_another_record_is_made_ends_as_its_record() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        const SLOTS: u32 = 96;
        const FIRST: u32 = 64;
        let title = owner(
            HeapDesc { kind: SAMPLERS, count: SLOTS, flags: SHADER_VISIBLE, nodes: 1 },
            Placement { address: 0, size: 0 },
        );
        let start = heap(title.0).start;
        let Some(host) = heap(title.0).host() else { return };
        let sampler = |k: u32| [21, 1 + k, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
        for i in FIRST..SLOTS {
            create_sampler(std::ptr::null(), &sampler(0), start + u64::from(i) * 16);
        }
        for round in 0..1500u32 {
            let barrier = std::sync::Barrier::new(2);
            std::thread::scope(|scope| {
                // Each slot is made from one of the two: one thread writes
                // it that one again, the other the other.
                for k in 0..2 {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        for i in FIRST..SLOTS {
                            create_sampler(std::ptr::null(), &sampler(k), start + u64::from(i) * 16);
                        }
                    });
                }
            });
            for i in FIRST..SLOTS {
                // SAFETY: a slot of the heap.
                let record = unsafe { ((start + u64::from(i) * 16) as *const [u32; 3]).read_unaligned() };
                assert!(views_current(&[record])[0], "round {round}, slot {i}: one thread's record, whole");
                let key = u64::from(record[2]) << 32 | u64::from(record[1]);
                assert_eq!(host.made_key(i), key, "round {round}, slot {i}");
            }
        }
    }

    /// A heap sync (its runs on helper threads) and a title thread make the
    /// same slots at once: the sync from records copied in as bytes, the
    /// title thread from records it writes by calls meanwhile. Each slot
    /// ends made from the record it holds, with no sync after.
    #[cfg(windows)]
    #[test]
    fn a_sync_and_a_title_thread_making_the_same_slots_leave_each_as_its_record() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const SLOTS: u32 = 2048;
        const FIRST: u32 = 64;
        let memory = crate::watch::tests::pool(1);
        let title = owner(
            HeapDesc { kind: SAMPLERS, count: SLOTS, flags: SHADER_VISIBLE, nodes: 1 },
            Placement { address: memory as u64, size: u64::from(SLOTS) * 16 },
        );
        let Some(slots) = heap(title.0).host() else {
            drop(title);
            return crate::watch::tests::free(memory);
        };
        let at = |i: u32| memory as u64 + u64::from(i) * 16;
        let sampler = |k: u32| [21, 1 + k % 4, 1 + k / 4 % 4, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
        // Records of sixteen samplers, where no heap sees them.
        let records = [0u64; 2 * 16];
        let from = |k: u32| records.as_ptr() as u64 + 16 * u64::from(k % 16);
        (0..16).for_each(|k| create_sampler(std::ptr::null(), &sampler(k), from(k)));
        for round in 0..300u32 {
            // No other test's submission syncs the heap meanwhile (it would
            // take the changed pages and still be making their slots when
            // this round looks).
            host.between_submissions(|| {
                // Copied in as bytes: the sync finds these (another sampler
                // than the round before left).
                for i in FIRST..SLOTS {
                    // SAFETY: a record of ours into a slot of the heap.
                    unsafe { std::ptr::copy_nonoverlapping(from(round + i) as *const u8, at(i) as *mut u8, 12) };
                }
                let barrier = std::sync::Barrier::new(2);
                std::thread::scope(|scope| {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        host::tests::SYNC_THREADS.with(|threads| threads.set(Some(4)));
                        barrier.wait();
                        sync_heaps();
                    });
                    scope.spawn(move || {
                        barrier.wait();
                        for i in (FIRST..SLOTS).step_by(3) {
                            create_sampler(std::ptr::null(), &sampler(round + i + 5), at(i));
                        }
                    });
                });
                for i in FIRST..SLOTS {
                    // SAFETY: a slot of the heap.
                    let record = unsafe { (at(i) as *const [u32; 3]).read_unaligned() };
                    let key = match views_current(&[record])[0] {
                        true => u64::from(record[2]) << 32 | u64::from(record[1]),
                        false => 0,
                    };
                    assert_eq!(slots.made_key(i), key, "round {round}, slot {i}");
                }
            });
        }
        drop(title);
        crate::watch::tests::free(memory);
    }

    #[cfg(windows)]
    #[test]
    fn slots_written_from_several_threads_during_syncs_end_as_their_records() {
        for _ in 0..3 {
            slots_written_during_syncs_end_as_their_records(false, false);
        }
    }

    /// The same with the heaps' lock held alone wherever it is taken (as
    /// `VICTORY_HEAPS_SHARED=0`) and one lock for all of a heap's slots (as
    /// `VICTORY_SLOT_LOCKS=0`, `VICTORY_SYNC_RUNS=0`), and with only the
    /// latter.
    #[cfg(windows)]
    #[test]
    fn slots_written_from_several_threads_end_as_their_records_with_the_heaps_held_alone() {
        slots_written_during_syncs_end_as_their_records(true, true);
        slots_written_during_syncs_end_as_their_records(false, true);
    }

    /// A placed heap reused at the same address replaces the stale range;
    /// descriptor writes go only to the new heap.
    #[cfg(windows)]
    #[test]
    fn overlapping_placed_heaps_mirror_into_the_replacement() {
        let mut memory = vec![0u8; 2 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count: 2,
            flags: SHADER_VISIBLE,
            nodes: 0,
        };
        let placement = Placement {
            address,
            size: 2 * 32,
        };
        let old = owner(desc, placement);
        let new = owner(desc, placement);
        {
            let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
            assert_eq!(
                heaps
                    .iter()
                    .filter(|&&(start, end, _)| (start..end).contains(&address))
                    .count(),
                1
            );
            assert_eq!(heap_at(&heaps, address), Some(new.0 as usize));
        }

        create_cbv(std::ptr::null(), &[0x1000, 1, 256, 0], address);
        assert!(heap(old.0).pending.lock().unwrap().is_empty());
        assert_eq!(heap(new.0).pending.lock().unwrap().as_slice(), [0]);

        drop(old);
        create_cbv(std::ptr::null(), &[0x2000, 1, 256, 0], address + 32);
        assert_eq!(heap(new.0).pending.lock().unwrap().as_slice(), [0, 1]);
    }

    /// A creation that does not hand out a heap must not retire an existing
    /// registration for the same memory.
    #[cfg(windows)]
    #[test]
    fn failed_overlapping_creation_keeps_the_existing_heap() {
        let mut memory = vec![0u8; 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count: 1,
            flags: SHADER_VISIBLE,
            nodes: 0,
        };
        let placement = Placement { address, size: 32 };
        let existing = owner(desc, placement);
        let extended = Extended(desc, placement);
        let bad_iid = [0xff; 16];
        let mut out = std::ptr::null();
        assert_ne!(
            create_heap(std::ptr::null(), &extended.0, &bad_iid, &mut out),
            S_OK
        );
        assert!(out.is_null());
        assert_eq!(
            create_heap(
                std::ptr::null(),
                &extended.0,
                &HEAP_IIDS[0],
                std::ptr::null_mut()
            ),
            crate::object::S_FALSE
        );
        {
            let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
            assert_eq!(heap_at(&heaps, address), Some(existing.0 as usize));
        }
        create_cbv(std::ptr::null(), &[0x3000, 1, 256, 0], address);
        assert_eq!(heap(existing.0).pending.lock().unwrap().as_slice(), [0]);
    }

    /// the title places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        assert_eq!(
            create(
                desc,
                Placement {
                    address,
                    size: 64 * 32
                },
                &mut h
            ),
            S_OK
        );
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // Unregister the placed heap while its backing memory is still live.
        assert_eq!(slot::<extern "system" fn(This) -> u32>(h, 0x10)(h), 0);

        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(h, 0x10)(h), 0);
    }
}
