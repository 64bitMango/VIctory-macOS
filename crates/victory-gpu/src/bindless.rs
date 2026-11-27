//! Views shaders reach through the resource heap itself (decision 0007,
//! "Heap-indexed views"). the title is largely bindless: its
//! shaders load descriptors at indices they take from root constants or
//! memory (`HostShader::heap`), so no draw names the views it reads. Every
//! view the title writes into a shader-visible heap gets its host view when
//! it is written (`descriptor::host`), and each heap keeps here what its
//! slots view that title memory feeds: ranges of host buffer copies, and
//! textures whose contents the title writes. A submission whose lists run
//! such shaders first brings all of them up to date (what changed since,
//! `crate::watch`).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use crate::descriptor::View;
use crate::host::HostList;
use crate::object::This;
use crate::shadow::Shadow;

/// What a title heap slot views that title memory feeds.
#[derive(Clone)]
enum Seen {
    /// `len` bytes at `address` of a buffer's host copy; unordered access.
    Buffer(Arc<Shadow>, u64, u64, bool),
    /// A title texture (resource object) whose contents the title writes.
    Texture(usize),
    /// A view left without a host view (textures the title placed itself),
    /// by kind (diagnostics: `KINDS`), with a placed texture's address and
    /// console bytes.
    Left(#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))] (usize, u64, u64)),
}

/// Kinds of views left without a host view, for the census.
#[cfg(feature = "diagnostics")]
const KINDS: [&str; 4] = ["buffer", "unordered buffer", "placed texture", "other"];

/// A heap's slots that view title memory, and the distinct buffer ranges
/// and textures among them (with how many slots view each).
///
/// One mutex held all of it, taken for every slot made (135,000 a frame,
/// by the title's recording threads and the submitting thread) and held by
/// the submitting thread while it went through every copy's hull for each
/// list and every texture for each submission, and by a resource's release
/// while it went through every slot: once slots had locks of their own the
/// recording threads waited 265-357 ms a frame here, 22,400 times (run
/// ab-slots-on). In parts now, each under its own lock:
/// - What each slot views, by the slot's run (the runs the slots' locks go
///   by, `crate::descriptor::host`, so the thread making a slot finds its
///   part free).
/// - The counts, hulls and textures, by the copy or texture they are of.
///
/// What the one lock gave, and what gives it now:
/// - A slot's entry and the counts move together: both change with the
///   slot's part held ([`Viewed::put`]), so changes to one slot are counted
///   in the order they are entered, and each count is the number of slots
///   whose entry names its range or texture once the threads writing them
///   are done.
/// - A reader sees each copy's ranges and hull, and each texture's slots,
///   as they are at one moment (its part's lock), not all copies at one
///   moment: a slot made while a submission is prepared was in or out of
///   it by chance before too.
/// - A title texture found here is alive while the lock is held: its
///   release ([`forget`]) takes it out of its part before it goes, so
///   [`uploads`] looks at a part's textures with that part held.
///
/// Locked in the order: a slots part, then one counts part at a time.
/// `VICTORY_VIEWED_PARTS=0`: one part of each, and the submitting thread
/// holds off every slot while it reads, as the one lock did.
pub(crate) struct Viewed {
    slots: Box<[Part<crate::quick::Map<u32, Seen>>]>,
    counts: Box<[Part<Inner>]>,
    /// Whether it is kept in parts ([`in_parts`] when it was made): every
    /// thread finds a slot or a copy in the same part.
    parts: bool,
    /// Whether a copy's ranges are counted in pieces ([`Viewed::piece`];
    /// unless `VICTORY_VIEWED_PIECES=0`, or it is not in parts).
    pieces: bool,
}

/// Parts of each kind to a heap ([`Viewed`]).
const PARTS: usize = 64;

/// Slots in a row in one part (`crate::descriptor::host::RUN`).
const RUN: u32 = 64;

/// A part of [`Viewed`], on cache lines of its own; with how often it was
/// found taken since the last statistics line (diagnostics).
#[derive(Default)]
#[repr(align(64))]
struct Part<T>(Mutex<T>, #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))] std::sync::atomic::AtomicU32);

impl<T> Part<T> {
    fn lock(&self) -> std::sync::MutexGuard<'_, T> {
        #[cfg(feature = "diagnostics")]
        match self.0.try_lock() {
            Ok(part) => return part,
            Err(_) => self.1.fetch_add(1, Relaxed),
        };
        crate::contend::lock(&self.0, crate::contend::Lock::Viewed)
    }

    /// How often it was found taken since this was last asked.
    #[cfg(feature = "diagnostics")]
    fn waits(&self) -> u32 {
        self.1.swap(0, Relaxed)
    }
}

impl Default for Viewed {
    fn default() -> Self {
        Self {
            slots: (0..PARTS).map(|_| Part::default()).collect(),
            counts: (0..PARTS).map(|_| Part::default()).collect(),
            parts: in_parts(),
            pieces: in_parts() && in_pieces(),
        }
    }
}

/// Whether a copy's ranges are counted in pieces ([`Viewed::piece`]): unless
/// `VICTORY_VIEWED_PIECES=0`.
fn in_pieces() -> bool {
    #[cfg(test)]
    if tests::WHOLE_COPIES.with(|whole| whole.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_VIEWED_PIECES"))
}

/// Whether [`Viewed`] is kept in parts: unless `VICTORY_VIEWED_PARTS=0`.
fn in_parts() -> bool {
    #[cfg(test)]
    if tests::ONE_PART.with(|one| one.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_VIEWED_PARTS"))
}

type Range = (usize, u64, u64, bool);

/// A piece of a copy whose ranges are counted together: the copy, the
/// 64 KiB of title memory its ranges start in ([`Viewed::piece`]; 0 where
/// copies are not cut up), unordered access.
type Piece = (usize, u64, bool);

/// A merged range: copy, start, bytes, unordered access, the copy's bit in
/// the change tracker (`crate::watch::clean_bit`).
type Hull = (Arc<Shadow>, u64, u64, bool, Option<u32>);

#[derive(Default)]
struct Inner {
    ranges: BTreeMap<Range, (u32, Arc<Shadow>)>,
    /// The slots viewing each texture, and its host texture as last asked.
    textures: BTreeMap<usize, Texture>,
    /// `ranges` merged per copy and kind ([`Viewed::ranges`]), kept current
    /// as ranges come and go (each list running heap shaders asks for all
    /// of them; rebuilt whole after every change, they cost 13-27 ms a frame,
    /// run perf4).
    hulls: crate::quick::Map<Piece, Hull>,
    /// Per copy and kind (unordered access): the copy, and how many ranges
    /// start and end at each offset, so that the hulls are made from the
    /// copies, not from every range (the title rewrites slots between most
    /// lists, and remerging the world's 130,000 ranges each time cost 0.15
    /// s per frame).
    ends: crate::quick::Map<Piece, Ends>,
}

/// A title texture some slots view: the slots, and what a submission found
/// of it ([`uploads`]): its host texture and staging state, as of its count
/// of host textures made and given up (`crate::resource::hosts_changed`;
/// 0: not asked yet). Not held: a host texture given up goes when nothing
/// else holds it, whether or not a submission looks here again (a heap no
/// submission uses for a while would keep it, and its staging buffer,
/// alive); one that went is asked for again.
#[derive(Default)]
struct Texture {
    slots: crate::quick::Set<u32>,
    host: Option<std::sync::Weak<crate::resource::HostTexture>>,
    staging: Option<std::sync::Weak<crate::texture::Staging>>,
    asked: u64,
}

/// Whether [`uploads`] keeps what it found of a texture until its host
/// texture is made or given up: unless `VICTORY_UPLOADS_KEPT=0`, which asks
/// for every viewed texture at every submission.
fn uploads_kept() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_UPLOADS_KEPT"))
}

/// A copy, and the counts of its ranges' starts and ends.
type Ends = (Arc<Shadow>, BTreeMap<u64, u32>, BTreeMap<u64, u32>);

/// One more (`add`) or one fewer of `at` in `counts`.
fn tally(counts: &mut BTreeMap<u64, u32>, at: u64, add: bool) {
    let n = counts.entry(at).or_insert(0);
    *n = if add { *n + 1 } else { n.saturating_sub(1) };
    if *n == 0 {
        counts.remove(&at);
    }
}

impl Inner {
    /// Slot `index` views `seen` now (`add`) or no more; a buffer's range
    /// is counted in its copy's piece `piece`.
    fn count(&mut self, seen: &Seen, index: u32, add: bool, piece: u64) {
        let step = |n: &mut u32| if add { *n += 1 } else { *n = n.saturating_sub(1) };
        match seen {
            Seen::Buffer(shadow, address, len, uav) => {
                let key = (Arc::as_ptr(shadow) as usize, *address, address + len, *uav);
                let e = self.ranges.entry(key).or_insert_with(|| (0, shadow.clone()));
                let was = e.0;
                step(&mut e.0);
                let now = e.0;
                if now == 0 {
                    self.ranges.remove(&key);
                }
                // A range came or went: its copy's hull may change.
                if (was == 0) != (now == 0) {
                    let piece = (key.0, piece, key.3);
                    let ends = self
                        .ends
                        .entry(piece)
                        .or_insert_with(|| (shadow.clone(), BTreeMap::new(), BTreeMap::new()));
                    tally(&mut ends.1, key.1, now > 0);
                    tally(&mut ends.2, key.2, now > 0);
                    let bounds = match (ends.1.keys().next(), ends.2.keys().next_back()) {
                        (Some(&from), Some(&to)) => Some((from, to - from)),
                        _ => None,
                    };
                    match bounds {
                        // Written only where it changes: most ranges that
                        // come and go lie inside their copy's hull (the
                        // views of a draw's constants in a buffer every
                        // draw's are in), and every thread making such
                        // slots meets at that copy's part.
                        Some((from, len)) => {
                            let bit = shadow.dirty_bit();
                            match self.hulls.get_mut(&piece) {
                                Some(hull) if (hull.1, hull.2, hull.4) == (from, len, bit) => {}
                                Some(hull) => (hull.1, hull.2, hull.4) = (from, len, bit),
                                None => {
                                    self.hulls.insert(piece, (shadow.clone(), from, len, key.3, bit));
                                }
                            }
                        }
                        None => {
                            self.ends.remove(&piece);
                            self.hulls.remove(&piece);
                        }
                    }
                }
            }
            Seen::Texture(r) if add => {
                self.textures.entry(*r).or_default().slots.insert(index);
            }
            Seen::Texture(r) => {
                if let Some(texture) = self.textures.get_mut(r) {
                    texture.slots.remove(&index);
                    if texture.slots.is_empty() {
                        self.textures.remove(r);
                    }
                }
            }
            Seen::Left(_) => {}
        }
    }
}

impl Viewed {
    /// Title slot `index` now holds `view` (none: no view), made on the
    /// host (`made`) or left without a host view.
    pub(crate) fn set(&self, index: u32, view: Option<&View>, made: bool) {
        // Found before any lock is taken.
        self.put(index, view.and_then(|v| if made { seen(v) } else { Some(Seen::Left(left_kind(v))) }));
    }

    /// [`Viewed::set`] of a slot made from `view` (`made`: on the host),
    /// with the copy its buffer view was made on if that is what [`seen`]
    /// would find for it (`crate::descriptor::host::found_copy`).
    pub(crate) fn set_found(&self, index: u32, view: &View, made: bool, found: Option<(Arc<Shadow>, u64, u64)>) {
        match found_seen(view, made, found) {
            Some(seen) => self.put(index, Some(seen)),
            None => self.set(index, Some(view), made),
        }
    }

    /// Slot `index` views `seen` now (none: nothing title memory feeds).
    fn put(&self, index: u32, seen: Option<Seen>) {
        // Held until the counts are changed: another change to this slot
        // is counted after this one.
        let mut slots = self.slots[self.slots_part(index)].lock();
        let old = match &seen {
            Some(seen) => slots.insert(index, seen.clone()),
            None => slots.remove(&index),
        };
        // The same range or texture as before: no count changes.
        if let (Some(old), Some(new)) = (&old, &seen)
            && old.same(new)
        {
            return;
        }
        for (seen, add) in [(old, false), (seen, true)] {
            if let Some(seen) = seen
                && let Some((part, piece)) = self.counts_part(&seen)
            {
                self.counts[part].lock().count(&seen, index, add, piece);
            }
        }
    }

    /// The part slot `index`'s entry is in.
    fn slots_part(&self, index: u32) -> usize {
        if self.parts { (index / RUN) as usize % PARTS } else { 0 }
    }

    /// The part the counts of what `seen` names are in, if it is counted,
    /// and for a buffer's range the piece of its copy it is counted in.
    fn counts_part(&self, seen: &Seen) -> Option<(usize, u64)> {
        use std::hash::BuildHasher;
        match seen {
            Seen::Buffer(shadow, address, ..) => {
                let (copy, piece) = (Arc::as_ptr(shadow) as usize, self.piece(*address));
                let part = if self.parts { crate::quick::Build.hash_one((copy, piece)) as usize % PARTS } else { 0 };
                Some((part, piece))
            }
            Seen::Texture(r) => Some((self.part_of(*r), 0)),
            Seen::Left(_) => None,
        }
    }

    /// The piece of its copy a range starting at `address` is counted in:
    /// the 64 KiB it starts in, so that the ranges of one copy are in many
    /// parts (more than half the world's buffer views, 290,000 slots and
    /// 77,000 ranges, are of one buffer of 174 MB: every thread making such
    /// a slot met at that copy's part, 152,000 of 153,000 waits in a run,
    /// ab-new-off). A copy's hull is its pieces' hulls joined
    /// ([`Viewed::hulls`]). 0 where copies are not cut up.
    fn piece(&self, address: u64) -> u64 {
        if self.pieces { address >> 16 } else { 0 }
    }

    /// The counts part of the copy or texture at `address`.
    fn part_of(&self, address: usize) -> usize {
        use std::hash::BuildHasher;
        if self.parts { crate::quick::Build.hash_one(address) as usize % PARTS } else { 0 }
    }

    /// With one part of each: every slot held off until dropped, as a
    /// reader of the one lock held them off.
    fn hold(&self) -> Option<std::sync::MutexGuard<'_, crate::quick::Map<u32, Seen>>> {
        (!self.parts).then(|| self.slots[0].lock())
    }

    /// The host texture [`uploads`] last found for title texture
    /// `resource`, by address (tests).
    #[cfg(test)]
    pub(crate) fn known_host(&self, resource: usize) -> Option<usize> {
        let inner = self.counts[self.part_of(resource)].lock();
        inner.textures.get(&resource)?.host.as_ref().map(|host| host.as_ptr() as usize)
    }

    /// The count of title texture `resource` [`uploads`] last asked at
    /// (tests; 0: not asked).
    #[cfg(test)]
    pub(crate) fn known_asked(&self, resource: usize) -> u64 {
        let inner = self.counts[self.part_of(resource)].lock();
        inner.textures.get(&resource).map_or(0, |texture| texture.asked)
    }

    /// How many title slots hold a view (tests).
    #[cfg(test)]
    pub(crate) fn slots(&self) -> usize {
        self.slots.iter().map(|part| part.lock().len()).sum()
    }

    /// The buffer ranges views reach, merged per host copy and kind
    /// (shader resource, unordered access): copy, start, bytes.
    #[cfg(any(test, feature = "diagnostics"))]
    fn ranges(&self) -> Vec<Hull> {
        self.hulls(|_| true)
    }

    /// Each copy's hull per kind, of the copies `wanted` (asked by their
    /// tracker's bit): its pieces' hulls, each read under its part's lock,
    /// joined (first start to last end, as one count of all its ranges
    /// gives).
    fn hulls(&self, wanted: impl Fn(Option<u32>) -> bool) -> Vec<Hull> {
        let mut joined: crate::quick::Map<(usize, bool), Hull> = Default::default();
        for part in self.counts.iter() {
            let inner = part.lock();
            for (&(copy, _, uav), piece) in inner.hulls.iter().filter(|(_, hull)| wanted(hull.4)) {
                match joined.get_mut(&(copy, uav)) {
                    Some(hull) => {
                        let (from, to) = (hull.1.min(piece.1), (hull.1 + hull.2).max(piece.1 + piece.2));
                        (hull.1, hull.2) = (from, to - from);
                    }
                    None => {
                        joined.insert((copy, uav), piece.clone());
                    }
                }
            }
        }
        joined.into_values().collect()
    }
}

impl Seen {
    /// Whether both name the same range of the same copy, or the same
    /// texture.
    fn same(&self, other: &Seen) -> bool {
        match (self, other) {
            (Seen::Buffer(a, from, len, uav), Seen::Buffer(b, from_b, len_b, uav_b)) => {
                Arc::ptr_eq(a, b) && (from, len, uav) == (from_b, len_b, uav_b)
            }
            (Seen::Texture(a), Seen::Texture(b)) => a == b,
            _ => false,
        }
    }
}

/// What [`seen`] finds for buffer view `view`, made on the host (`made`)
/// on the copy `found`, from that copy; none to ask [`seen`].
fn found_seen(view: &View, made: bool, found: Option<(Arc<Shadow>, u64, u64)>) -> Option<Seen> {
    let (shadow, address, len) = found.filter(|_| made)?;
    if len == 0 {
        return None;
    }
    let uav = matches!(view, View::Unordered(..) | View::Placed(true, ..));
    let len = len.min(shadow.left(address));
    Some(Seen::Buffer(shadow, address, len, uav))
}

/// Whether [`found_seen`] gives what [`seen`] finds for `view` (tests).
#[cfg(test)]
pub(crate) fn found_as_seen(view: &View, found: Option<(Arc<Shadow>, u64, u64)>) -> bool {
    match (found_seen(view, true, found), seen(view)) {
        (Some(a), Some(b)) => a.same(&b),
        _ => false,
    }
}

/// What a made view reads that title memory feeds.
fn seen(view: &View) -> Option<Seen> {
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    if let View::Shader(resource, _) = view
        && let Some(texture) = crate::resource::made_copy(*resource as This)
    {
        return crate::texture::title_written(&texture.texture).then_some(Seen::Texture(*resource));
    }
    // The copy the host view reads.
    let (shadow, address, len) = match *view {
        View::Constant(address, size) => {
            let len = u64::from(size);
            ((len > 0).then(|| crate::shadow::find(address, len)).flatten()?, address, len)
        }
        _ => crate::descriptor::host::view_copy(view)?,
    };
    let uav = matches!(view, View::Unordered(..) | View::Placed(true, ..));
    if len == 0 {
        return None;
    }
    let len = len.min(shadow.left(address));
    Some(Seen::Buffer(shadow, address, len, uav))
}

fn left_kind(view: &View) -> (usize, u64, u64) {
    match view {
        View::Placed(_, resource, desc) if resource[0] != 1 => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            (2, address, crate::resource::placed_size(resource).unwrap_or(0))
        }
        View::Unordered(..) | View::Placed(true, ..) => (1, 0, 0),
        View::Shader(..) | View::Placed(false, ..) | View::Constant(..) => (0, 0, 0),
        _ => (3, 0, 0),
    }
}

/// At submission, before anything else of it: the textures the heaps of its
/// heap-indexing lists view, and `textures` (staging and host texture of
/// those its lists read through tables, `crate::texture::upload`), are
/// brought up to date, those whose title memory changed copied into their
/// host textures by a list of ours, which the submission runs first (and
/// leaves them readable by shaders): each at most once.
///
/// shortcut: a title copy into texture memory recorded in the same
/// submission shows from the next one on.
pub(crate) fn uploads(
    heaps: &[Arc<Viewed>],
    textures: &[(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)],
) -> Option<Uploads> {
    static LIST: Mutex<Holder> = Mutex::new(Holder(None));
    crate::unwaited::not_beside("C6", "the shared upload list is reset", String::new);
    let mut guard = LIST.lock().unwrap_or_else(|e| e.into_inner());
    crate::texture::recycle();
    match &guard.0 {
        // Closed by the last submission, which has run.
        Some(list) => list.reset(),
        // Made open.
        None => {
            let list = crate::host::host()?.new_list().ok()?;
            list.set_ours();
            guard.0 = Some(list);
        }
    }
    let list = guard.0.as_ref()?;
    // What to bring up: the textures lists read through tables, and those
    // the heaps view that shaders read lately. Collected under the heaps'
    // locks and brought up without them, several at a time: reading title
    // memory, detiling and making staging buffers took half the submitting
    // thread in a new shot, and title threads setting views waited on the
    // lock meanwhile (run evict5).
    let mut work: Vec<(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)> = Vec::new();
    let mut staged = crate::quick::Set::default();
    for (staging, copy) in textures {
        // A host copy wrote it in a submission made before this one (it
        // was not when the list was recorded).
        if copy.host_written.load(Relaxed) {
            continue;
        }
        if staged.insert(Arc::as_ptr(staging)) {
            work.push((staging.clone(), copy.clone()));
        }
    }
    let mut done = crate::quick::Set::default();
    for viewed in heaps {
        let _held = viewed.hold();
        // Each part's textures with its lock held: a title texture is
        // alive while it is in a part and the part is held (its release
        // takes it out first, `forget`).
        // What was found of a texture holds until its host texture is made or
        // given up (its count read before looking): a world submission's
        // heaps view some 2,200 textures, a few of them changed, and asking
        // each for its host texture and staging state at each of a frame's
        // twenty submissions was most of this function's time.
        for part in viewed.counts.iter() {
            let mut inner = part.lock();
            for (&r, known) in inner.textures.iter_mut() {
                if !done.insert(r) {
                    continue;
                }
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::UploadsLooked, 1);
                let now = crate::resource::hosts_changed(r as This);
                let mut found = known.host.as_ref().and_then(std::sync::Weak::upgrade);
                // Asked again also when the one found went (not expected
                // while the count holds: the title texture holds its host
                // texture until it is given up, which moves the count on).
                if known.asked != now || !uploads_kept() || (found.is_none() && known.host.is_some()) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::UploadsAsked, 1);
                    found = crate::resource::made_copy(r as This);
                    (known.host, known.staging, known.asked) = (found.as_ref().map(Arc::downgrade), None, now);
                }
                // Not made (given up, or never read): nothing to bring up.
                let Some(texture) = found else {
                    continue;
                };
                // Only what shaders read lately (the eviction pass's stamps):
                // the heaps view thousands of textures a frame never reads, and
                // bringing each up whenever its pages change re-uploaded about
                // 4 GB in 30-46 world frames (runs evict2, noevict1).
                // shortcut: a texture read again after a gap is brought up after
                // the submission that read it, which can see bytes one refresh
                // old (a slot the title rewrites is made fresh anyway); upload on
                // the mark itself if that shows.
                if !texture.used_recently() {
                    continue;
                }
                // Its contents are the host's since a host copy wrote it.
                if texture.host_written.load(Relaxed) {
                    continue;
                }
                // Its host texture's own, made on first use (and held by it).
                let mut staging = known.staging.as_ref().and_then(std::sync::Weak::upgrade);
                if staging.is_none() {
                    staging = crate::texture::staging(r as This, &texture.texture);
                    known.staging = staging.as_ref().map(Arc::downgrade);
                }
                if let Some(staging) = staging
                    && staged.insert(Arc::as_ptr(&staging))
                {
                    work.push((staging, texture));
                }
            }
        }
    }
    let mut recorded = false;
    // The textures copied into stay alive in `work` until recorded.
    let mut readable = Vec::new();
    for ((staging, copy), upload) in work.iter().zip(crate::texture::refresh_all(&work)) {
        if let Some(upload) = upload {
            readable.extend(crate::texture::record_upload(list, copy, staging, &upload));
            recorded = true;
            #[cfg(feature = "diagnostics")]
            UPLOADED.fetch_add(1, Relaxed);
        }
    }
    if !readable.is_empty() {
        // SAFETY: an open list; the barriers name live textures.
        unsafe { list.list.ResourceBarrier(&readable) };
    }
    list.close();
    // A list recording nothing is not run (its reset next time is fine).
    recorded.then_some(Uploads(guard))
}

/// The upload list, used under its lock only.
pub(crate) struct Holder(Option<HostList>);
// SAFETY: D3D12 objects are free-threaded; the list is used by one
// submission at a time, under the lock.
unsafe impl Send for Holder {}

/// The upload list of a submission, ours until it has run.
pub(crate) struct Uploads(std::sync::MutexGuard<'static, Holder>);

impl Uploads {
    pub(crate) fn list(&self) -> &HostList {
        self.0.0.as_ref().expect("made before it is handed out")
    }
}

/// Per submission: brings the buffer ranges a heap's views reach up to date
/// (in the submission's batch of refreshes): the pages the title changed
/// since their copies took them.
///
/// shortcut: what shaders write through the heap's unordered access views
/// stays in the host copies (later draws read it there) and does not go
/// back to title memory, so the CPU does not see it, and a page the title
/// also writes is copied over from title memory, losing the GPU's bytes
/// there. Copying back the whole unordered ranges after every submission
/// (about 1 GB of them in the world) cost seconds per frame; the upgrade
/// is shaders marking the pages they write (as `crate::memory` does).
///
/// Copies the title has not written since they took it are passed by: a
/// world submission's heaps reach ~9,000 copies, a few of them written.
pub(crate) fn refresh(viewed: &Viewed, mut add: impl FnMut(&Arc<Shadow>, usize, usize)) {
    let _held = viewed.hold();
    // The hulls of the copies written, read part by part; entered without
    // the parts' locks.
    for (shadow, address, len, ..) in &viewed.hulls(|bit| !crate::watch::clean_bit(bit)) {
        add(shadow, *address as usize, *len as usize);
    }
}

/// Of `copies` (by address), those a slot of some view heap views.
pub(crate) fn shown(copies: &std::collections::HashSet<usize>) -> std::collections::HashSet<usize> {
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let mut shown = std::collections::HashSet::new();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        // A copy's ranges are counted in pieces, in any part.
        for part in viewed.counts.iter() {
            shown.extend(part.lock().ends.keys().map(|piece| piece.0).filter(|copy| copies.contains(copy)));
        }
    }
    shown
}

/// Draws and dispatches run with heap-indexing shaders, those of them
/// writing through the heap, textures uploaded for heaps (diagnostics).
static HEAP_DRAWS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
#[cfg(feature = "diagnostics")]
static UPLOADED: AtomicU64 = AtomicU64::new(0);

/// A draw or dispatch runs a shader that indexes the heap (and writes
/// through it).
pub(crate) fn note_draw(writes: bool) {
    HEAP_DRAWS[0].fetch_add(1, Relaxed);
    if writes {
        HEAP_DRAWS[1].fetch_add(1, Relaxed);
    }
}

/// Every view heap's registry.
static HEAPS: Mutex<Vec<std::sync::Weak<Viewed>>> = Mutex::new(Vec::new());

pub(crate) fn register(viewed: &Arc<Viewed>) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    heaps.retain(|h| h.strong_count() > 0);
    heaps.push(Arc::downgrade(viewed));
}

/// The title released `resource`: slots viewing it as a texture view
/// nothing to upload any more (its memory may be gone).
pub(crate) fn forget(resource: usize) {
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        // The slots viewing it (kept with it: going through every slot of
        // the heap held the lock for all of them), each cleared as a slot
        // made null is, if it still views it. Until its part has it no
        // more: whoever holds the part then does not find it
        // (`uploads` looks at the textures it finds there).
        loop {
            let slots = viewed.counts[viewed.part_of(resource)].lock().textures.get(&resource).map(|t| t.slots.clone());
            let Some(slots) = slots else { break };
            for index in slots {
                let mut slots = viewed.slots[viewed.slots_part(index)].lock();
                if matches!(slots.get(&index), Some(Seen::Texture(r)) if *r == resource) {
                    slots.remove(&index);
                }
                // Counted out either way: a slot that views another thing
                // by now was counted out already, which changes nothing.
                viewed.counts[viewed.part_of(resource)].lock().count(&Seen::Texture(resource), index, false, 0);
            }
        }
    }
}

/// Logs, per frame over `frames`, the heap-indexed draws, and what the
/// view heaps' slots view (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64) {
    let [draws, writes] = [0, 1].map(|i| HEAP_DRAWS[i].swap(0, Relaxed) as f64 / frames);
    let uploaded = UPLOADED.swap(0, Relaxed) as f64 / frames;
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        let mut left = [0u32; KINDS.len()];
        let (mut buffers, mut textures) = (0u32, 0u32);
        // Placed textures left null: the largest at each address.
        let mut placed = std::collections::HashMap::new();
        for part in viewed.slots.iter() {
            for seen in part.lock().values() {
                match seen {
                    Seen::Buffer(..) => buffers += 1,
                    Seen::Texture(_) => textures += 1,
                    Seen::Left((k, address, bytes)) => {
                        left[*k] += 1;
                        if *k == 2 {
                            let b = placed.entry(*address).or_insert(0u64);
                            *b = (*b).max(*bytes);
                        }
                    }
                }
            }
        }
        let placed_mb = placed.values().sum::<u64>() as f64 / 1e6;
        let hulls = viewed.ranges();
        let uav_hulls: u64 = hulls.iter().filter(|r| r.3).map(|r| r.2).sum();
        let all_hulls: u64 = hulls.iter().map(|r| r.2).sum();
        // Ranges, their bytes, copies and textures, part by part (a copy's
        // ranges are all in one part).
        let (mut ranges, mut bytes, mut distinct) = (0usize, 0u64, 0usize);
        let mut copies = std::collections::BTreeSet::new();
        for part in viewed.counts.iter() {
            let inner = part.lock();
            ranges += inner.ranges.len();
            bytes += inner.ranges.keys().map(|&(_, a, b, _)| b - a).sum::<u64>();
            copies.extend(inner.ranges.keys().map(|k| k.0));
            distinct += inner.textures.len();
        }
        let copies = copies.len();
        // Which parts the threads meet at: the slots' parts, the counts'
        // parts, the busiest of those and its copy with most ranges.
        let slot_waits: u32 = viewed.slots.iter().map(Part::waits).sum();
        let count_waits: Vec<u32> = viewed.counts.iter().map(Part::waits).collect();
        // Heaps with no slot viewing anything have none (they read as a line
        // of zeros after the title heap's).
        if let Some((busiest, &most)) = count_waits.iter().enumerate().max_by_key(|&(_, &n)| n)
            && slot_waits + most > 0
        {
            let inner = viewed.counts[busiest].lock();
            let mut by_copy: BTreeMap<usize, (u32, u32)> = BTreeMap::new();
            for (range, (slots, _)) in inner.ranges.iter() {
                let copy = by_copy.entry(range.0).or_default();
                (copy.0, copy.1) = (copy.0 + 1, copy.1 + slots);
            }
            let top = by_copy.iter().max_by_key(|(_, n)| n.1).map(|(copy, n)| (*copy, *n));
            let top = top.and_then(|(copy, n)| inner.ranges.range((copy, 0, 0, false)..).next().map(|(_, (_, s))| (s.range(), n)));
            tracing::info!(
                "  what slots view, times a part was found taken per frame: slots' parts {:.0}, counts' parts {:.0} (the busiest {:.0}, with {} copies and {} textures; its copy with most slots {})",
                f64::from(slot_waits) / frames,
                f64::from(count_waits.iter().sum::<u32>()) / frames,
                f64::from(most) / frames,
                by_copy.len(),
                inner.textures.len(),
                top.map_or("none".into(), |((start, len), (ranges, slots))| format!("{start:#x}+{len:#x}: {ranges} ranges, {slots} slots")),
            );
        }
        let left: Vec<String> = KINDS.iter().zip(left).map(|(k, n)| format!("{k} {n}")).collect();
        let all = crate::shadow::all();
        let all_bytes: u64 = all.iter().map(|s| s.range().1).sum();
        tracing::info!(
            "heap views: {draws:.0} draws/frame index the heap ({writes:.0} writing through it), \
             {uploaded:.0} textures uploaded/frame; slots: {buffers} buffer views ({} ranges, \
             {:.1} MB, {copies} copies), {textures} title-written textures ({} distinct); \
             left null: {} (placed textures at {} addresses, {placed_mb:.0} MB); \
             host copies of title memory: {} ({:.1} MB); hulls {:.1} MB, unordered {:.1} MB",
            ranges,
            bytes as f64 / 1e6,
            distinct,
            left.join(", "),
            placed.len(),
            all.len(),
            all_bytes as f64 / 1e6,
            all_hulls as f64 / 1e6,
            uav_hulls as f64 / 1e6
        );
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::object::{HRESULT, S_OK};

    thread_local! {
        /// What heaps made on this thread have their slots view is kept in
        /// one part (as `VICTORY_VIEWED_PARTS=0`).
        pub(crate) static ONE_PART: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// What heaps made on this thread have their slots view counts each
        /// copy's ranges together (as `VICTORY_VIEWED_PIECES=0`).
        pub(crate) static WHOLE_COPIES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Threads enter and clear what the same slots view (ranges of a few
    /// copies, a few textures, views left null) while others read as a
    /// submission does: once they are done, the counts of ranges, the hulls
    /// and the textures' slots are those of a table one thread gave the
    /// entries the slots ended with; a texture released then has no slots.
    /// In parts and in one.
    #[test]
    fn what_slots_view_entered_from_several_threads_adds_up() {
        // Four buffers of 256 KiB: four pieces each, more where they are not
        // aligned.
        const BUFFER: u64 = 0x4_0000;
        let memory = vec![0u8; 4 * BUFFER as usize];
        let at = memory.as_ptr() as u64;
        let buffers: Vec<(usize, u64)> = (0..4).map(|k| (at as usize + k, at + BUFFER * k as u64)).collect();
        buffers.iter().for_each(|&(resource, start)| crate::shadow::register(resource, start, BUFFER));
        let copies: Vec<Arc<Shadow>> = buffers.iter().filter_map(|&(_, start)| crate::shadow::find(start, BUFFER)).collect();
        let done = || buffers.iter().for_each(|&(resource, start)| crate::shadow::unregister(resource, start));
        if copies.len() < 4 {
            return done(); // no host GPU here
        }
        const SLOTS: u32 = 256;
        // What a slot views, from a number: nothing, a range, a texture
        // (addresses no test's resource has), a view left null.
        let seen = |n: u64| -> Option<Seen> {
            let k = (n >> 8) as usize;
            match n % 8 {
                0 => None,
                1..=4 => {
                    let (copy, from) = (&copies[k % 4], 0x3f00 * (k as u64 / 4 % 16));
                    Some(Seen::Buffer(
                        copy.clone(),
                        copy.range().0 + from,
                        0x40 + 0x40 * (k as u64 / 64 % 4),
                        k.is_multiple_of(3),
                    ))
                }
                5 | 6 => Some(Seen::Texture(0x7e57_0000_0000 + 0x100 * (k % 6))),
                _ => Some(Seen::Left((2, 0, 0))),
            }
        };
        // In parts with copies in pieces, in one part, in parts with copies
        // whole.
        for (one, whole) in [(false, false), (true, false), (false, true)] {
            ONE_PART.with(|o| o.set(one));
            WHOLE_COPIES.with(|w| w.set(whole));
            let viewed = Arc::new(Viewed::default());
            register(&viewed);
            let stop = std::sync::atomic::AtomicBool::new(false);
            std::thread::scope(|scope| {
                let (stop, viewed, seen, copies) = (&stop, &viewed, &seen, &copies);
                scope.spawn(move || {
                    ONE_PART.with(|o| o.set(one));
                    while !stop.load(Relaxed) {
                        refresh(viewed, |_, _, _| {});
                        let _ = shown(&copies.iter().map(|c| Arc::as_ptr(c) as usize).collect());
                        let _ = viewed.ranges();
                    }
                });
                let writers: Vec<_> = (0..4u64)
                    .map(|t| {
                        scope.spawn(move || {
                            ONE_PART.with(|o| o.set(one));
                            let mut n = 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(t + 1);
                            for _ in 0..20_000 {
                                n = n.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                                // The same slots from every thread.
                                viewed.put(((n >> 40) % u64::from(SLOTS)) as u32, seen(n >> 16));
                            }
                        })
                    })
                    .collect();
                writers.into_iter().for_each(|w| w.join().expect("a writer"));
                stop.store(true, Relaxed);
            });
            // One thread enters what the slots ended with, each copy's
            // ranges counted together: a copy's pieces join to that hull.
            WHOLE_COPIES.with(|w| w.set(true));
            let alone = Viewed::default();
            WHOLE_COPIES.with(|w| w.set(false));
            for part in viewed.slots.iter() {
                for (&slot, seen) in part.lock().iter() {
                    alone.put(slot, Some(seen.clone()));
                }
            }
            let hulls = |v: &Viewed| {
                let mut hulls: Vec<(usize, u64, u64, bool)> =
                    v.ranges().iter().map(|h| (Arc::as_ptr(&h.0) as usize, h.1, h.2, h.3)).collect();
                hulls.sort_unstable();
                hulls
            };
            let textures = |v: &Viewed| {
                let mut all: Vec<(usize, Vec<u32>)> = Vec::new();
                for part in v.counts.iter() {
                    all.extend(part.lock().textures.iter().map(|(r, texture)| {
                        let mut slots: Vec<u32> = texture.slots.iter().copied().collect();
                        slots.sort_unstable();
                        (*r, slots)
                    }));
                }
                all.sort_unstable();
                all
            };
            let ranges = |v: &Viewed| {
                let mut all: Vec<(Range, u32)> = Vec::new();
                for part in v.counts.iter() {
                    all.extend(part.lock().ranges.iter().map(|(range, (n, _))| (*range, *n)));
                }
                all.sort_unstable();
                all
            };
            assert_eq!(viewed.slots(), alone.slots(), "one part: {one}, copies whole: {whole}");
            assert_eq!(ranges(&viewed), ranges(&alone), "one part: {one}, copies whole: {whole}");
            assert_eq!(hulls(&viewed), hulls(&alone), "one part: {one}, copies whole: {whole}");
            assert_eq!(textures(&viewed), textures(&alone), "one part: {one}, copies whole: {whole}");
            assert!(!hulls(&viewed).is_empty() && !textures(&viewed).is_empty());
            // A texture released: no slot views it, the others stay.
            let gone = 0x7e57_0000_0000usize;
            let before = viewed.slots();
            let its = textures(&viewed).iter().find(|(r, _)| *r == gone).map_or(0, |(_, slots)| slots.len());
            forget(gone);
            assert!(textures(&viewed).iter().all(|(r, _)| *r != gone));
            assert_eq!(viewed.slots(), before - its);
            ONE_PART.with(|o| o.set(false));
        }
        done();
    }
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::s;

    /// Reads slots 0 (raw buffer) and 2 (texture) of the heap and writes
    /// slot 1 (raw unordered buffer): through a table over the heap's
    /// start, the same slots a shader indexing the heap reads.
    const SOURCE: &str = r#"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
Texture2D<float4> tex : register(t1);
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint texel = (uint)round(tex.Load(int3(id.x % 4, 0, 0)).x * 255.0);
    dst.Store(id.x * 4, src.Load(id.x * 4) + texel);
}
"#;

    fn compile() -> (ID3D12RootSignature, ID3D12PipelineState) {
        use crate::compute::{Param, srv, uav};
        let params = &[Param::Table(&[srv(0, 1), uav(0, 1), srv(1, 1)])];
        crate::compute::pipeline(SOURCE, s!("heap"), 0, params).unwrap()
    }

    /// A title resource from a description in the console's layout (56
    /// bytes: dimension, alignment, width, height, depth and mips, format,
    /// samples, layout, flags).
    fn resource(dimension: u32, width: u64, height: u32, format: u32, layout: u32) -> This {
        let desc: [u32; 14] = [
            dimension, 0, 0, 0, width as u32, (width >> 32) as u32, height, 1 | 1 << 16, format, 1, 0,
            layout, 0, 0,
        ];
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    /// A title resource at `address` (title memory of its own), as the
    /// title places buffers and textures in its heaps.
    fn resource_at(address: usize, dimension: u32, width: u64, height: u32, format: u32, layout: u32) -> This {
        let desc: [u32; 14] = [
            dimension, 0, 0, 0, width as u32, (width >> 32) as u32, height, 1 | 1 << 16, format, 1, 0,
            layout, 0, 0,
        ];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made = crate::resource::create_at_address(
            std::ptr::null(),
            address as u64,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    const LINEAR: u32 = 0x100;

    /// A copy's hull is its live ranges' first start to last end, kept as
    /// slots come and go; a copy no slot views any more has none.
    #[test]
    fn hulls_follow_the_ranges() {
        let memory = vec![0u8; 0x1000];
        let at = memory.as_ptr() as u64;
        crate::shadow::register(at as usize, at, 0x1000);
        let Some(copy) = crate::shadow::find(at, 0x1000) else {
            crate::shadow::unregister(at as usize, at);
            return; // no host GPU here
        };
        let viewed = Viewed::default();
        let set = |slot: u32, range: Option<(u64, u64)>| {
            viewed.put(slot, range.map(|(from, len)| Seen::Buffer(copy.clone(), at + from, len, false)));
        };
        let hull = || viewed.ranges().iter().map(|h| (h.1 - at, h.2)).collect::<Vec<_>>();
        set(0, Some((0x100, 0x100)));
        set(1, Some((0x800, 0x200)));
        set(2, Some((0x100, 0x100)));
        assert_eq!(hull(), [(0x100, 0x900)]);
        set(1, None);
        assert_eq!(hull(), [(0x100, 0x100)], "the far range gone");
        set(0, None);
        assert_eq!(hull(), [(0x100, 0x100)], "one slot still views it");
        set(2, None);
        assert!(hull().is_empty());
        crate::shadow::unregister(at as usize, at);
    }

    /// The title writes a buffer and a texture, and views of them and of an
    /// output buffer into its heap; a shader reaching them only through the
    /// heap (no table entry names them) reads what the title wrote at
    /// submission, every round (the title rewrites both each time, as Bink
    /// rewrites its planes every frame). The output buffer is not read back
    /// into title memory (`refresh`'s shortcut): it is read from its host
    /// copy.
    #[test]
    fn heap_views_are_current() {
        let (src, dst) = (resource(1, 256, 1, 0, 1), resource(1, 256, 1, 0, 1));
        let tex = resource(3, 4, 1, 28, LINEAR);
        let write = |base: u32, texel: u8| {
            let (src_bytes, _) = crate::resource::buffer_bytes(src).unwrap();
            let (tex_bytes, tex_desc) = crate::resource::texture_memory(tex).unwrap();
            let size = crate::resource::texture_layout(&tex_desc).unwrap().planes[0].layout.size as usize;
            // SAFETY: the resources' own memory, 256 bytes and the texture's.
            unsafe {
                let words = std::slice::from_raw_parts_mut(src_bytes.cast::<u32>(), 64);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = base + i as u32;
                }
                std::ptr::write_bytes(tex_bytes.cast_mut(), texel, size);
            }
        };
        rounds(src, dst, tex, write);
    }

    /// The same with the buffer and the texture in one view of physical
    /// pages and the title writing them through another (the title fills its
    /// heaps through a pool view): every round must reach the shader.
    #[test]
    fn heap_views_follow_writes_through_another_mapping() {
        let (a, b) = crate::watch::tests::twin_views(2);
        let src = resource_at(a, 1, 256, 1, 0, 1);
        let tex = resource_at(a + PAGE_BYTES, 3, 4, 1, 28, LINEAR);
        let dst = resource(1, 256, 1, 0, 1);
        let (_, tex_desc) = crate::resource::texture_memory(tex).unwrap();
        let size = crate::resource::texture_layout(&tex_desc).unwrap().planes[0].layout.size as usize;
        let write = |base: u32, texel: u8| {
            // SAFETY: the second view of the buffer's and the texture's pages.
            unsafe {
                let words = std::slice::from_raw_parts_mut(b as *mut u32, 64);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = base + i as u32;
                }
                std::ptr::write_bytes((b + PAGE_BYTES) as *mut u8, texel, size);
            }
        };
        rounds(src, dst, tex, write);
    }

    const PAGE_BYTES: usize = crate::watch::PAGE;

    /// Three rounds of: record a dispatch reading slots 0 and 2 and writing
    /// slot 1, let the title `write` (after recording, before submitting),
    /// submit, check the output's host copy.
    fn rounds(src: This, dst: This, tex: This, write: impl Fn(u32, u8)) {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        // A shader-visible view heap of 8 slots, in memory of its own.
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        let slot = |i: u64| title_heap.start + 32 * i;
        // Raw views of 64 dwords: format R32_TYPELESS, buffer dimension,
        // the default mapping, then first element, count, stride, flags.
        let srv: [u32; 10] = [39, 1, 0x1688, 0, 0, 0, 64, 0, 1, 0];
        let uav: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_srv(null, src, &srv, slot(0));
        crate::descriptor::create_uav(null, dst, null, &uav, slot(1));
        crate::descriptor::create_srv(null, tex, std::ptr::null(), slot(2));
        let heap_host = title_heap.host().expect("a host view heap");
        let (root, pso) = compile();
        let list = host.new_list().unwrap();
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output's host copy");
        for (round, (base, texel)) in [(1000u32, 0x40u8), (5000, 0x10), (9000, 0x22)].into_iter().enumerate() {
            if round > 0 {
                list.reset();
            }
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(4, 1, 1);
            }
            list.defer_heap(&heap_host.viewed);
            list.close();
            // The title writes after recording, before submitting.
            write(base, texel);
            host.execute(&[&list]);
            let words = out.read_host(dst_bytes as u64, 64);
            for (i, &v) in words.iter().enumerate() {
                assert_eq!(v, base + i as u32 + u32::from(texel), "round {round}, dword {i}");
            }
        }
    }
}
