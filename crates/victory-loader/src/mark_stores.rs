//! Write marks at the source (`VICTORY_WRITE_MARKS=1`, read in
//! `victory-gpu`'s `write_marks`, #14): the title's own stores that fill
//! its draws' and dispatches' descriptor tables tell the layer which slots
//! of its shader-visible view heap they wrote, so the layer's heap sync
//! need not read every record of the pages the write watch says changed.
//!
//! What the title does (this build; image offsets): its shader-visible view
//! heap is the memory of its frame allocator, and two table fillers (one for
//! draws, one for dispatches) copy one 32-byte record at a time into a table
//! in three loops each: `and r, 7FE0h` (the record's byte offset in the
//! table), `vmovups ymm0, [record]`, `vmovups [table + r], ymm0`, then a
//! `movzx` and a `cmp`. The six stores were located by another fork's
//! "renderer port 2" (avatarmanz); the offsets below are this build's, and
//! the code here is our own. No bytes of the title are kept here: a site is
//! checked by its form ([`form_matches`]).
//!
//! Each store becomes a rel32 jump to a stub of ours near the image, which
//! does the store, then, when its address lies in the heap named by the
//! cells ([`CELLS_LEN`] bytes, `victory-gpu`'s `write_marks::Cells`), sets
//! the slot's byte in a map of slots and its 128-slot chunk's byte in a map
//! of chunks, and jumps back. Every register is put back. The stub
//! compares, so it changes the flags: where the bytes after the store are
//! shown to set every status flag before anything reads one
//! ([`flags_dead_after`]) the flags are left changed, elsewhere the stub
//! keeps them (`pushfq`/`popfq`, slower under Rosetta).
//!
//! A site's window (from the `and` through the `cmp`) is checked before
//! anything is written: the `and` of the offset with 7FE0h at its start, a
//! `vmovups` load after it, and a store of the known form at its offset. A
//! site whose window is not of that form (another build) is left alone and
//! logged, and `victory-gpu` then keeps today's scan of changed pages.

/// Bytes of the cells the stubs read: the heap's first byte (u64), its
/// length in bytes (u64; 0: nothing is marked), the map of slots (a byte a
/// 32-byte slot) and the map of chunks (a byte a [`CHUNK_SHIFT`] chunk).
pub const CELLS_LEN: usize = 32;

/// log2 of the bytes of a slot (32-byte view records).
pub const SLOT_SHIFT: u8 = 5;

/// log2 of the slots of a chunk (128).
pub const CHUNK_SHIFT: u8 = 7;

/// A store of the title's to patch.
#[derive(Debug, Clone, Copy)]
pub struct Site {
    /// What it is, for the log (role only).
    pub role: &'static str,
    /// Image offset of the first byte checked.
    pub window: usize,
    /// Offset of the store in the window, and its length.
    pub store: usize,
    pub len: usize,
    /// Bytes of the window: the `and`, the load, the store and what
    /// follows it through the `cmp`.
    pub span: usize,
}

impl Site {
    /// The store's bytes in `there` (the window's bytes).
    fn store_bytes<'a>(&self, there: &'a [u8]) -> Option<&'a [u8]> {
        there.get(self.store..self.store + self.len)
    }
    /// The bytes after the store in `there`.
    fn after<'a>(&self, there: &'a [u8]) -> &'a [u8] {
        there.get(self.store + self.len..).unwrap_or_default()
    }
}

/// The length of `and r/m32, 7FE0h` between registers (opcode 81h /4,
/// ModRM mod 3, an optional REX prefix) at the start of `code`.
fn offset_mask_len(code: &[u8]) -> Option<usize> {
    let skip = usize::from(matches!(code.first(), Some(0x40..=0x4f)));
    match code.get(skip..)? {
        [0x81, modrm, 0xe0, 0x7f, 0x00, 0x00, ..] if modrm >> 6 == 3 && modrm >> 3 & 7 == 4 => Some(skip + 6),
        _ => None,
    }
}

/// The length of a VEX `vmovups` (map 0F) with `opcode` (10h a load, 11h a
/// store) at the start of `code`, ModRM mod 0 with no SIB or a SIB with a
/// base, and no displacement. `None` for any other form.
fn vmovups_len(code: &[u8], opcode: u8) -> Option<usize> {
    let opcode_at = match code {
        [0xc5, ..] => 2,
        [0xc4, p1, ..] if p1 & 0x1f == 1 => 3,
        _ => return None,
    };
    let &[op, modrm, ..] = code.get(opcode_at..)? else {
        return None;
    };
    if op != opcode || modrm >> 6 != 0 || modrm & 7 == 5 {
        return None;
    }
    if modrm & 7 != 4 {
        return Some(opcode_at + 2);
    }
    let sib = *code.get(opcode_at + 2)?;
    (sib & 7 != 5).then_some(opcode_at + 3)
}

/// Whether `there` (the bytes at a site's window) is of the form the table
/// fillers' stores take: the `and` of the record's offset with 7FE0h first,
/// a `vmovups` load right after it, and at the site's offset a store whose
/// address [`store_address`] gives, `site.len` bytes long.
pub(crate) fn form_matches(site: &Site, there: &[u8]) -> bool {
    if there.len() != site.span || site.store + site.len > site.span {
        return false;
    }
    let Some(mask) = offset_mask_len(there) else {
        return false;
    };
    let Some(load) = vmovups_len(&there[mask..], 0x10) else {
        return false;
    };
    let store = &there[site.store..site.store + site.len];
    mask + load <= site.store && vmovups_len(store, 0x11) == Some(site.len) && store_address(store).is_some()
}

/// The six record stores of the table fillers of the build these offsets
/// are for. Each window: the `and` that masks the offset, the record's
/// load, the store, a `movzx` and a `cmp` (checked by [`form_matches`]).
pub const SITES: [Site; 6] = [
    Site {
        role: "draw table filler, loop 1",
        window: 0x4d9_f04f,
        store: 10,
        len: 5,
        span: 21,
    },
    Site {
        role: "draw table filler, loop 2",
        window: 0x4d9_f10d,
        store: 10,
        len: 5,
        span: 22,
    },
    Site {
        role: "draw table filler, loop 3",
        window: 0x4d9_f19c,
        store: 12,
        len: 6,
        span: 26,
    },
    Site {
        role: "dispatch table filler, loop 1",
        window: 0x4ee_c95d,
        store: 11,
        len: 6,
        span: 23,
    },
    Site {
        role: "dispatch table filler, loop 2",
        window: 0x4ee_ca08,
        store: 10,
        len: 6,
        span: 22,
    },
    Site {
        role: "dispatch table filler, loop 3",
        window: 0x4ee_ca82,
        store: 11,
        len: 6,
        span: 24,
    },
];

/// The address operand of a `vmovups [base + index], xmm/ymm` (VEX, map
/// 0F, opcode 11h, ModRM mod 0 with a SIB byte and no displacement) as the
/// REX prefix and SIB byte of a `lea` of it. `None` for any other form, a
/// base of `rsp` (the stub pushes before its `lea`) or no base, or no
/// index.
fn store_address(store: &[u8]) -> Option<(u8, u8)> {
    // (REX.X, REX.B, where the opcode is)
    let (x, b, opcode_at) = match store {
        [0xc5, ..] => (false, false, 2),
        [0xc4, p1, ..] => {
            // Map 0F only.
            if p1 & 0x1f != 1 {
                return None;
            }
            // The prefix holds X and B inverted.
            (p1 & 0x40 == 0, p1 & 0x20 == 0, 3)
        }
        _ => return None,
    };
    let &[opcode, modrm, sib] = store.get(opcode_at..)? else {
        return None;
    };
    if opcode != 0x11 || modrm >> 6 != 0 || modrm & 7 != 4 {
        return None;
    }
    let (index, base) = (sib >> 3 & 7 | u8::from(x) << 3, sib & 7 | u8::from(b) << 3);
    // Base 101 with mod 0: a displacement, no base. Index 100 (not r12): none.
    if sib & 7 == 5 || base == 4 || index == 4 {
        return None;
    }
    Some((0x48 | u8::from(x) << 1 | u8::from(b), sib))
}

/// Whether the bytes after a store set every status flag before any
/// instruction reads one or control leaves them: only the forms the table
/// fillers use are known, `movzx r32, byte [reg + disp8]` (no flags) and
/// `cmp r/m32, r32` between registers (writes CF, PF, AF, ZF, SF, OF).
/// Anything else: not shown, and the stub keeps the flags.
pub(crate) fn flags_dead_after(mut code: &[u8]) -> bool {
    let rex = |c: &[u8]| matches!(c.first(), Some(0x40..=0x4f));
    loop {
        let skip = usize::from(rex(code));
        match code.get(skip..) {
            // movzx r32, byte [r/m + disp8]: mod 1, no SIB.
            Some([0x0f, 0xb6, modrm, _disp, ..]) if modrm >> 6 == 1 && modrm & 7 != 4 => code = &code[skip + 4..],
            // cmp r/m32, r32 with mod 3.
            Some([0x39, modrm, ..]) if modrm >> 6 == 3 => return true,
            _ => return false,
        }
    }
}

/// A rel32 `jmp` at `from` to `to`, 5 bytes.
fn jmp(from: usize, to: usize) -> Option<[u8; 5]> {
    let code = crate::masked::jump(from as u64, 5, to as u64)?;
    code.try_into().ok()
}

/// The stub placed at `to` for `store` (the store's bytes) at `at`: the
/// store, the marks, a jump back to `at + store.len()`. `keep_flags`: the
/// flags are saved and put back.
pub(crate) fn stub(store: &[u8], at: usize, to: usize, cells: usize, keep_flags: bool) -> Option<Vec<u8>> {
    let (rex, sib) = store_address(store)?;
    let mut c = store.to_vec();
    if keep_flags {
        c.push(0x9c); // pushfq
    }
    c.extend([0x50, 0x51, 0x52]); // push rax ; push rcx ; push rdx
    c.extend([rex, 0x8d, 0x04, sib]); // lea rax, [base + index]
    c.extend([0x48, 0xba]); // mov rdx, cells
    c.extend((cells as u64).to_le_bytes());
    c.extend([0x48, 0x2b, 0x02]); // sub rax, [rdx]: the offset in the heap
    c.extend([0x48, 0x3b, 0x42, 0x08]); // cmp rax, [rdx + 8]
    let marks = [
        0x48, 0xc1, 0xe8, SLOT_SHIFT, // shr rax, 5: the slot
        0x48, 0x8b, 0x4a, 0x10, // mov rcx, [rdx + 16]
        0xc6, 0x04, 0x01, 0x01, // mov byte [rcx + rax], 1
        0x48, 0xc1, 0xe8, CHUNK_SHIFT, // shr rax, 7: the chunk
        0x48, 0x8b, 0x4a, 0x18, // mov rcx, [rdx + 24]
        0xc6, 0x04, 0x01, 0x01, // mov byte [rcx + rax], 1
    ];
    c.extend([0x73, marks.len() as u8]); // jae past the marks (unsigned: below the heap too)
    c.extend(marks);
    c.extend([0x5a, 0x59, 0x58]); // pop rdx ; pop rcx ; pop rax
    if keep_flags {
        c.push(0x9d); // popfq
    }
    let back = jmp(to + c.len(), at + store.len())?;
    c.extend(back);
    Some(c)
}

/// What became of a site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Patched; whether the stub leaves the flags changed (shown dead).
    Patched { flags_dead: bool },
    /// Its window is not of the form expected ([`form_matches`]): left
    /// alone.
    Differs,
    /// Outside the image, no room near the image, out of reach, or Windows
    /// refused.
    Failed(&'static str),
}

/// Bytes of the room for stubs.
const ROOM: usize = 64 << 10;

/// Room for code within a rel32 jump of every byte of `[lo, hi)`: below
/// `lo` first, then above `hi`, 1 MiB steps. Read-write until the caller
/// makes it executable.
#[cfg(windows)]
fn room_near(lo: usize, hi: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    const STEP: usize = 1 << 20;
    let reach = |place: usize| hi.max(place + ROOM) - lo.min(place) < (1usize << 31) - STEP;
    let below = (1..2048).filter_map(|k| lo.checked_sub(k * STEP));
    let above = (1..2048).filter_map(|k| (hi.next_multiple_of(STEP)).checked_add(k * STEP));
    for want in below.chain(above).filter(|&p| reach(p)) {
        // SAFETY: a fresh reservation at a hint; null when the place is taken.
        let got = unsafe { VirtualAlloc(want as *const _, ROOM, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        if got != 0 {
            if reach(got) {
                return Some(got);
            }
            // SAFETY: ours, just reserved.
            unsafe {
                windows_sys::Win32::System::Memory::VirtualFree(got as *mut _, 0, windows_sys::Win32::System::Memory::MEM_RELEASE)
            };
        }
    }
    None
}

/// Patches `sites` of the image of `size` bytes at `base`; the stubs read
/// the cells at `cells` ([`CELLS_LEN`] bytes). What became of each site, in
/// order, each also logged.
///
/// # Safety
/// The image stays mapped and none of its code has run yet; `cells` stays
/// valid as long as the image runs.
#[cfg(windows)]
pub unsafe fn patch(base: usize, size: usize, cells: usize, sites: &[Site]) -> Vec<Outcome> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // Checked first, all of them: no room is taken for a build with none.
    // The window's bytes are copied, so the patch of one site never changes
    // what another's check or stub reads.
    let checked: Vec<Result<Vec<u8>, Outcome>> = sites
        .iter()
        .map(|site| {
            let end = site.window.checked_add(site.span).filter(|&end| end <= size);
            if end.is_none() || site.store + site.len > site.span {
                return Err(Outcome::Failed("outside the image"));
            }
            // SAFETY: inside the mapped image (checked above).
            let there = unsafe { std::slice::from_raw_parts((base + site.window) as *const u8, site.span) };
            if !form_matches(site, there) {
                return Err(Outcome::Differs);
            }
            Ok(there.to_vec())
        })
        .collect();
    let room = checked.iter().any(Result::is_ok).then(|| room_near(base, base + size)).flatten();
    let mut used = 0;
    let mut out = Vec::new();
    for (site, checked) in sites.iter().zip(checked) {
        let outcome = checked.map_or_else(|outcome| outcome, |there| {
            let Some(room) = room else {
                return Outcome::Failed("no room for code near the image");
            };
            let at = base + site.window + site.store;
            let to = room + used;
            let flags_dead = flags_dead_after(site.after(&there));
            let store = site.store_bytes(&there).unwrap_or_default();
            let (Some(body), Some(jump)) = (
                stub(store, at, to, cells, !flags_dead),
                crate::masked::jump(at as u64, site.len, to as u64),
            ) else {
                return Outcome::Failed("out of reach");
            };
            if used + body.len() > ROOM {
                return Outcome::Failed("no room for code near the image");
            }
            let mut old = 0;
            // SAFETY: our room, which nothing runs yet; then the store, made
            // writable for the write and put back, which nothing runs yet.
            unsafe {
                std::ptr::copy_nonoverlapping(body.as_ptr(), to as *mut u8, body.len());
                if VirtualProtect(at as *const _, site.len, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                    return Outcome::Failed("Windows would not make it writable");
                }
                std::ptr::copy_nonoverlapping(jump.as_ptr(), at as *mut u8, jump.len());
                VirtualProtect(at as *const _, site.len, old, &mut old);
            }
            used = (used + body.len()).next_multiple_of(16);
            Outcome::Patched { flags_dead }
        });
        let offset = format_args!("{:#x}", site.window + site.store);
        match outcome {
            Outcome::Patched { flags_dead } => tracing::info!(
                site = %offset,
                role = site.role,
                flags_dead,
                "write marks: the title's descriptor record store patched to mark its slot (VICTORY_WRITE_MARKS)"
            ),
            Outcome::Differs => tracing::warn!(
                site = %offset,
                role = site.role,
                "write marks: store left alone, its window is not of the form expected (another build?); heap syncs keep reading changed pages"
            ),
            Outcome::Failed(why) => tracing::warn!(site = %offset, role = site.role, why, "write marks: store not patched"),
        }
        out.push(outcome);
    }
    if let Some(room) = room {
        let mut old = 0;
        // SAFETY: our room; then the whole instruction cache.
        unsafe {
            VirtualProtect(room as *const _, ROOM, PAGE_EXECUTE_READ, &mut old);
            FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic code, assembled here (registers and offsets of our choosing).
    /// `vmovups [rcx + rdx], ymm0`
    const STORE_RCX_RDX: [u8; 5] = [0xc5, 0xfc, 0x11, 0x04, 0x11];
    /// `vmovups [r11 + rbx], ymm1` (three-byte VEX, B set)
    const STORE_R11_RBX: [u8; 6] = [0xc4, 0xc1, 0x7c, 0x11, 0x0c, 0x1b];
    /// `vmovups [rdi + r10], ymm0` (three-byte VEX, X set)
    const STORE_RDI_R10: [u8; 6] = [0xc4, 0xa1, 0x7c, 0x11, 0x04, 0x17];

    #[test]
    fn stores_of_the_known_form_give_their_address() {
        assert_eq!(store_address(&STORE_RCX_RDX), Some((0x48, 0x11)));
        assert_eq!(store_address(&STORE_R11_RBX), Some((0x49, 0x1b)));
        assert_eq!(store_address(&STORE_RDI_R10), Some((0x4a, 0x17)));
        // A displacement (mod 1), a load (10h), rsp as base, no base, no index.
        assert_eq!(store_address(&[0xc5, 0xfc, 0x11, 0x44, 0x11, 0x08]), None);
        assert_eq!(store_address(&[0xc5, 0xfc, 0x10, 0x04, 0x11]), None);
        assert_eq!(store_address(&[0xc5, 0xfc, 0x11, 0x04, 0x14]), None);
        assert_eq!(store_address(&[0xc5, 0xfc, 0x11, 0x04, 0x15]), None);
        assert_eq!(store_address(&[0xc5, 0xfc, 0x11, 0x04, 0x21]), None);
        // Map 0F38, or no VEX.
        assert_eq!(store_address(&[0xc4, 0xe2, 0x7c, 0x11, 0x04, 0x11]), None);
        assert_eq!(store_address(&[0x0f, 0x11, 0x04, 0x11]), None);
    }

    /// Synthetic windows (our own registers): `and edx, 7FE0h` or
    /// `and r10d, 7FE0h`, `vmovups ymm0, [rax]` or `vmovups ymm1, [r8]`, a
    /// store, `movzx eax, byte [rsi + 3]`, `cmp ebx, eax`.
    fn window(rex_mask: bool, vex3_load: bool, store: &[u8]) -> (Vec<u8>, Site) {
        let mut w = if rex_mask { vec![0x41, 0x81, 0xe2] } else { vec![0x81, 0xe2] };
        w.extend([0xe0, 0x7f, 0x00, 0x00]);
        w.extend::<&[u8]>(if vex3_load { &[0xc4, 0xc1, 0x7c, 0x10, 0x08] } else { &[0xc5, 0xfc, 0x10, 0x00] });
        let at = w.len();
        w.extend(store);
        w.extend([0x0f, 0xb6, 0x46, 0x03, 0x39, 0xc3]);
        let site = Site { role: "test", window: 0, store: at, len: store.len(), span: w.len() };
        (w, site)
    }

    #[test]
    fn windows_of_the_known_form_match_and_others_do_not() {
        for (rex, vex3, store) in [
            (false, false, &STORE_RCX_RDX[..]),
            (true, true, &STORE_R11_RBX[..]),
            (true, false, &STORE_RDI_R10[..]),
        ] {
            let (w, site) = window(rex, vex3, store);
            assert!(form_matches(&site, &w), "{w:02x?}");
            assert!(flags_dead_after(site.after(&w)));
            assert_eq!(site.store_bytes(&w), Some(store));
            // Another mask, another operation, no load, a short window, the
            // store elsewhere, a store of another length.
            let mut other = w.clone();
            other[usize::from(rex) + 2] = 0xc0;
            assert!(!form_matches(&site, &other), "mask 7FC0h");
            let mut or = w.clone();
            or[usize::from(rex) + 1] = 0xca; // /1: or
            assert!(!form_matches(&site, &or), "or, not and");
            let mut no_load = w.clone();
            no_load[usize::from(rex) + 6] = 0x90;
            assert!(!form_matches(&site, &no_load), "no load");
            assert!(!form_matches(&site, &w[..w.len() - 1]), "short");
            assert!(!form_matches(&Site { store: site.store + 1, ..site }, &w), "store elsewhere");
            assert!(!form_matches(&Site { len: site.len + 1, ..site }, &w), "length");
        }
        // A store with a displacement is not of the known form.
        let (w, site) = window(false, false, &[0xc5, 0xfc, 0x11, 0x44, 0x11, 0x08]);
        assert!(!form_matches(&site, &w));
        // The sites' windows hold their stores.
        for site in &SITES {
            assert!(site.store + site.len <= site.span && (5..=6).contains(&site.len), "{}", site.role);
        }
    }

    #[test]
    fn flags_are_dead_only_where_a_cmp_sets_them_first() {
        // movzx eax, byte [rsi + 3] ; cmp ebx, eax
        assert!(flags_dead_after(&[0x0f, 0xb6, 0x46, 0x03, 0x39, 0xc3]));
        // movzx r10d, byte [r9 + 1] ; cmp r10d, r11d (REX on both)
        assert!(flags_dead_after(&[0x45, 0x0f, 0xb6, 0x51, 0x01, 0x45, 0x39, 0xda]));
        // cmp straight away
        assert!(flags_dead_after(&[0x39, 0xd8]));
        // setc al first: reads CF.
        assert!(!flags_dead_after(&[0x0f, 0x92, 0xc0, 0x39, 0xd8]));
        // movzx, then a jump that reads them.
        assert!(!flags_dead_after(&[0x0f, 0xb6, 0x46, 0x03, 0x72, 0x10]));
        // cmp with memory: not a form known here.
        assert!(!flags_dead_after(&[0x39, 0x06]));
        // Nothing after the store in the window.
        assert!(!flags_dead_after(&[]));
    }

    #[test]
    fn a_stub_does_the_store_then_marks_and_jumps_back() {
        let (at, to, cells) = (0x1_4000_0000usize, 0x1_3ff0_0000usize, 0x1122_3344_5566_7788usize);
        let code = stub(&STORE_RCX_RDX, at, to, cells, false).unwrap();
        assert_eq!(&code[..5], &STORE_RCX_RDX, "the store first");
        assert_eq!(&code[5..8], &[0x50, 0x51, 0x52]);
        assert_eq!(&code[8..12], &[0x48, 0x8d, 0x04, 0x11], "lea rax, [rcx + rdx]");
        assert_eq!(&code[14..22], &cells.to_le_bytes());
        let back = &code[code.len() - 5..];
        assert_eq!(back[0], 0xe9);
        let rel = i32::from_le_bytes(back[1..].try_into().unwrap());
        assert_eq!((to + code.len()) as i64 + i64::from(rel), (at + 5) as i64, "back past the store");
        let kept = stub(&STORE_RCX_RDX, at, to, cells, true).unwrap();
        assert_eq!(kept.len(), code.len() + 2);
        assert_eq!(kept[5], 0x9c, "pushfq");
        assert_eq!(kept[kept.len() - 6], 0x9d, "popfq");
        // Out of reach of a rel32 jump.
        assert!(stub(&STORE_RCX_RDX, at, at + (3 << 30), cells, false).is_none());
        assert!(stub(&[0xc5, 0xfc, 0x10, 0x04, 0x11], at, to, cells, false).is_none());
    }

    /// The marks `victory-gpu` keeps, laid out as the stubs read them.
    #[repr(C)]
    struct Cells {
        start: u64,
        len: u64,
        slots: *mut u8,
        chunks: *mut u8,
    }

    /// Synthetic code: `fn(table: rcx, offset: rdx, record: r8) -> rax`
    /// with `window` (a store of `ymm0` to `[rcx + rdx]` and what follows)
    /// at the returned offset, between a prologue and an epilogue.
    #[cfg(windows)]
    fn function(prologue: &[u8], window: &[u8], epilogue: &[u8]) -> (Vec<u8>, usize) {
        let mut code = vec![0xc4, 0xc1, 0x7c, 0x10, 0x00]; // vmovups ymm0, [r8]
        code.extend(prologue);
        let at = code.len();
        code.extend(window);
        code.extend(epilogue);
        code.extend([0xc5, 0xf8, 0x77, 0xc3]); // vzeroupper ; ret
        (code, at)
    }

    /// Two copies of synthetic code, one patched, both run: the record is
    /// stored alike, the registers the window leaves are the same, and the
    /// slot and its chunk are marked when the store is in the heap, not
    /// otherwise. With flags not shown dead after the store, the stub keeps
    /// them.
    #[cfg(windows)]
    #[test]
    fn a_patched_store_stores_marks_and_keeps_what_it_must() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};
        type F = unsafe extern "system" fn(*mut u8, u64, *const u8) -> u64;
        // and edx, 7FE0h ; vmovups ymm0, [r8]: the window's start.
        let start = [0x81, 0xe2, 0xe0, 0x7f, 0x00, 0x00, 0xc4, 0xc1, 0x7c, 0x10, 0x00];
        let mut window_dead = start.to_vec();
        window_dead.extend(STORE_RCX_RDX);
        window_dead.extend([0x41, 0x0f, 0xb6, 0x40, 0x01]); // movzx eax, byte [r8 + 1]
        window_dead.extend([0x39, 0xc0]); // cmp eax, eax
        let mut window_live = start.to_vec();
        window_live.push(0xf9); // stc: CF set before the store, read by setc after it
        window_live.extend(STORE_RCX_RDX);
        window_live.extend([0x0f, 0x92, 0xc0]); // setc al
        // Returns rax + rcx + rdx: what the window left in them.
        let sum = [0x48, 0x01, 0xc8, 0x48, 0x01, 0xd0]; // add rax, rcx ; add rax, rdx
        // xor eax, eax first where setc writes only al.
        let cases: [(&[u8], &[u8], usize, bool); 2] =
            [(&[], &window_dead, start.len(), true), (&[0x31, 0xc0], &window_live, start.len() + 1, false)];
        for (prologue, window, store, dead) in cases {
            let (code, at) = function(prologue, window, &sum);
            // SAFETY: fresh memory for code and data; one copy patched before it runs.
            unsafe {
                let image = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as usize;
                assert_ne!(image, 0);
                let (plain, patched) = (image + 0x100, image + 0x1100);
                std::ptr::copy_nonoverlapping(code.as_ptr(), plain as *mut u8, code.len());
                std::ptr::copy_nonoverlapping(code.as_ptr(), patched as *mut u8, code.len());
                let heap = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
                let (mut slots, mut chunks) = (vec![0u8; 0x8000 / 32], vec![0u8; 0x8000 / 32 / 128]);
                let cells = Box::new(Cells { start: heap as u64 + 0x2000, len: 0x8000, slots: slots.as_mut_ptr(), chunks: chunks.as_mut_ptr() });
                let site = Site { role: "test", window: 0x1100 + at, store, len: 5, span: window.len() };
                assert_eq!(patch(image, 0x10000, &raw const *cells as usize, &[site]), [Outcome::Patched { flags_dead: dead }]);
                assert_eq!(*((patched + at + store) as *const u8), 0xe9, "a jump at the store");
                let (plain, patched) = (std::mem::transmute::<usize, F>(plain), std::mem::transmute::<usize, F>(patched));
                let record: [u8; 32] = std::array::from_fn(|k| k as u8 ^ 0xa5);
                // Slot 0x80 + 5 of the heap: 0x80 slots in, then 0xa0 bytes.
                let table = heap.add(0x2000 + 0x80 * 32);
                let first = plain(table.add(0x4000), 0xa0, record.as_ptr());
                let second = patched(table, 0xa0, record.as_ptr());
                assert_eq!(first - 0x4000, second, "the registers the window leaves");
                if !dead {
                    assert_eq!(second & 0xff, (table as u64 + 0xa0 + 1) & 0xff, "CF kept: setc saw it");
                }
                assert_eq!(std::slice::from_raw_parts(table.add(0xa0), 32), &record);
                assert_eq!(std::slice::from_raw_parts(table.add(0x40a0), 32), &record);
                let slot = 0x80 + 5;
                assert_eq!(slots.iter().enumerate().filter(|(_, b)| **b != 0).map(|(i, _)| i).collect::<Vec<_>>(), [slot]);
                assert_eq!(chunks.iter().enumerate().filter(|(_, b)| **b != 0).map(|(i, _)| i).collect::<Vec<_>>(), [slot >> 7]);
                // Below the heap, and past it: stored, not marked.
                slots.fill(0);
                chunks.fill(0);
                patched(heap.add(0x1000), 0x20, record.as_ptr());
                patched(heap.add(0xa000), 0x20, record.as_ptr());
                assert_eq!(std::slice::from_raw_parts(heap.add(0x1020), 32), &record);
                assert!(slots.iter().chain(&chunks).all(|&b| b == 0));
                // No heap (length 0): nothing marked.
                let cells_at = &raw const *cells as *mut Cells;
                (*cells_at).len = 0;
                patched(table, 0xa0, record.as_ptr());
                assert!(slots.iter().chain(&chunks).all(|&b| b == 0));
                drop(cells);
            }
        }
    }

    /// A window not of the form expected is left as it is, and a site
    /// outside the image is not touched.
    #[cfg(windows)]
    #[test]
    fn a_site_whose_bytes_differ_is_left_alone() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};
        let (expected, site) = window(true, true, &STORE_R11_RBX);
        // SAFETY: fresh memory of our own.
        unsafe {
            let image = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
            let mut there = expected.clone();
            there[4] = 0xc0; // and r10d, 7FC0h
            std::ptr::copy_nonoverlapping(there.as_ptr(), image.add(0x200), there.len());
            let site = Site { window: 0x200, ..site };
            assert_eq!(patch(image as usize, 0x10000, 0, &[site]), [Outcome::Differs]);
            assert_eq!(std::slice::from_raw_parts(image.add(0x200), there.len()), &there[..]);
            let outside = Site { window: 0xfffc, ..site };
            assert!(matches!(patch(image as usize, 0x10000, 0, &[outside])[..], [Outcome::Failed(_)]));
        }
    }
}
