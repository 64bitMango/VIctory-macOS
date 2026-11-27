//! Fewer exceptions from the AMD-only instructions a title uses: how often
//! each kind is emulated ([`crate::crash`]'s handler), and the ones replaced
//! in the title's code so that they raise none.
//!
//! An exception that reaches a vectored handler under Wine and Rosetta costs
//! tens of microseconds, and titles run these instructions in loops:
//!
//! - `clzero` loops (the two forms the handler does whole, one exception
//!   each) are found when the image is mapped and made to jump to the same
//!   loop with plain stores ([`patch_clzero_loops`], `VICTORY_PATCH_CLZERO=0`:
//!   left to the handler).
//! - `monitorx` and `mwaitx`, wait hints the handler steps over, are replaced
//!   at their first exception by a jump over themselves ([`replace_wait`],
//!   `VICTORY_PATCH_WAITS=0`: emulated each time). Found by the exception and
//!   not by their three bytes, which other instructions' operands hold too.
//!
//! - `extrq` and `insertq` (SSE4a, `66`/`f2 0f 78/79`), emulated a whole
//!   exception each by the handler, are replaced when the image is mapped
//!   by a jump to the same work in plain instructions
//!   ([`patch_sse4a`], `VICTORY_PATCH_SSE4A=0`: emulated, as before). The
//!   immediate forms (six or seven bytes) and the register form with a REX
//!   (five) hold the jump; the plain four-byte register form is left to
//!   the handler.
//!
//! Diagnostics builds log the counts per frame ([`log_emulated`]).

/// `cmp rm, reg` between 64-bit registers (`REX.W 39 /r`): (`rm`, `reg`).
pub(crate) fn compare_registers(code: &[u8]) -> Option<(u8, u8)> {
    let &[rex @ 0x48..=0x4f, 0x39, modrm] = code else {
        return None;
    };
    (modrm >> 6 == 3).then_some((modrm & 7 | (rex & 1) << 3, (modrm >> 3) & 7 | (rex & 4) << 1))
}

/// A `clzero` loop as the handler knows it (`crash::clzero`): one of
///
/// ```text
/// top: clzero                 top: mov rax, a
///      add rax, 0x40               add a, 0x40
///      cmp rax, b                  clzero
///      jb  top                     cmp a, b
///                                  jb  top
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClzeroLoop {
    /// Where the loop starts and where the instruction after its `jb` is,
    /// as offsets in the code searched.
    pub top: usize,
    pub end: usize,
    /// The instructions ahead of the stores in a pass (`mov rax, a` and
    /// `add a, 0x40`; none in the first form) and after them (`add rax,
    /// 0x40` in the first form; then `cmp`), as the title has them.
    before: [u8; 7],
    before_len: usize,
    after: [u8; 7],
    after_len: usize,
    /// Where the instruction is whose first bytes the jump takes besides the
    /// loop's first instruction: nothing may jump there.
    pub cut: usize,
}

/// The `clzero` loop whose `clzero` is at `code[at]`, if it is one the
/// handler does whole and neither of its registers is `rsp` (the stores
/// that take the loop's place use the stack for a moment).
pub(crate) fn clzero_loop(code: &[u8], at: usize) -> Option<ClzeroLoop> {
    if code.get(at..at + 3)? != [0x0f, 0x01, 0xfc] {
        return None;
    }
    const RSP: u8 = 4;
    if let Some(&[0x48, 0x83, 0xc0, 0x40, c0, c1, c2, 0x72, 0xf4]) = code.get(at + 3..at + 12)
        && let Some((0, limit)) = compare_registers(&[c0, c1, c2])
        && limit != 0
        && limit != RSP
    {
        return Some(ClzeroLoop {
            top: at,
            end: at + 12,
            before: [0; 7],
            before_len: 0,
            after: [0x48, 0x83, 0xc0, 0x40, c0, c1, c2],
            after_len: 7,
            cut: at + 3,
        });
    }
    let before: [u8; 7] = code.get(at.checked_sub(7)?..at)?.try_into().ok()?;
    let [rex @ (0x48 | 0x4c), 0x89, load, add_rex @ (0x48 | 0x49), 0x83, add, 0x40] = before else {
        return None;
    };
    let &[c0, c1, c2, 0x72, 0xf1] = code.get(at + 3..at + 8)? else {
        return None;
    };
    let (a, b) = compare_registers(&[c0, c1, c2])?;
    let found = load & 0xc7 == 0xc0
        && a == (load >> 3) & 7 | (rex & 4) << 1
        && add & 0xf8 == 0xc0
        && a == add & 7 | (add_rex & 1) << 3
        && a != 0
        && b != 0
        && a != RSP
        && b != RSP;
    found.then_some(ClzeroLoop {
        top: at - 7,
        end: at + 8,
        before,
        before_len: 7,
        after: [c0, c1, c2, 0, 0, 0, 0],
        after_len: 3,
        cut: at - 4,
    })
}

/// Every [`clzero_loop`] of `code`, in order.
pub(crate) fn clzero_loops(code: &[u8]) -> Vec<ClzeroLoop> {
    let mut found = Vec::new();
    let mut at = 0;
    while at + 3 <= code.len() {
        if code[at] == 0x0f
            && let Some(l) = clzero_loop(code, at)
        {
            at = l.end;
            found.push(l);
        } else {
            at += 1;
        }
    }
    found
}

/// Whether bytes near `target` read as a short jump to it (`jcc`, `jmp`,
/// `loop`, `jrcxz` with an 8-bit offset). Bytes, not decoded instructions:
/// it says yes too often, never no wrongly for such a jump.
pub(crate) fn short_jump_to(code: &[u8], target: usize) -> bool {
    let from = target.saturating_sub(129);
    let to = (target + 127).min(code.len().saturating_sub(1));
    (from..to).any(|p| {
        matches!(code[p], 0x70..=0x7f | 0xeb | 0xe0..=0xe3)
            && (p as i64 + 2 + i64::from(code[p + 1] as i8)) == target as i64
    })
}

/// `jmp rel32` at `at` to `to`, if it reaches.
fn jump(at: u64, to: u64) -> Option<[u8; 5]> {
    let rel = i32::try_from(to as i64 - (at as i64 + 5)).ok()?;
    let [a, b, c, d] = rel.to_le_bytes();
    Some([0xe9, a, b, c, d])
}

/// The loop `l` (of code mapped at `code_at`) as code at `stub`, `clzero`
/// done by eight plain stores, ending with a jump to the instruction after
/// the loop. It leaves every register, the flags and memory as the loop
/// does, and so as the handler's emulation of it does:
///
/// ```text
/// pass: [mov rax, a; add a, 0x40]      the second form's, as the title has them
///       push rcx
///       mov  rcx, rax
///       and  rcx, -64                  clzero's line: the 64 bytes holding rax
///       mov  qword [rcx + 8k], 0       k = 0..8
///       pop  rcx                       before the compare, whose register it may be
///       [add rax, 0x40]                the first form's
///       cmp  .., b                     the title's own; the flags are its flags
///       jb   pass
///       jmp  the instruction after the loop
/// ```
///
/// (The eight bytes under the stack pointer are written, which a `clzero`
/// leaves alone; an exception writes far more there, and Windows code keeps
/// nothing below the stack pointer.)
pub(crate) fn clzero_stub(l: &ClzeroLoop, code_at: u64, stub: u64) -> Option<Vec<u8>> {
    let mut code = l.before[..l.before_len].to_vec();
    code.extend([0x51, 0x48, 0x89, 0xc1, 0x48, 0x83, 0xe1, 0xc0]);
    for k in 0..8u8 {
        code.extend([0x48, 0xc7, 0x41, k * 8, 0, 0, 0, 0]);
    }
    code.push(0x59);
    code.extend(&l.after[..l.after_len]);
    let back = i8::try_from(-(code.len() as i64 + 2)).ok()?;
    code.extend([0x72, back as u8]);
    let here = stub + code.len() as u64;
    code.extend(jump(here, code_at + l.end as u64)?);
    Some(code)
}

/// Whether `clzero` loops are replaced when the image is mapped: 0 not
/// (`VICTORY_PATCH_CLZERO=0`), 1 unless bytes nearby read as a jump to the
/// one instruction the replacement cuts in two ([`short_jump_to`]), 2 those
/// too (`VICTORY_PATCH_CLZERO=all`).
fn patches_clzero(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("0") => 0,
        Some("all") => 2,
        _ => 1,
    }
}

/// Whether wait instructions are replaced at their first exception; not
/// with `VICTORY_PATCH_WAITS=0`.
fn patches_waits_from(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

/// SSE4a's `extrq`/`insertq` where they sit, as the handler emulates them
/// ([`crate::crash::sse4a`]): one of
///
/// ```text
/// 66 0f 78 /0 ib ib   extrq xmm, len, at   the immediate form
/// f2 0f 78 /0 ib ib   insertq ...          (a REX may precede the 0f)
/// 66 0f 79 /r         extrq xmm, xmm      the register form
/// f2 0f 79 /r         insertq ...
/// ```
///
/// The immediate form's target is the register the modrm names (`src`),
/// `extrq`'s fields are the source's low qword and `insertq`'s its high,
/// and the register form works `extrq` on `dst` and `insertq` from `src`
/// into `dst`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sse4aSite {
    /// Where the instruction starts, as an offset in the code searched.
    pub at: usize,
    /// How long it is.
    pub len: usize,
    /// 0x66 (`extrq`) or 0xf2 (`insertq`).
    prefix: u8,
    /// The immediate form (opcode 0x78), or the register form (0x79).
    immediate: bool,
    /// What the modrm names: `insertq`'s destination and both forms' source;
    /// the immediate form's target.
    dst: u8,
    src: u8,
    /// The immediate form's bytes: the field's length and index.
    len_imm: u8,
    at_imm: u8,
}

/// The `extrq`/`insertq` at `code[at]`, if it is one of the four forms
/// ([`Sse4aSite`]); the modrm's register field must be 0 in the immediate
/// form, as only that is ever emitted, and both forms are register-to-
/// register.
pub(crate) fn sse4a_site(code: &[u8], at: usize) -> Option<Sse4aSite> {
    let &prefix = code.get(at)?;
    if !matches!(prefix, 0x66 | 0xf2) {
        return None;
    }
    let (rex, o) = if let Some(rex) = code.get(at + 1)
        && rex.wrapping_sub(0x40) < 0x10
    {
        (*rex, at + 2)
    } else {
        (0, at + 1)
    };
    if *code.get(o)? != 0x0f {
        return None;
    }
    let immediate = match code.get(o + 1)? {
        0x78 => true,
        0x79 => false,
        _ => return None,
    };
    let modrm = *code.get(o + 2)?;
    if modrm >> 6 != 3 || immediate && (modrm >> 3) & 7 != 0 {
        return None;
    }
    let dst = (modrm >> 3) & 7 | (rex & 4) << 1;
    let src = modrm & 7 | (rex & 1) << 3;
    let (len_imm, at_imm, end) = if immediate {
        (*code.get(o + 3)?, *code.get(o + 4)?, o + 5)
    } else {
        (0, 0, o + 3)
    };
    Some(Sse4aSite {
        at,
        len: end - at,
        prefix,
        immediate,
        dst,
        src,
        len_imm,
        at_imm,
    })
}

/// Every [`sse4a_site`] of `code`, in order.
pub(crate) fn sse4a_sites(code: &[u8]) -> Vec<Sse4aSite> {
    let mut found = Vec::new();
    let mut at = 0;
    while at < code.len() {
        if matches!(code[at], 0x66 | 0xf2)
            && let Some(s) = sse4a_site(code, at)
        {
            at += s.len;
            found.push(s);
        } else {
            at += 1;
        }
    }
    found
}

/// `movq rax, xmm`/`movq xmm, rax` (`66 48 0f 7e`/`6e`, a REX.R for the
/// high half of the registers).
fn xmm_rax(x: u8, from_xmm: bool, out: &mut Vec<u8>) {
    let rex = if x >= 8 { 0x4c } else { 0x48 };
    out.extend([0x66, rex, 0x0f, if from_xmm { 0x7e } else { 0x6e }, 0xc0 | (x & 7) << 3]);
}

/// `movdqu [rsp], xmm` (`f3 0f 7f 04 24`), both halves of the register
/// where the field lengths and indices come from.
fn xmm_to_stack(x: u8, out: &mut Vec<u8>) {
    if x >= 8 {
        out.extend([0xf3, 0x44, 0x0f, 0x7f, 0x04 | (x & 7) << 3, 0x24]);
    } else {
        out.extend([0xf3, 0x0f, 0x7f, 0x04 | (x & 7) << 3, 0x24]);
    }
}

/// The low qword of `rax` into the low half of `xmm` (`movq` zeroes the
/// upper one) with its high half as it was: the register saved under the
/// space the stub's `sub rsp` made, written, and its high half read back
/// (`movhpd`, the qword at `[rsp + 8]`).
fn rax_into_xmm_low(x: u8, out: &mut Vec<u8>) {
    xmm_to_stack(x, out);
    xmm_rax(x, false, out);
    out.push(0x66);
    if x >= 8 {
        out.push(0x44);
    }
    out.extend([0x0f, 0x16, 0x44 | (x & 7) << 3, 0x24, 0x08]);
}

/// The register form's fields: `rax`'s low word to `edx` (the field's
/// length) and `r9` (its index), and both masked to the six bits the
/// instruction reads; then the length's mask in `rdx`, through a shift of
/// all ones by `64 - bits` (a length of 0 means the whole qword, and
/// leaves the shift count masked to 0).
fn fields_and_mask(out: &mut Vec<u8>) {
    out.extend([
        0x0f, 0xb6, 0xd0, // movzx edx, al
        0x0f, 0xb6, 0xcc, // movzx ecx, ah (a REX would read SPL, not AH)
        0x83, 0xe2, 0x3f, // and edx, 63
        0x83, 0xe1, 0x3f, // and ecx, 63
        0x41, 0x89, 0xc9, // mov r9d, ecx
        0x41, 0x89, 0xd0, // mov r8d, edx
        0x41, 0xf7, 0xd8, // neg r8d
        0x49, 0x83, 0xc0, 0x40, // add r8, 64
        0x4c, 0x89, 0xc1, // mov rcx, r8
        0x48, 0xc7, 0xc2, 0xff, 0xff, 0xff, 0xff, // mov rdx, -1
        0x48, 0xd3, 0xea, // shr rdx, cl
    ]);
}

/// The site `s` (of code mapped at `code_at`) as code at `stub`: what
/// `extrq`/`insertq` does to its XMM registers done by `movq` and integer
/// instructions every CPU has, ending with a jump to the instruction
/// after it. It leaves every other register, the flags and memory as the
/// instruction does (it changes none but its target).
///
/// ```text
/// extrq, both forms, the field known (the immediate form) or read
/// (the register form) at run:
///   pushfq; push rax, rcx, rdx, r8, r9; sub rsp, 16
///   [mov rax, the length and index as one word]     the immediate form's
///   movdqu [rsp], the source; mov rax, [rsp]        extrq: the low half
///   or mov rax, [rsp + 8]                          insertq: the high
///   movzx edx, al; movzx ecx, ah; and both, 63; the index to r9
///   mov r8, 64 - the length; shr rdx (-1), r8      the mask
///   movq rax, the target; shr rax, r9; and rax, rdx
///   the target saved at [rsp], its low half written, its high half read
///   back (`movq` zeroes it; `movhpd` restores it, as the instruction and
///   the handler's emulation leave it)
///   add rsp, 16; pop the registers; popfq
///   jmp the instruction after the site
/// ```
///
/// `insertq` keeps its source's masked field and writes it into the
/// destination's: the same start, and `and rax, rdx; mov rcx, r9; shl
/// rax, rdx` and the destination and-ed by the mask's complement and
/// or-ed in.
pub(crate) fn sse4a_stub(s: &Sse4aSite, code_at: u64, stub: u64) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    out.extend([
        0x9c, // pushfq
        0x50, 0x51, 0x52, // push rax, rcx, rdx
        0x41, 0x50, 0x41, 0x51, // push r8, r9
        0x48, 0x83, 0xec, 0x10, // sub rsp, 16
    ]);
    if s.immediate {
        // The field as a constant: the length and index bytes as one word.
        out.extend([0x48, 0xb8]);
        out.extend([s.len_imm, s.at_imm, 0, 0, 0, 0, 0, 0]);
    } else {
        xmm_to_stack(s.src, &mut out);
        if s.prefix == 0x66 {
            out.extend([0x48, 0x8b, 0x04, 0x24]); // mov rax, [rsp]
        } else {
            out.extend([0x48, 0x8b, 0x44, 0x24, 0x08]); // mov rax, [rsp + 8]
        }
    }
    fields_and_mask(&mut out);
    if s.prefix == 0x66 {
        let target = if s.immediate { s.src } else { s.dst };
        out.extend([
            0x4c, 0x89, 0xc9, // mov rcx, r9
        ]);
        xmm_rax(target, true, &mut out);
        out.extend([
            0x48, 0xd3, 0xe8, // shr rax, cl
            0x48, 0x21, 0xd0, // and rax, rdx
        ]);
        rax_into_xmm_low(target, &mut out);
    } else {
        xmm_rax(s.src, true, &mut out);
        out.extend([
            0x48, 0x21, 0xd0, // and rax, rdx
            0x4c, 0x89, 0xc9, // mov rcx, r9
            0x48, 0xd3, 0xe0, // shl rax, cl
            0x48, 0xd3, 0xe2, // shl rdx, cl
            0x48, 0xf7, 0xd2, // not rdx
        ]);
        // movq rcx, the destination: the low qword the field goes into.
        out.extend([0x66]);
        if s.dst >= 8 {
            out.extend([0x4c]);
        } else {
            out.extend([0x48]);
        }
        out.extend([0x0f, 0x7e, 0xc0 | (s.dst & 7) << 3 | 1]);
        out.extend([
            0x48, 0x21, 0xd1, // and rcx, rdx
            0x48, 0x09, 0xc1, // or rcx, rax
        ]);
        // The destination's low half from rcx, its high half as it was.
        let dst = s.dst;
        xmm_to_stack(dst, &mut out);
        out.extend([0x66]);
        if dst >= 8 {
            out.extend([0x4c]);
        } else {
            out.extend([0x48]);
        }
        out.extend([0x0f, 0x6e, 0xc0 | (dst & 7) << 3 | 1]);
        out.push(0x66);
        if dst >= 8 {
            out.push(0x44);
        }
        out.extend([0x0f, 0x16, 0x44 | (dst & 7) << 3, 0x24, 0x08]);
    }
    out.extend([
        0x48, 0x83, 0xc4, 0x10, // add rsp, 16
        0x41, 0x59, 0x41, 0x58, // pop r9, r8
        0x5a, 0x59, 0x58, // pop rdx, rcx, rax
        0x9d, // popfq
    ]);
    let here = stub + out.len() as u64;
    out.extend(jump(here, code_at + s.at as u64 + s.len as u64)?);
    Some(out)
}

/// Whether `extrq`/`insertq` are replaced when the image is mapped; not
/// with `VICTORY_PATCH_SSE4A=0`.
fn patches_sse4a_from(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

#[cfg(windows)]
pub use on_windows::*;

#[cfg(windows)]
mod on_windows {
    use super::*;
    use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed, Ordering::SeqCst};
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE, VirtualAlloc,
        VirtualProtect,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    /// `IMAGE_SCN_MEM_EXECUTE`.
    const EXECUTE: u32 = 0x2000_0000;

    /// What the handler emulated, by kind (an exception each).
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Kind {
        /// `clzero`, one line.
        Clzero,
        /// A whole `clzero` loop in one exception.
        ClzeroLoop,
        Monitorx,
        Mwaitx,
        Rdpru,
        /// `clwb`, one instruction.
        Clwb,
        /// A whole `clwb` write-back loop in one exception.
        ClwbLoop,
        /// `extrq`, `insertq`.
        Sse4a,
        /// A wait instruction that was replaced before its exception was
        /// handled (or that the translator still runs as it was).
        Replaced,
    }
    const KINDS: usize = Kind::Replaced as usize + 1;

    static EMULATED: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
    /// Nanoseconds in the handler for the exceptions counted.
    #[cfg(feature = "diagnostics")]
    static HANDLER_NS: AtomicU64 = AtomicU64::new(0);
    /// `clzero` loops replaced at load, and wait instructions replaced at
    /// their first exception, so far.
    static LOOPS_REPLACED: AtomicU64 = AtomicU64::new(0);
    static WAITS_REPLACED: AtomicU64 = AtomicU64::new(0);
    /// `extrq`/`insertq` replaced at load, so far.
    static SSE4A_REPLACED: AtomicU64 = AtomicU64::new(0);

    /// The handler emulated one of `kind`.
    pub(crate) fn count(kind: Kind) {
        EMULATED[kind as usize].fetch_add(1, Relaxed);
    }

    /// How many of `kind` the handler emulated since the last
    /// [`log_emulated`].
    #[cfg(test)]
    pub(crate) fn counted(kind: Kind) -> u64 {
        EMULATED[kind as usize].load(Relaxed)
    }

    /// Adds the time since it was made to the handler's time when dropped.
    #[cfg(feature = "diagnostics")]
    pub(crate) struct Timed(pub(crate) std::time::Instant);

    #[cfg(feature = "diagnostics")]
    impl Drop for Timed {
        fn drop(&mut self) {
            HANDLER_NS.fetch_add(self.0.elapsed().as_nanos() as u64, Relaxed);
        }
    }

    /// Logs what the handler emulated since the last call, per frame over
    /// `frames` frames: every one of them was an exception. With the frame
    /// statistics (`victory_gpu::also_log_per_frame`).
    #[cfg(feature = "diagnostics")]
    pub fn log_emulated(frames: f64) {
        let n = |kind: Kind| EMULATED[kind as usize].swap(0, Relaxed) as f64 / frames.max(1.0);
        tracing::info!(
            "  AMD-only instructions emulated per frame (an exception each): clzero {:.1} single lines and {:.1} whole loops, \
             monitorx {:.1}, mwaitx {:.1}, rdpru {:.1}, clwb {:.1} single and {:.1} whole loops, extrq/insertq {:.1}, \
             waits already replaced {:.1}; {:.3} ms in their handler (the exception's own cost not counted); \
             replaced so far: {} clzero loops at load, {} extrq/insertq at load, {} wait instructions at their first exception",
            n(Kind::Clzero),
            n(Kind::ClzeroLoop),
            n(Kind::Monitorx),
            n(Kind::Mwaitx),
            n(Kind::Rdpru),
            n(Kind::Clwb),
            n(Kind::ClwbLoop),
            n(Kind::Sse4a),
            n(Kind::Replaced),
            HANDLER_NS.swap(0, Relaxed) as f64 / 1e6 / frames.max(1.0),
            LOOPS_REPLACED.load(Relaxed),
            SSE4A_REPLACED.load(Relaxed),
            WAITS_REPLACED.load(Relaxed),
        );
    }

    /// Whether `monitorx` and `mwaitx` are replaced at their first exception
    /// (`VICTORY_PATCH_WAITS=0`: emulated each time, as before).
    pub fn patches_waits() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| patches_waits_from(std::env::var("VICTORY_PATCH_WAITS").ok().as_deref()))
    }

    /// One thread at a time changes code: a second one would take the
    /// first one's loosened protection for the page's own.
    static PATCHING: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Code tests let [`replace_wait`] change: (start, end) each.
    #[cfg(test)]
    pub(crate) static TEST_CODE: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

    /// The first byte of what replaces a wait instruction: `jmp` over one
    /// byte (`eb 01`), which with the instruction's last byte left behind it
    /// is three bytes that do nothing.
    pub(crate) const JUMP_OVER: [u8; 2] = [0xeb, 0x01];

    /// Replaces the `monitorx` or `mwaitx` (`0f 01 fa`, `0f 01 fb`) at `at`,
    /// where it has just raised an exception in a title's code, by a jump
    /// over itself (`eb 01 fa`, `eb 01 fb`): whether this call did.
    ///
    /// What the handler does for these two is nothing to registers, flags
    /// or memory, and neither does the jump. The difference is in time: no
    /// exception, and for `mwaitx` no yield to another ready thread
    /// (`SwitchToThread`), so a loop waiting with it turns faster.
    ///
    /// Safe with other threads running the same bytes: one byte changes,
    /// in one store, so a thread fetches either the old instruction (it
    /// faults and is emulated, as ever) or the jump, never a mix, and no
    /// thread can be inside the instruction. A thread whose exception was
    /// raised by the old bytes and handled after the change is known by the
    /// new bytes at its instruction pointer (`crash::emulate_amd_only`).
    ///
    /// # Safety
    /// `at` is where an illegal-instruction exception was just raised, and
    /// its three bytes were read as one of the two instructions.
    pub(crate) unsafe fn replace_wait(at: usize) -> bool {
        if !patches_waits() {
            return false;
        }
        #[cfg(test)]
        let known = TEST_CODE.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|&(a, b)| (a..b).contains(&at));
        #[cfg(not(test))]
        let known = false;
        // Only a title's own code: never the host's, or memory that merely
        // holds the bytes.
        if !known && !crate::threads::in_title_image(at) {
            return false;
        }
        let _one = PATCHING.lock().unwrap_or_else(|e| e.into_inner());
        let mut old = 0;
        // SAFETY: mapped code (it just ran), writable for the one store and
        // put back; the store is of one byte, whole to every other thread.
        unsafe {
            if VirtualProtect(at as *const _, 1, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                return false;
            }
            let done = AtomicU8::from_ptr(at as *mut u8).compare_exchange(0x0f, JUMP_OVER[0], SeqCst, SeqCst).is_ok();
            VirtualProtect(at as *const _, 1, old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), at as *const _, 3);
            if done {
                WAITS_REPLACED.fetch_add(1, Relaxed);
            }
            done
        }
    }

    /// Makes the `clzero` loops in the executable sections of the image
    /// mapped at `base` jump to plain stores ([`clzero_stub`]), and returns
    /// how many, and how many it left to the handler because bytes nearby
    /// read as a jump into them ([`short_jump_to`]; none are left with
    /// `VICTORY_PATCH_CLZERO=all`). Nothing with `VICTORY_PATCH_CLZERO=0`.
    /// Before any title thread runs.
    ///
    /// The jump takes the loop's first five bytes: its first instruction
    /// and the start of its second. The rest stays, so code that enters the
    /// loop at its compare still finds it, and jumps back to the top; only
    /// a jump straight to that second instruction would land in the middle
    /// of ours, which is what is looked for.
    pub fn patch_clzero_loops(base: u64, info: &victory_pe::ImageInfo) -> (usize, usize) {
        let mode = patches_clzero(std::env::var("VICTORY_PATCH_CLZERO").ok().as_deref());
        if mode == 0 {
            return (0, 0);
        }
        let (mut patched, mut left) = (0, 0);
        let mut pool = None;
        for s in info.sections.iter().filter(|s| s.characteristics & EXECUTE != 0) {
            let start = base + u64::from(s.rva);
            let len = s.virtual_size.min(s.file_size) as usize;
            // SAFETY: the mapped section, before any title thread runs.
            let (p, l) = unsafe { patch_clzero_loops_in(start, len, base, mode == 2, &mut pool) };
            patched += p;
            left += l;
        }
        seal(pool);
        LOOPS_REPLACED.fetch_add(patched as u64, Relaxed);
        (patched, left)
    }

    /// [`patch_clzero_loops`] of the code `[start, start + len)`, its plain
    /// stores in `pool` (where it is and the bytes used; made near `base`
    /// when first needed); `every`: also the loops bytes nearby read as a
    /// jump into. Returns the loops replaced and left.
    ///
    /// # Safety
    /// The range is mapped code that no thread runs meanwhile.
    pub(crate) unsafe fn patch_clzero_loops_in(
        start: u64,
        len: usize,
        base: u64,
        every: bool,
        pool: &mut Option<(u64, usize)>,
    ) -> (usize, usize) {
        let (mut patched, mut left) = (0, 0);
        // SAFETY: mapped, readable code (the caller's).
        let code = unsafe { std::slice::from_raw_parts(start as *const u8, len) };
        for l in clzero_loops(code) {
            if !every && short_jump_to(code, l.cut) {
                left += 1;
                continue;
            }
            if pool.is_none() {
                *pool = near(base).map(|at| (at, 0));
            }
            let Some((at, used)) = pool.as_mut() else {
                tracing::warn!("no memory near the image for clzero loops: left to the handler");
                left += 1;
                continue;
            };
            let stub = *at + *used as u64;
            let site = start + l.top as u64;
            let (Some(body), Some(jump)) = (clzero_stub(&l, start, stub), jump(site, stub)) else {
                left += 1;
                continue;
            };
            if *used + body.len() > POOL {
                left += 1;
                continue;
            }
            let mut old = 0;
            // SAFETY: our pool has room for the body; the site is the
            // title's code, writable for the moment of the write, which no
            // thread runs.
            unsafe {
                std::ptr::copy_nonoverlapping(body.as_ptr(), stub as *mut u8, body.len());
                if VirtualProtect(site as *const _, jump.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                    left += 1;
                    continue;
                }
                std::ptr::copy_nonoverlapping(jump.as_ptr(), site as *mut u8, jump.len());
                VirtualProtect(site as *const _, jump.len(), old, &mut old);
            }
            *used += body.len().next_multiple_of(16);
            patched += 1;
        }
        (patched, left)
    }

    /// Makes the `extrq`/`insertq` in the executable sections of the image
    /// mapped at `base` jump to plain instructions ([`sse4a_stub`]), and
    /// returns how many, and how many it left to the handler: the register
    /// form without a REX is four bytes, too few for the jump that reaches
    /// the plain code. Nothing with `VICTORY_PATCH_SSE4A=0`. Before any
    /// title thread runs.
    ///
    /// The jump takes the site's first five bytes, always whole
    /// instructions: a jump straight to what follows them was already into
    /// the middle of the instruction, and a register form with a REX is
    /// exactly five. The immediate forms leave their own last one or two
    /// bytes behind the jump, unread.
    pub fn patch_sse4a(base: u64, info: &victory_pe::ImageInfo) -> (usize, usize) {
        if !patches_sse4a_from(std::env::var("VICTORY_PATCH_SSE4A").ok().as_deref()) {
            return (0, 0);
        }
        let (mut patched, mut left) = (0, 0);
        let mut pool = None;
        for s in info.sections.iter().filter(|s| s.characteristics & EXECUTE != 0) {
            let start = base + u64::from(s.rva);
            let len = s.virtual_size.min(s.file_size) as usize;
            // SAFETY: the mapped section, before any title thread runs.
            let (p, l) = unsafe { patch_sse4a_in(start, len, base, &mut pool) };
            patched += p;
            left += l;
        }
        seal(pool);
        SSE4A_REPLACED.fetch_add(patched as u64, Relaxed);
        (patched, left)
    }

    /// [`patch_sse4a`] of the code `[start, start + len)`, its plain code in
    /// `pool` (where it is and the bytes used; made near `base` when first
    /// needed). Returns the instructions replaced and left.
    ///
    /// # Safety
    /// The range is mapped code that no thread runs meanwhile.
    pub(crate) unsafe fn patch_sse4a_in(
        start: u64,
        len: usize,
        base: u64,
        pool: &mut Option<(u64, usize)>,
    ) -> (usize, usize) {
        let (mut patched, mut left) = (0, 0);
        // SAFETY: mapped, readable code (the caller's).
        let code = unsafe { std::slice::from_raw_parts(start as *const u8, len) };
        for s in sse4a_sites(code) {
            if s.len < 5 {
                left += 1;
                continue;
            }
            if pool.is_none() {
                *pool = near(base).map(|at| (at, 0));
            }
            let Some((at, used)) = pool.as_mut() else {
                tracing::warn!("no memory near the image for extrq/insertq: left to the handler");
                left += 1;
                continue;
            };
            let stub = *at + *used as u64;
            let site = start + s.at as u64;
            let (Some(body), Some(jump)) = (sse4a_stub(&s, start, stub), jump(site, stub)) else {
                left += 1;
                continue;
            };
            if *used + body.len() > POOL {
                left += 1;
                continue;
            }
            let mut old = 0;
            // SAFETY: our pool has room for the body; the site is the
            // title's code, writable for the moment of the write, which no
            // thread runs.
            unsafe {
                std::ptr::copy_nonoverlapping(body.as_ptr(), stub as *mut u8, body.len());
                if VirtualProtect(site as *const _, jump.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                    left += 1;
                    continue;
                }
                std::ptr::copy_nonoverlapping(jump.as_ptr(), site as *mut u8, jump.len());
                VirtualProtect(site as *const _, jump.len(), old, &mut old);
            }
            *used += body.len().next_multiple_of(16);
            patched += 1;
        }
        (patched, left)
    }

    /// The plain stores are written: their memory becomes code.
    pub(crate) fn seal(pool: Option<(u64, usize)>) {
        if let Some((at, _)) = pool {
            let mut old = 0;
            // SAFETY: our pool, written; then our process's whole cache.
            unsafe {
                VirtualProtect(at as *const _, POOL, PAGE_EXECUTE_READ, &mut old);
                FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
            }
        }
    }

    /// Bytes for the loops' plain stores: 64 KiB, some 600 loops.
    const POOL: usize = 64 << 10;

    /// Memory for code within reach of a `jmp rel32` from the image at
    /// `base`: below it, as the masked stores' (`crate::masked::patch`), or
    /// above it where nothing is free below (an image mapped low).
    fn near(base: u64) -> Option<u64> {
        let below = (1..2048u64).filter_map(|step| base.checked_sub(step << 20));
        let above = (1..2048u64).filter_map(|step| base.checked_add(step << 20));
        below.chain(above).find_map(|want| {
            // SAFETY: a plain allocation at a hint; failure returns null.
            let got = unsafe { VirtualAlloc(want as *const _, POOL, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
            (!got.is_null()).then_some(got as u64)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `clzero; add rax, 0x40; cmp rax, r10; jb`.
    const FIRST: [u8; 12] = [0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x4c, 0x39, 0xd0, 0x72, 0xf4];
    /// `mov rax, rdx; add rdx, 0x40; clzero; cmp rdx, r8; jb`.
    const SECOND: [u8; 15] =
        [0x48, 0x89, 0xd0, 0x48, 0x83, 0xc2, 0x40, 0x0f, 0x01, 0xfc, 0x4c, 0x39, 0xc2, 0x72, 0xf1];

    #[test]
    fn both_loops_are_found_with_their_ends() {
        let mut code = vec![0x90; 4];
        code.extend(FIRST);
        code.extend([0x90; 3]);
        code.extend(SECOND);
        code.push(0xc3);
        let found = clzero_loops(&code);
        assert_eq!(found.len(), 2);
        assert_eq!((found[0].top, found[0].end, found[0].cut), (4, 16, 7));
        assert_eq!((found[1].top, found[1].end, found[1].cut), (19, 34, 22));
        // A lone clzero, and the loops the handler does one line at a time:
        // against rax itself, with another step, with rax as the limit.
        assert!(clzero_loops(&[0x0f, 0x01, 0xfc, 0xc3]).is_empty());
        let mut own = FIRST;
        own[7..10].copy_from_slice(&[0x48, 0x39, 0xc0]);
        assert!(clzero_loops(&own).is_empty());
        let mut step = FIRST;
        step[6] = 0x20;
        assert!(clzero_loops(&step).is_empty());
        let mut limit = SECOND;
        limit[10..13].copy_from_slice(&[0x48, 0x39, 0xc2]);
        assert!(clzero_loops(&limit).is_empty());
        // Against the stack pointer: left to the handler.
        let mut stack = FIRST;
        stack[7..10].copy_from_slice(&[0x48, 0x39, 0xe0]);
        assert!(clzero_loops(&stack).is_empty());
        // Cut short.
        assert!(clzero_loops(&FIRST[..11]).is_empty());
        assert!(clzero_loops(&SECOND[7..]).is_empty());
    }

    #[test]
    fn the_stores_keep_the_titles_own_instructions() {
        let l = clzero_loop(&FIRST, 0).unwrap();
        let stub = clzero_stub(&l, 0x1_0000, 0x9000).unwrap();
        // push rcx; mov rcx, rax; and rcx, -64; eight stores; pop rcx.
        assert_eq!(stub[..8], [0x51, 0x48, 0x89, 0xc1, 0x48, 0x83, 0xe1, 0xc0]);
        assert_eq!(stub[8..16], [0x48, 0xc7, 0x41, 0, 0, 0, 0, 0]);
        assert_eq!(stub[64..72], [0x48, 0xc7, 0x41, 56, 0, 0, 0, 0]);
        assert_eq!(stub[72], 0x59);
        // add rax, 0x40; cmp rax, r10; jb to the start.
        assert_eq!(stub[73..80], FIRST[3..10]);
        assert_eq!(stub[80..82], [0x72, (-82i8) as u8]);
        // jmp to the instruction after the loop.
        assert_eq!(stub[82], 0xe9);
        let rel = i32::from_le_bytes(stub[83..87].try_into().unwrap());
        assert_eq!(0x9000 + 87 + i64::from(rel), 0x1_0000 + 12);
        assert_eq!(stub.len(), 87);
        // The second form: its mov and add first, its compare after.
        let l = clzero_loop(&SECOND, 7).unwrap();
        let stub = clzero_stub(&l, 0x1_0000, 0x9000).unwrap();
        assert_eq!(stub[..7], SECOND[..7]);
        assert_eq!(stub[7], 0x51);
        assert_eq!(stub[80..83], SECOND[10..13]);
        assert_eq!(stub[83..85], [0x72, (-85i8) as u8]);
        let rel = i32::from_le_bytes(stub[86..90].try_into().unwrap());
        assert_eq!(0x9000 + 90 + i64::from(rel), 0x1_0000 + 15);
        // Out of a jump's reach.
        assert!(clzero_stub(&l, 0x1_0000, 0x9000_0000_0000).is_none());
    }

    #[test]
    fn short_jumps_to_an_address_are_seen() {
        let mut code = vec![0x90; 64];
        assert!(!short_jump_to(&code, 30));
        // jne +8 at 20: to 30.
        code[20..22].copy_from_slice(&[0x75, 8]);
        assert!(short_jump_to(&code, 30));
        assert!(!short_jump_to(&code, 31));
        // jmp -12 at 40: to 30.
        code[20] = 0x90;
        code[40..42].copy_from_slice(&[0xeb, (-12i8) as u8]);
        assert!(short_jump_to(&code, 30));
        // The loops' own `jb` goes to their top, not to what the jump cuts.
        let first = clzero_loop(&FIRST, 0).unwrap();
        assert!(!short_jump_to(&FIRST, first.cut));
        let second = clzero_loop(&SECOND, 7).unwrap();
        assert!(!short_jump_to(&SECOND, second.cut));
    }

    #[test]
    fn the_switches_are_on_unless_zero() {
        assert_eq!(patches_clzero(None), 1);
        assert_eq!(patches_clzero(Some("1")), 1);
        assert_eq!(patches_clzero(Some("0")), 0);
        assert_eq!(patches_clzero(Some("all")), 2);
        assert!(patches_waits_from(None));
        assert!(patches_waits_from(Some("1")));
        assert!(!patches_waits_from(Some(" 0 ")));
        assert!(patches_sse4a_from(None));
        assert!(patches_sse4a_from(Some("1")));
        assert!(!patches_sse4a_from(Some(" 0 ")));
    }

    /// `extrq xmm, len, at` (the immediate form, target xmm6).
    const EXTRQ_IMM: [u8; 6] = [0x66, 0x0f, 0x78, 0xc6, 0x08, 0x14];
    /// `insertq xmm0, xmm1, len 16, at 16`.
    const INSERTQ_IMM: [u8; 6] = [0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10];
    /// `extrq xmm8, xmm1` (the register form with a REX).
    const EXTRQ_REG: [u8; 5] = [0x66, 0x44, 0x0f, 0x79, 0xc1];
    /// `insertq xmm0, xmm9`.
    const INSERTQ_REG: [u8; 5] = [0xf2, 0x41, 0x0f, 0x79, 0xc1];

    #[test]
    fn sse4a_sites_are_found_with_their_lengths() {
        let f = sse4a_site(&EXTRQ_IMM, 0).unwrap();
        assert_eq!(
            (f.len, f.prefix, f.immediate, f.src, f.dst),
            (6, 0x66, true, 6, 0)
        );
        let f = sse4a_site(&INSERTQ_IMM, 0).unwrap();
        assert_eq!(
            (f.len, f.prefix, f.immediate, f.src, f.dst),
            (6, 0xf2, true, 1, 0)
        );
        let f = sse4a_site(&EXTRQ_REG, 0).unwrap();
        assert_eq!(
            (f.len, f.prefix, f.immediate, f.src, f.dst),
            (5, 0x66, false, 1, 8)
        );
        let f = sse4a_site(&INSERTQ_REG, 0).unwrap();
        assert_eq!(
            (f.len, f.prefix, f.immediate, f.src, f.dst),
            (5, 0xf2, false, 9, 0)
        );
        // The register form without a REX: found (so it can be counted as
        // left), though too short to patch.
        let found = sse4a_site(&[0x66, 0x0f, 0x79, 0xc1], 0).unwrap();
        assert_eq!(found.len, 4);
        // The forms' own rules: the immediate form's register field is 0,
        // and both are register-to-register.
        assert!(sse4a_site(&[0x66, 0x0f, 0x78, 0xc9, 0x08, 0x14], 0).is_none());
        assert!(sse4a_site(&[0x66, 0x0f, 0x78, 0x06, 0x08, 0x14], 0).is_none());
        assert!(sse4a_site(&[0xf3, 0x0f, 0x79, 0xc1], 0).is_none());
        assert!(sse4a_site(&[0x66, 0x0f, 0x7a, 0xc1], 0).is_none());
    }

    #[test]
    fn every_sse4a_site_of_the_code_in_order() {
        let mut code = vec![0x90; 4];
        code.extend(EXTRQ_IMM);
        code.extend(INSERTQ_REG);
        code.push(0xc3);
        let found = sse4a_sites(&code);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].at, 4);
        assert_eq!(found[1].at, 10);
    }

    #[test]
    fn the_sse4a_stubs_shape() {
        // pushfq; push rax, rcx, rdx; push r8, r9; sub rsp, 16.
        let prologue = [
            0x9c, 0x50, 0x51, 0x52, 0x41, 0x50, 0x41, 0x51, 0x48, 0x83, 0xec, 0x10,
        ];
        // add rsp, 16; pop r9, r8, rdx, rcx, rax; popfq.
        let epilogue = [
            0x48, 0x83, 0xc4, 0x10, 0x41, 0x59, 0x41, 0x58, 0x5a, 0x59, 0x58, 0x9d,
        ];
        for site in [
            &EXTRQ_IMM[..],
            &INSERTQ_IMM[..],
            &EXTRQ_REG[..],
            &INSERTQ_REG[..],
        ] {
            let s = sse4a_site(site, 0).unwrap();
            let stub = sse4a_stub(&s, 0x1_0000, 0x9000).unwrap();
            assert_eq!(stub[..prologue.len()], prologue, "{site:02x?}");
            let back = stub.len() - 5;
            assert_eq!(stub[back - epilogue.len()..back], epilogue, "{site:02x?}");
            assert_eq!(stub[back], 0xe9, "{site:02x?}");
            let rel = i32::from_le_bytes(stub[back + 1..].try_into().unwrap());
            let to = 0x9000 + back as i64 + 5 + i64::from(rel);
            assert_eq!(to, 0x1_0000 + s.len as i64, "{site:02x?}");
            // The immediate form's field as one word.
            if s.immediate {
                assert_eq!(stub[12..14], [0x48, 0xb8], "{site:02x?}");
                assert_eq!(&stub[14..16], &[s.len_imm, s.at_imm], "{site:02x?}");
            }
        }
        // Out of a jump's reach.
        let s = sse4a_site(&EXTRQ_IMM, 0).unwrap();
        assert!(sse4a_stub(&s, 0x1_0000, 0x9000_0000_0000).is_none());
    }
}
