//! Scalar (SGPR) operations: SOP1/SOP2/SOPC instructions, SCC, quad masks
//! and carry chains, and the raw dword reads a view's size needs. Methods
//! of [`super::Lower`] called from its instruction dispatch.

use super::*;

impl<'a> Lower<'a> {
    pub(super) fn sop(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Sop { dst, src, imm } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let scc = inst.write(Reg::Scc);
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        if inst.write(Reg::S(EXEC)).is_some() || self.is_mask(lo) {
            // The SCC of a mask operation whose result is a register (not
            // EXEC, whose tests the branches take): "some bit is set", which
            // code may read as a scalar (a poll's "some lane is not ready",
            // selected into an all-or-none mask). Resolved where it is read.
            let width = if self.x.native.wave64 { "_b64" } else { "_b32" };
            let nonzero = name
                .strip_prefix("s_")
                .and_then(|n| n.strip_suffix(width))
                .is_some_and(|op| {
                    matches!(
                        op,
                        "and" | "or" | "xor" | "andn2" | "orn2" | "nand" | "nor" | "xnor" | "not"
                    )
                });
            if nonzero
                && inst.write(Reg::S(EXEC)).is_none()
                && let (Some(scc), Some(lo), Some(code)) = (scc, lo, dst_code)
            {
                let hi = if self.x.native.wave64 {
                    Some(inst.write(Reg::S(code + 1)).ok_or(LANE_MASK)?)
                } else {
                    None
                };
                self.mask_sccs.insert(scc, (lo, hi));
            }
            return Err(LANE_MASK);
        }
        if matches!(
            name,
            "s_setreg_imm32_b32" | "s_setreg_b32" | "s_swappc_b64" | "s_version"
        ) || name.starts_with("s_waitcnt")
        {
            // shortcut: the float MODE register is not modelled; s_swappc_b64 is the
            // vertex fetch call, whose results arrive as inputs.
            return Err("a register the emitter does not model");
        }
        let c = self.cur;
        let src32 = |l: &mut Self, k: usize| -> Result<V, Why> {
            // SOPK arithmetic reads its destination as the first source.
            if k == 0 && matches!(name, "s_addk_i32" | "s_mulk_i32") {
                let d = dst.as_ref().ok_or("a missing operand")?;
                return l.operand(inst, d);
            }
            let op = src.get(k).ok_or("a missing operand")?;
            l.operand(inst, op)
        };
        let sext16 = |l: &mut Self| l.s.m.i32(i32::from(imm.unwrap_or(0) as i16) as u32);
        // Dword `h` of 64-bit source `k` (inline constants sign-extend).
        let half = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                    l.i32_of(v)
                }
                Some(op) if h == 0 => l.operand(inst, op),
                Some(Operand::Const(k)) => {
                    let x = crate::resource::inline(*k).ok_or("a special operand")?;
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                // Of a signed operand, a literal sign-extends too.
                Some(&Operand::Literal(x)) if name.ends_with("_i64") => {
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                _ => Ok(l.s.m.i32(0)),
            }
        };
        if let Some(op) = name.strip_prefix("s_bitcmp") {
            // SCC is whether bit src1 (mod 32, or mod 64 for the 64-bit form)
            // of src0 is clear (bitcmp0) or set (bitcmp1).
            let (bit, k31) = (src32(self, 1)?, self.s.m.i32(31));
            let idx = self.s.m.bin(c, Bin::And, bit, k31);
            let word = if op.ends_with("_b64") {
                let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                let k32 = self.s.m.i32(32);
                let upper = self.s.m.bin(c, Bin::And, bit, k32);
                let z = self.s.m.i32(0);
                let upper = self.s.m.cmp(c, pred::NE, upper, z);
                self.s.m.select(c, upper, b, a)
            } else {
                src32(self, 0)?
            };
            let shifted = self.s.m.bin(c, Bin::LShr, word, idx);
            let one = self.s.m.i32(1);
            let tested = self.s.m.bin(c, Bin::And, shifted, one);
            let want = self.s.m.i32(u32::from(op.starts_with('1')));
            let r = self.s.m.cmp(c, pred::EQ, tested, want);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").filter(|o| o.ends_with("_u64")) {
            // eq and lg only: both dwords.
            let mut r = None;
            for h in 0..2 {
                let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                let e = self.s.m.cmp(c, pred::EQ, a, b);
                r = Some(match r {
                    Some(x) => self.s.m.bin(c, Bin::And, x, e),
                    None => e,
                });
            }
            let mut r = r.expect("two dwords");
            if op.starts_with("lg") {
                let t = self.s.m.bool(true);
                r = self.s.m.bin(c, Bin::Xor, r, t);
            }
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
            let (a, b) = if name.starts_with("s_cmpk_") {
                let b = if op.ends_with("_i32") {
                    sext16(self)
                } else {
                    self.s.m.i32(u32::from(imm.unwrap_or(0)))
                };
                (src32(self, 0)?, b)
            } else {
                if op.ends_with("64") {
                    return Err("a 64-bit scalar compare");
                }
                (src32(self, 0)?, src32(self, 1)?)
            };
            let signed = op.ends_with("_i32");
            let p = match &op[..2] {
                "eq" => pred::EQ,
                "lg" => pred::NE,
                "gt" if signed => pred::SGT,
                "ge" if signed => pred::SGE,
                "lt" if signed => pred::SLT,
                "le" if signed => pred::SLE,
                "gt" => pred::IUGT,
                "ge" => pred::IUGE,
                "lt" => pred::IULT,
                "le" => pred::IULE,
                _ => return Err("a scalar compare"),
            };
            let r = self.s.m.cmp(c, p, a, b);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        if wide {
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            let nonzero = |l: &mut Self, a: V, b: V| {
                let o = l.s.m.bin(c, Bin::Or, a, b);
                let z = l.s.m.i32(0);
                l.s.m.cmp(c, pred::NE, o, z)
            };
            return match name {
                "s_quadmask_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let low = self.quad_mask(a);
                    let high = self.quad_mask(b);
                    let k8 = self.s.m.i32(8);
                    let high = self.s.m.bin(c, Bin::Shl, high, k8);
                    let mask = self.s.m.bin(c, Bin::Or, low, high);
                    let zero = self.s.m.i32(0);
                    self.set(lo, Val::I32(mask));
                    self.set(hi, Val::I32(zero));
                    let nonzero = self.s.m.cmp(c, pred::NE, mask, zero);
                    self.set(scc, Val::Bit(nonzero));
                    Ok(())
                }
                "s_bcnt1_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let na = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[a]);
                    let nb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[b]);
                    let r = self.s.m.bin(c, Bin::Add, na, nb);
                    self.set(lo, Val::I32(r));
                    let z = self.s.m.i32(0);
                    let s = self.s.m.cmp(c, pred::NE, r, z);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_ff1_i32_b64" => {
                    // The lowest set bit; -1 for none (as DXIL's first bit low).
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let fa = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[a]);
                    let fb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[b]);
                    let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                    let fb32 = self.s.m.bin(c, Bin::Add, fb, k32);
                    let none = self.s.m.cmp(c, pred::EQ, b, z);
                    let fb = self.s.m.select(c, none, fb, fb32);
                    let low = self.s.m.cmp(c, pred::NE, a, z);
                    let r = self.s.m.select(c, low, fa, fb);
                    self.set(lo, Val::I32(r));
                    Ok(())
                }
                "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
                | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" | "s_not_b64" => {
                    let ones = self.s.m.i32(u32::MAX);
                    let mut r = Vec::new();
                    for h in 0..2u8 {
                        let a = half(self, 0, h)?;
                        let x = if name == "s_not_b64" {
                            self.s.m.bin(c, Bin::Xor, a, ones)
                        } else {
                            let mut b = half(self, 1, h)?;
                            if name.contains("n2") {
                                b = self.s.m.bin(c, Bin::Xor, b, ones);
                            }
                            let op = match &name[2..5] {
                                "and" | "nan" => Bin::And,
                                "or_" | "orn" | "nor" => Bin::Or,
                                _ => Bin::Xor,
                            };
                            let x = self.s.m.bin(c, op, a, b);
                            if matches!(name, "s_nand_b64" | "s_nor_b64" | "s_xnor_b64") {
                                self.s.m.bin(c, Bin::Xor, x, ones)
                            } else {
                                x
                            }
                        };
                        r.push(x);
                    }
                    self.set(lo, Val::I32(r[0]));
                    self.set(hi, Val::I32(r[1]));
                    let s = nonzero(self, r[0], r[1]);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let n = src32(self, 1)?;
                    let i64t = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let a = self.s.m.cast(c, Cast::ZExt, a, i64t);
                    let b = self.s.m.cast(c, Cast::ZExt, b, i64t);
                    let k32 = self.s.m.int_const(i64t, 32);
                    let b = self.s.m.bin(c, Bin::Shl, b, k32);
                    let x = self.s.m.bin(c, Bin::Or, a, b);
                    let k63 = self.s.m.i32(63);
                    let n = self.s.m.bin(c, Bin::And, n, k63);
                    let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                    let op = match name {
                        "s_lshl_b64" => Bin::Shl,
                        "s_lshr_b64" => Bin::LShr,
                        _ => Bin::AShr,
                    };
                    let r = self.s.m.bin(c, op, x, n);
                    let rl = self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32);
                    let rh = self.s.m.bin(c, Bin::LShr, r, k32);
                    let rh = self.s.m.cast(c, Cast::Trunc, rh, self.s.t.i32);
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    let s = nonzero(self, rl, rh);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_mov_b64" => {
                    // Dword by dword: one without a value (the high dword of
                    // a 64-bit wave sum) leaves the other usable.
                    for (h, v) in [(0, lo), (1, hi)] {
                        let x = half(self, 0, h).map_or_else(Val::Missing, Val::I32);
                        self.set(v, x);
                    }
                    Ok(())
                }
                // The code is taken to sit at address 0: what the program
                // computes from the counter is a byte offset in the code,
                // which accesses there read through the code table
                // (`resource::CODE` with that offset as its computed term).
                "s_getpc_b64" => {
                    let next = self.s.m.i32(inst.machine.offset as u32 + 4);
                    let zero = self.s.m.i32(0);
                    self.set(lo, Val::I32(next));
                    self.set(hi, Val::I32(zero));
                    Ok(())
                }
                "s_cselect_b64" => {
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                        let r = self.s.m.select(c, cond, a, b);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                _ => Err(name),
            };
        }
        let m = &mut self.s.m;
        let _ = m;
        let (r, carry): (V, Option<V>) = match name {
            "s_mov_b32" => (src32(self, 0)?, None),
            "s_movk_i32" => (sext16(self), None),
            "s_quadmask_b32" => {
                let pixels = src32(self, 0)?;
                (self.quad_mask(pixels), None)
            }
            "s_cselect_b32" => {
                let cond = self.scc_in(inst)?;
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.s.m.select(c, cond, a, b), None)
            }
            "s_brev_b32" => {
                let a = src32(self, 0)?;
                let r = self
                    .s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a]);
                (r, None)
            }
            "s_not_b32" => {
                let a = src32(self, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                (self.s.m.bin(c, Bin::Xor, a, ones), None)
            }
            "s_add_u32" | "s_add_i32" | "s_addk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_addk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                let r = self.s.m.bin(c, Bin::Add, a, b);
                let carry = if name == "s_add_u32" {
                    Some(self.s.m.cmp(c, pred::IULT, r, a))
                } else {
                    None
                };
                (r, carry)
            }
            "s_sub_u32" | "s_sub_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let r = self.s.m.bin(c, Bin::Sub, a, b);
                let carry = (name == "s_sub_u32").then(|| self.s.m.cmp(c, pred::IULT, a, b));
                (r, carry)
            }
            "s_addc_u32" | "s_subb_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let cin = self.scc_in(inst)?;
                let (r, carry) = self.add_sub_carry(c, name == "s_addc_u32", a, b, Some(cin));
                (r, Some(carry))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_mulk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                (self.s.m.bin(c, Bin::Mul, a, b), None)
            }
            "s_mul_hi_u32" | "s_mul_hi_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.mul_hi(a, b, name.ends_with("i32")), None)
            }
            "s_flbit_i32_b32" | "s_flbit_i32" => {
                // The first bit from the top (differing from the sign for
                // the signed form), -1 for none: DXIL's first-bit-high.
                let a = src32(self, 0)?;
                let op = if name == "s_flbit_i32_b32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_ff1_i32_b32" | "s_bcnt1_i32_b32" => {
                // First set bit (-1 for zero, as DXIL's), or the bit count.
                let a = src32(self, 0)?;
                let op = if name == "s_ff1_i32_b32" { 32 } else { 31 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                // src1: offset in bits 0-4, width in bits 16-22.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let (k16, k127) = (self.s.m.i32(16), self.s.m.i32(0x7F));
                let w = self.s.m.bin(c, Bin::LShr, b, k16);
                let w = self.s.m.bin(c, Bin::And, w, k127);
                (self.bfe(a, b, w, name == "s_bfe_i32"), None)
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                let (a, mut b) = (src32(self, 0)?, src32(self, 1)?);
                let ones = self.s.m.i32(u32::MAX);
                if name.contains("n2") {
                    b = self.s.m.bin(c, Bin::Xor, b, ones);
                }
                let op = match &name[2..5] {
                    "and" | "nan" => Bin::And,
                    "or_" | "orn" | "nor" => Bin::Or,
                    _ => Bin::Xor,
                };
                let mut r = self.s.m.bin(c, op, a, b);
                if matches!(name, "s_nand_b32" | "s_nor_b32" | "s_xnor_b32") {
                    r = self.s.m.bin(c, Bin::Xor, r, ones);
                }
                (r, None)
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let op = match name {
                    "s_lshl_b32" => Bin::Shl,
                    "s_lshr_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                (self.s.m.bin(c, op, a, b), None)
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k = self.s.m.i32(u32::from(name.as_bytes()[6] - b'0'));
                let t = self.s.m.bin(c, Bin::Shl, a, k);
                (self.s.m.bin(c, Bin::Add, t, b), None)
            }
            "s_min_u32" | "s_max_u32" | "s_min_i32" | "s_max_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let p = match name {
                    "s_min_u32" => pred::IULT,
                    "s_max_u32" => pred::IUGT,
                    "s_min_i32" => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                // SCC: whether the first operand was chosen (RDNA2 ISA).
                (self.s.m.select(c, t, a, b), Some(t))
            }
            "s_abs_i32" => {
                let a = src32(self, 0)?;
                let z = self.s.m.i32(0);
                let neg = self.s.m.cmp(c, pred::SLT, a, z);
                let n = self.s.m.bin(c, Bin::Sub, z, a);
                (self.s.m.select(c, neg, n, a), None)
            }
            "s_bitset0_b32" | "s_bitset1_b32" => {
                // Sets (clears) bit src[4:0] of the destination it reads.
                let old = self.operand(inst, dst.as_ref().ok_or("a missing operand")?)?;
                let a = src32(self, 0)?;
                let (k31, one) = (self.s.m.i32(31), self.s.m.i32(1));
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let bit = self.s.m.bin(c, Bin::Shl, one, a);
                if name == "s_bitset1_b32" {
                    (self.s.m.bin(c, Bin::Or, old, bit), None)
                } else {
                    let ones = self.s.m.i32(u32::MAX);
                    let nb = self.s.m.bin(c, Bin::Xor, bit, ones);
                    (self.s.m.bin(c, Bin::And, old, nb), None)
                }
            }
            "s_bfm_b32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let one = self.s.m.i32(1);
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                (self.s.m.bin(c, Bin::Shl, m, b), None)
            }
            _ => return Err(name),
        };
        self.set(lo, Val::I32(r));
        if let Some(scc) = scc {
            let val = match carry {
                Some(k) => Val::Bit(k),
                None if matches!(
                    name,
                    "s_mov_b32"
                        | "s_movk_i32"
                        | "s_cselect_b32"
                        | "s_brev_b32"
                        | "s_mul_i32"
                        | "s_mulk_i32"
                ) =>
                {
                    return Ok(());
                }
                None if name.starts_with("s_add") || name.starts_with("s_sub") => {
                    Val::Missing("the carry of a scalar operation")
                }
                None => {
                    let z = self.s.m.i32(0);
                    Val::Bit(self.s.m.cmp(c, pred::NE, r, z))
                }
            };
            self.vals.insert(scc, val);
        }
        Ok(())
    }

    /// Packs whether each consecutive group of four pixels has a set bit.
    pub(super) fn quad_mask(&mut self, pixels: V) -> V {
        let c = self.cur;
        let one = self.s.m.i32(1);
        let shifted = self.s.m.bin(c, Bin::LShr, pixels, one);
        let mut mask = self.s.m.bin(c, Bin::Or, pixels, shifted);
        // Fold each nibble into its bottom bit, then gather those bits into
        // the low byte in three stages without needing a 64-bit operation.
        for (distance, keep) in [
            (2, 0x1111_1111),
            (3, 0x0303_0303),
            (6, 0x000f_000f),
            (12, 0x0000_00ff),
        ] {
            let distance = self.s.m.i32(distance);
            let shifted = self.s.m.bin(c, Bin::LShr, mask, distance);
            let combined = self.s.m.bin(c, Bin::Or, mask, shifted);
            let keep = self.s.m.i32(keep);
            mask = self.s.m.bin(c, Bin::And, combined, keep);
        }
        mask
    }

    /// `a + b (+ cin)` or `a - b (- cin)` with the carry (borrow) out, as the
    /// scalar and vector add-with-carry instructions set it.
    pub(super) fn add_sub_carry(
        &mut self,
        c: usize,
        add: bool,
        a: V,
        b: V,
        cin: Option<V>,
    ) -> (V, V) {
        let op = if add { Bin::Add } else { Bin::Sub };
        let mut r = self.s.m.bin(c, op, a, b);
        // Without a carry in: a + b wraps below a; a - b borrows when a < b.
        let mut carry = if add {
            self.s.m.cmp(c, pred::IULT, r, a)
        } else {
            self.s.m.cmp(c, pred::IULT, a, b)
        };
        if let Some(cin) = cin {
            let one = self.s.m.cast(c, Cast::ZExt, cin, self.s.t.i32);
            let t = r;
            r = self.s.m.bin(c, op, t, one);
            // The carry in carries on when the partial result is all ones
            // (add) or zero (sub).
            let edge = self.s.m.i32(if add { u32::MAX } else { 0 });
            let at_edge = self.s.m.cmp(c, pred::EQ, t, edge);
            let more = self.s.m.bin(c, Bin::And, cin, at_edge);
            carry = self.s.m.bin(c, Bin::Or, carry, more);
        }
        (r, carry)
    }

    pub(super) fn scc_in(&mut self, inst: &Inst) -> Result<V, Why> {
        let v = inst.read(Reg::Scc).ok_or("an unread SCC")?;
        match self.get(v)? {
            Val::Bit(x) => Ok(x),
            _ => Err("SCC without a value"),
        }
    }

    /// Whether the dwords of one console load are read by one call for up
    /// to four of them (`bitcode::opt::WIDE` of `VICTORY_SHADER_OPT`), not
    /// by a call each.
    pub(super) fn wide_loads() -> bool {
        super::super::shader_opt() & super::super::bitcode::opt::WIDE != 0
    }

    /// `count` dwords from byte `addr` on of the raw view `h`, as integers,
    /// read by one call for every four of them ([`Self::wide_loads`]).
    /// They are the dwords of ONE console load, which the console reads as
    /// one (address plus 4, 8, ... without wrapping at 32 bits): the same
    /// bytes as a call a dword reads. Loads of different console
    /// instructions are never joined: the program computes their offsets in
    /// 32-bit registers, and `x - 4` and `x` at `x = 0` are one dword out of
    /// the view and its first one, where one call from `x - 4` on would
    /// read both out of it. After fxceless/victory-fork c36971b2.
    ///
    /// The console answers each dword past the view's end with zero, the
    /// dwords before it as they are. D3DMetal answers a call that reaches
    /// past the end the same way (`stride_tests`, a scalar load past its
    /// view's end); a host may answer it with zeros for all of it instead
    /// (the fork's water drawn black, on another host). With
    /// `bitcode::opt::CHECKED`, a call that starts within the view (its
    /// size by `getDimensions`) and ends past it is read again a dword a
    /// call. Measured on the title's frame such calls are common enough
    /// that the check costs all the four-dword calls save.
    pub(super) fn raw_dwords(&mut self, h: V, addr: V, count: u32) -> Vec<V> {
        let wide = self.raw_calls(h, addr, count, 4);
        if super::super::shader_opt() & super::super::bitcode::opt::CHECKED == 0 {
            return wide;
        }
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let u = self.s.m.undef(i32t);
        let c = self.cur;
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, u]);
        let size = self.s.m.extract(c, d, 0);
        let mut out = Vec::with_capacity(wide.len());
        for (g, group) in wide.chunks(4).enumerate() {
            let c = self.cur;
            let first = 4 * g as u32;
            let n = group.len() as u32;
            let start = match first {
                0 => addr,
                _ => {
                    let off = self.s.m.i32(4 * first);
                    self.s.m.bin(c, Bin::Add, addr, off)
                }
            };
            let bytes = self.s.m.i32(4 * n);
            let end = self.s.m.bin(c, Bin::Add, start, bytes);
            let starts_in = self.s.m.cmp(c, pred::IULT, start, size);
            let ends_out = self.s.m.cmp(c, pred::IUGT, end, size);
            let straddles = self.s.m.bin(c, Bin::And, starts_in, ends_out);
            let at = self.begin_guarded(Some(straddles));
            let one_by_one = self.raw_calls(h, start, n, 1);
            let pairs: Vec<(V, V)> = one_by_one.into_iter().zip(group.iter().copied()).collect();
            out.extend(self.end_guarded(at, &pairs));
        }
        out
    }

    /// `count` dwords from byte `addr` on of the raw view `h`, as integers,
    /// read in the current block by one call for every `per` of them.
    pub(super) fn raw_calls(&mut self, h: V, addr: V, count: u32, per: u32) -> Vec<V> {
        let c = self.cur;
        let (u, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let mut out = Vec::with_capacity(count as usize);
        for first in (0..count).step_by(per as usize) {
            let n = (count - first).min(per);
            let a = match first {
                0 => addr,
                _ => {
                    let off = self.s.m.i32(4 * first);
                    self.s.m.bin(c, Bin::Add, addr, off)
                }
            };
            let mask = self.s.m.i8((1u8 << n) - 1);
            let r = self.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                139,
                &[h, a, u, mask, al],
            );
            for k in 0..n {
                let x = self.s.m.extract(c, r, k);
                out.push(self.as_int(x));
            }
        }
        out
    }
}
