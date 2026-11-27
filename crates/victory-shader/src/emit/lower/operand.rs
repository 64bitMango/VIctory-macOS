//! Operands and their reads: scalar/vector operand sources, f16/f64
//! handling, SDWA, and the small ALU helpers. Methods of [`super::Lower`]
//! called from its dispatch and the ALU/SOP code.

use super::*;

impl<'a> Lower<'a> {
    // ---- operands ----

    pub(super) fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok(self.s.m.i32(0)),
            Operand::Scalar { code, .. } => {
                let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Vector { reg, .. } => {
                let v = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            // src_scc: SCC as 0 or 1.
            Operand::Const(253) => {
                let bit = self.scc_in(inst)?;
                Ok(self.s.m.cast(self.cur, Cast::ZExt, bit, self.s.t.i32))
            }
            // `src_pops_exiting_wave_id` (see `inputs`).
            Operand::Const(239) if self.pops => Ok(self.s.m.i32(0)),
            Operand::Const(c) => crate::resource::inline(c)
                .map(|x| self.s.m.i32(x))
                .ok_or("a special operand (vccz, execz, scc, lds_direct)"),
            Operand::Literal(x) => Ok(self.s.m.i32(x)),
            Operand::Attr { .. } => Err("an attribute operand"),
        }
    }

    /// GFX10 puts an inline f16 constant in the low half of the operand.
    pub(super) fn f16_operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Const(c @ 240..=248) => {
                let h = [
                    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                ][usize::from(c - 240)];
                Ok(self.s.m.i32(h))
            }
            _ => self.operand(inst, op),
        }
    }

    /// VGPR source `k` read for its low 16 bits alone, when only those are
    /// known (`low16`); the caller applies the modifiers and selects.
    pub(super) fn low16_src(&self, inst: &Inst, k: usize) -> Option<V> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        match v.src.get(k) {
            Some(&Operand::Vector { reg, .. }) if v.dpp.is_none() => inst
                .read(Reg::V(reg))
                .and_then(|o| self.low16.get(&o).copied()),
            _ => None,
        }
    }

    /// The f32 bits nearest the double whose dwords are `lo` and `hi`
    /// (`v_cvt_f32_f64`), in integer arithmetic: the host may have no
    /// doubles (Metal). Rounds to nearest even; results below the f32
    /// normals flush to a signed zero, NaNs come out quiet.
    pub(super) fn f64_to_f32_bits(&mut self, lo: V, hi: V) -> V {
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let (sign_k, k20, k0x7ff, mant_k) = (
            k(self, 0x8000_0000),
            k(self, 20),
            k(self, 0x7FF),
            k(self, 0xF_FFFF),
        );
        let sign = self.s.m.bin(c, Bin::And, hi, sign_k);
        let e = self.s.m.bin(c, Bin::LShr, hi, k20);
        let e = self.s.m.bin(c, Bin::And, e, k0x7ff);
        let mh = self.s.m.bin(c, Bin::And, hi, mant_k);
        // The 23 mantissa bits kept and the 29 below them.
        let (k3, k29, low29) = (k(self, 3), k(self, 29), k(self, 0x1FFF_FFFF));
        let top = self.s.m.bin(c, Bin::Shl, mh, k3);
        let next = self.s.m.bin(c, Bin::LShr, lo, k29);
        let m = self.s.m.bin(c, Bin::Or, top, next);
        let rest = self.s.m.bin(c, Bin::And, lo, low29);
        // Exponent rebiased (1023 to 127), as a signed number.
        let bias = k(self, 1023 - 127);
        let e32 = self.s.m.bin(c, Bin::Sub, e, bias);
        let (k23, one, half) = (k(self, 23), k(self, 1), k(self, 0x1000_0000));
        let ex = self.s.m.bin(c, Bin::Shl, e32, k23);
        let r = self.s.m.bin(c, Bin::Or, ex, m);
        // Round to nearest, ties to even (a carry into the exponent is
        // right, up to infinity).
        let above = self.s.m.cmp(c, pred::IUGT, rest, half);
        let tie = self.s.m.cmp(c, pred::EQ, rest, half);
        let odd = self.s.m.bin(c, Bin::And, m, one);
        let z = k(self, 0);
        let odd = self.s.m.cmp(c, pred::NE, odd, z);
        let tie_up = self.s.m.bin(c, Bin::And, tie, odd);
        let up = self.s.m.bin(c, Bin::Or, above, tie_up);
        let inc = self.s.m.select(c, up, one, z);
        let r = self.s.m.bin(c, Bin::Add, r, inc);
        // Out of range: too large is infinity, too small zero.
        let inf = k(self, 0x7F80_0000);
        let k255 = k(self, 255);
        let big = self.s.m.cmp(c, pred::SGE, e32, k255);
        let r = self.s.m.select(c, big, inf, r);
        let small = self.s.m.cmp(c, pred::SLE, e32, z);
        let r = self.s.m.select(c, small, z, r);
        // Infinity and NaN.
        let special = self.s.m.cmp(c, pred::EQ, e, k0x7ff);
        let any = self.s.m.bin(c, Bin::Or, mh, lo);
        let nan = self.s.m.cmp(c, pred::NE, any, z);
        let qnan = k(self, 0x7FC0_0000);
        let sp = self.s.m.select(c, nan, qnan, inf);
        let r = self.s.m.select(c, special, sp, r);
        self.s.m.bin(c, Bin::Or, r, sign)
    }

    /// A dword (`high`: the high one) of the double equal to the f32 with
    /// bits `x` (exact; f32 denormals as signed zeros).
    pub(super) fn f32_bits_to_f64_dword(&mut self, x: V, high: bool) -> V {
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let (k23, kff, mant_k, z) = (k(self, 23), k(self, 0xFF), k(self, 0x7F_FFFF), k(self, 0));
        let e = self.s.m.bin(c, Bin::LShr, x, k23);
        let e = self.s.m.bin(c, Bin::And, e, kff);
        let m = self.s.m.bin(c, Bin::And, x, mant_k);
        let zero = self.s.m.cmp(c, pred::EQ, e, z);
        if !high {
            let k29 = k(self, 29);
            let lo = self.s.m.bin(c, Bin::Shl, m, k29);
            return self.s.m.select(c, zero, z, lo);
        }
        let sign_k = k(self, 0x8000_0000);
        let sign = self.s.m.bin(c, Bin::And, x, sign_k);
        let special = self.s.m.cmp(c, pred::EQ, e, kff);
        let rebias = k(self, 1023 - 127);
        let e64 = self.s.m.bin(c, Bin::Add, e, rebias);
        let k0x7ff = k(self, 0x7FF);
        let e64 = self.s.m.select(c, special, k0x7ff, e64);
        let (k20, k3) = (k(self, 20), k(self, 3));
        let e64 = self.s.m.bin(c, Bin::Shl, e64, k20);
        let mh = self.s.m.bin(c, Bin::LShr, m, k3);
        let hi = self.s.m.bin(c, Bin::Or, e64, mh);
        let hi = self.s.m.select(c, zero, z, hi);
        self.s.m.bin(c, Bin::Or, hi, sign)
    }

    /// The dwords of f64 source `k`, its `neg` and `abs` applied.
    pub(super) fn f64_src(&mut self, inst: &Inst, k: usize) -> Result<(V, V), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.sdwa.is_some() || v.dpp.is_some() {
            return Err("f64 operand modifiers");
        }
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let (lo, hi) = match op {
            Operand::Vector { reg, .. } => (Reg::V(reg), Reg::V(reg + 1)),
            Operand::Scalar { code, .. } if code < 105 => (Reg::S(code), Reg::S(code + 1)),
            Operand::Const(c @ (128..=208 | 240..=248)) => {
                let d: f64 = match c {
                    128..=192 => f64::from(c - 128),
                    193..=208 => -f64::from(c - 192),
                    248 => 0.5 / std::f64::consts::PI,
                    _ => [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0][usize::from(c - 240)],
                };
                let b = d.to_bits();
                let (lo, hi) = (self.s.m.i32(b as u32), self.s.m.i32((b >> 32) as u32));
                return Ok(self.f64_modifiers(v, k, lo, hi));
            }
            // A literal is the double's high dword.
            Operand::Literal(x) => {
                let (lo, hi) = (self.s.m.i32(0), self.s.m.i32(x));
                return Ok(self.f64_modifiers(v, k, lo, hi));
            }
            _ => return Err("an f64 operand"),
        };
        let lo = self.i32_of(inst.read(lo).ok_or("an unread operand")?)?;
        let hi = self.i32_of(inst.read(hi).ok_or("an unread operand")?)?;
        Ok(self.f64_modifiers(v, k, lo, hi))
    }

    pub(super) fn f64_modifiers(
        &mut self,
        v: &crate::gfx10::Valu,
        k: usize,
        lo: V,
        hi: V,
    ) -> (V, V) {
        let c = self.cur;
        let mut hi = hi;
        if v.abs >> k & 1 != 0 {
            let m = self.s.m.i32(0x7FFF_FFFF);
            hi = self.s.m.bin(c, Bin::And, hi, m);
        }
        if v.neg >> k & 1 != 0 {
            let m = self.s.m.i32(0x8000_0000);
            hi = self.s.m.bin(c, Bin::Xor, hi, m);
        }
        (lo, hi)
    }

    pub(super) fn as_float(&mut self, x: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, x, self.s.t.f32)
    }

    pub(super) fn as_int(&mut self, f: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, f, self.s.t.i32)
    }

    /// The SDWA operand selection of source `k`: a byte or word of the
    /// dword, zero- or sign-extended (none: the whole dword).
    pub(super) fn sdwa_src(inst: &Inst, k: usize) -> Option<(u8, bool)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        let sel = *s.src_sel.get(k)?;
        let sext = s.sext >> k & 1 != 0;
        (sel < 6).then_some((sel, sext))
    }

    /// Bits `sel` (0-3 a byte, 4-5 a word) of `x`, extended.
    pub(super) fn select_bits(&mut self, x: V, sel: u8, sext: bool) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8, 8 * sel)
        } else {
            (16, 16 * (sel - 4))
        };
        let left = self.s.m.i32(u32::from(32 - width - at));
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - u32::from(width));
        let op = if sext { Bin::AShr } else { Bin::LShr };
        self.s.m.bin(c, op, t, right)
    }

    /// A float source with its abs/neg modifiers.
    pub(super) fn fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let selected = Self::sdwa_src(inst, k);
        let mut f = match op {
            _ if selected.is_some() => {
                let (sel, sext) = selected.expect("checked");
                let x = self.operand(inst, &op)?;
                let x = self.select_bits(x, sel, sext);
                self.as_float(x)
            }
            _ if k == 0 && self.src0.is_some() => {
                let x = self.src0.expect("checked");
                self.as_float(x)
            }
            Operand::Vector { reg, .. } | Operand::Scalar { code: reg, .. }
                if !matches!(op, Operand::Scalar { code: NULL, .. }) =>
            {
                let r = if matches!(op, Operand::Vector { .. }) {
                    Reg::V(reg)
                } else {
                    Reg::S(reg)
                };
                let val = inst.read(r).ok_or("an unread operand")?;
                match self.floats.get(&val) {
                    Some(&f) => f,
                    None => {
                        let x = self.i32_of(val)?;
                        self.as_float(x)
                    }
                }
            }
            _ => {
                let x = self.operand(inst, &op)?;
                self.as_float(x)
            }
        };
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    pub(super) fn src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 {
            return Err("integer operand modifiers");
        }
        match self.src0 {
            Some(x) if k == 0 => Ok(x),
            _ => {
                let sel = Self::sdwa_src(inst, k);
                // A byte or word select within the low half.
                let low = matches!(sel, Some((0 | 1 | 4, _)));
                let x = match self.low16_src(inst, k).filter(|_| low) {
                    Some(x) => x,
                    None => self.operand(inst, &op)?,
                };
                Ok(match sel {
                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                    None => x,
                })
            }
        }
    }

    pub(super) fn unary(&mut self, op: u32, f: V) -> V {
        self.s.op(
            self.cur,
            "unary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[f],
        )
    }

    pub(super) fn binary(&mut self, op: u32, a: V, b: V) -> V {
        self.s.op(
            self.cur,
            "binary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[a, b],
        )
    }

    /// A 32-bit denormal as the console's float mode reads it: a zero of its
    /// sign, decided on the bits (the host's own denormal mode varies).
    pub(super) fn ftz(&mut self, f: V) -> V {
        if self.x.relax.denormals {
            return f;
        }
        let c = self.cur;
        let x = self.as_int(f);
        let (exp, sign, zero) = (
            self.s.m.i32(0x7F80_0000),
            self.s.m.i32(0x8000_0000),
            self.s.m.i32(0),
        );
        let e = self.s.m.bin(c, Bin::And, x, exp);
        let den = self.s.m.cmp(c, pred::EQ, e, zero);
        let s = self.s.m.bin(c, Bin::And, x, sign);
        let r = self.s.m.select(c, den, s, x);
        self.as_float(r)
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but of the operands with denormals read as zeros and with
    /// -0 ordered below +0 (RDNA2 ISA, `v_max_f32`), where the compare and
    /// select DXIL specifies gives either zero.
    pub(super) fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let (a, b) = (self.ftz(a), self.ftz(b));
        self.max_min_flushed(max, a, b)
    }

    /// [`Self::max_min`] of operands without denormals.
    pub(super) fn max_min_flushed(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        if self.x.relax.signed_zeros {
            return r;
        }
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    pub(super) fn fmad(&mut self, a: V, b: V, c: V) -> V {
        self.s.op(
            self.cur,
            "tertiary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            46,
            &[a, b, c],
        )
    }

    /// `f` as an f16 in the low bits, rounded as the program's MODE says at
    /// this point: toward zero where it set that for 16-bit results (the
    /// title does around its 16-bit interpolation), else to nearest even;
    /// f32 denormals read as zero; on the bits, where the host's own
    /// conversion rounds as it likes (toward zero on the RX 6700 XT and WARP).
    // shortcut: rounding toward +-infinity, or a mode not known at this
    // point (paths that set it differently join), rounds to nearest even.
    pub(super) fn narrow16(&mut self, f: V) -> V {
        if self.x.relax.half_rounding {
            let i32t = self.s.t.i32;
            return self
                .s
                .op(self.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        }
        let rtz = self.mode16 == Some(3);
        let c = self.cur;
        let x = self.as_int(f);
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k13, k16, k31) = (k(self, 13), k(self, 16), k(self, 31));
        let t = self.s.m.bin(c, Bin::LShr, x, k16);
        let m = k(self, 0x8000);
        let sign = self.s.m.bin(c, Bin::And, t, m);
        let m = k(self, 0x7FFF_FFFF);
        let a = self.s.m.bin(c, Bin::And, x, m);
        // Normal halves: the exponent rebased, the mantissa rounded.
        let bias = k(self, 0x3800_0000);
        let rebased = self.s.m.bin(c, Bin::Sub, a, bias);
        let normal = if rtz {
            self.s.m.bin(c, Bin::LShr, rebased, k13)
        } else {
            let one = k(self, 1);
            let lsb = self.s.m.bin(c, Bin::LShr, rebased, k13);
            let lsb = self.s.m.bin(c, Bin::And, lsb, one);
            let half = k(self, 0xFFF);
            let up = self.s.m.bin(c, Bin::Add, rebased, half);
            let up = self.s.m.bin(c, Bin::Add, up, lsb);
            self.s.m.bin(c, Bin::LShr, up, k13)
        };
        let (inf, big) = (k(self, 0x7C00), k(self, if rtz { 0x7BFF } else { 0x7C00 }));
        let over = self.s.m.cmp(c, pred::IUGE, normal, inf);
        let normal = self.s.m.select(c, over, big, normal);
        let infinity = k(self, 0x7F80_0000);
        let is_inf = self.s.m.cmp(c, pred::EQ, a, infinity);
        let normal = self.s.m.select(c, is_inf, inf, normal);
        // Denormal halves: the mantissa shifted into units of 2^-24.
        let k23 = k(self, 23);
        let e = self.s.m.bin(c, Bin::LShr, a, k23);
        let k126 = k(self, 126);
        let s = self.s.m.bin(c, Bin::Sub, k126, e);
        let far = self.s.m.cmp(c, pred::IUGT, s, k31);
        let s = self.s.m.select(c, far, k31, s);
        let (frac, hidden, zero) = (k(self, 0x7F_FFFF), k(self, 0x80_0000), k(self, 0));
        let mant = self.s.m.bin(c, Bin::And, a, frac);
        let mant = self.s.m.bin(c, Bin::Or, mant, hidden);
        let f32_denormal = self.s.m.cmp(c, pred::EQ, e, zero);
        let mant = self.s.m.select(c, f32_denormal, zero, mant);
        let q = self.s.m.bin(c, Bin::LShr, mant, s);
        let den = if rtz {
            q
        } else {
            let one = k(self, 1);
            let s1 = self.s.m.bin(c, Bin::Sub, s, one);
            let half = self.s.m.bin(c, Bin::LShr, mant, s1);
            let half = self.s.m.bin(c, Bin::And, half, one);
            let below = self.s.m.bin(c, Bin::Shl, one, s1);
            let below = self.s.m.bin(c, Bin::Sub, below, one);
            let rest = self.s.m.bin(c, Bin::And, mant, below);
            let rest = self.s.m.cmp(c, pred::NE, rest, zero);
            let rest = self.s.m.cast(c, Cast::ZExt, rest, self.s.t.i32);
            let lsb = self.s.m.bin(c, Bin::And, q, one);
            let tie = self.s.m.bin(c, Bin::Or, rest, lsb);
            let up = self.s.m.bin(c, Bin::And, half, tie);
            self.s.m.bin(c, Bin::Add, q, up)
        };
        let min_normal = k(self, 0x3880_0000);
        let small = self.s.m.cmp(c, pred::IULT, a, min_normal);
        let r = self.s.m.select(c, small, den, normal);
        let is_nan = self.s.m.cmp(c, pred::IUGT, a, infinity);
        // A NaN keeps the top of its payload, quieted (as the RX 6700 XT,
        // RDNA2 like the console, converts).
        let payload = self.s.m.bin(c, Bin::LShr, a, k13);
        let (low10, quiet) = (k(self, 0x3FF), k(self, 0x7E00));
        let payload = self.s.m.bin(c, Bin::And, payload, low10);
        let nan = self.s.m.bin(c, Bin::Or, payload, quiet);
        let r = self.s.m.select(c, is_nan, nan, r);
        self.s.m.bin(c, Bin::Or, r, sign)
    }

    /// `f` as an f16 in the low bits, rounded toward zero: the nearest-even
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    pub(super) fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let h = self
            .s
            .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        let back = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's
    /// top bit when `signed`; widths from 32 on take every bit. A signed
    /// field that runs past bit 31 extends from bit 31, as the hardware
    /// does: the shift is arithmetic then.
    pub(super) fn bfe(&mut self, a: V, off: V, width: V, signed: bool) -> V {
        let c = self.cur;
        let (k31, k32, zero, one, ones) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(u32::MAX),
        );
        let off = self.s.m.bin(c, Bin::And, off, k31);
        let shift = if signed { Bin::AShr } else { Bin::LShr };
        let shifted = self.s.m.bin(c, shift, a, off);
        let full = self.s.m.cmp(c, pred::IUGE, width, k32);
        let w = self.s.m.bin(c, Bin::And, width, k31);
        let mask = self.s.m.bin(c, Bin::Shl, one, w);
        let mask = self.s.m.bin(c, Bin::Sub, mask, one);
        let mask = self.s.m.select(c, full, ones, mask);
        let r = self.s.m.bin(c, Bin::And, shifted, mask);
        if !signed {
            return r;
        }
        // Sign-extend a field narrower than 32 bits (none for width 0).
        let left = self.s.m.bin(c, Bin::Sub, k32, w);
        let left = self.s.m.bin(c, Bin::And, left, k31);
        let t = self.s.m.bin(c, Bin::Shl, r, left);
        let t = self.s.m.bin(c, Bin::AShr, t, left);
        let is0 = self.s.m.cmp(c, pred::EQ, w, zero);
        let keep = self.s.m.bin(c, Bin::Or, full, is0);
        self.s.m.select(c, keep, r, t)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the three instead (the host's minimum skips NaNs, so that
    /// is the smallest number among them).
    pub(super) fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let (a, b, third) = (self.ftz(a), self.ftz(b), self.ftz(third));
        let lo = self.max_min_flushed(false, a, b);
        let hi = self.max_min_flushed(true, a, b);
        let t = self.max_min_flushed(false, hi, third);
        let median = self.max_min_flushed(true, lo, t);
        if self.x.relax.nan {
            return median;
        }
        let least = self.max_min_flushed(false, lo, third);
        let ab = self.s.m.cmp(c, pred::UNO, a, b);
        let nan3 = self.s.m.cmp(c, pred::UNO, third, third);
        let nan = self.s.m.bin(c, Bin::Or, ab, nan3);
        self.s.m.select(c, nan, least, median)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    pub(super) fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    pub(super) fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        let r = self.s.m.select(c, exact, zero, r);
        // The hardware returns NaN for NaN and for infinity, whose reduced
        // part is NaN. Metal's sin and cos give 0 there, so that NaN is
        // passed through here instead of relying on the host.
        let nan = self.s.m.cmp(c, pred::UNO, part, part);
        let r = self.s.m.select(c, nan, part, r);
        if cosine {
            return r;
        }
        // sin(±0) is ±0. Some Metal GPUs (M2) drop the sign through the
        // host's sin or a float select, so a zero input is passed through
        // as integer bits (from nickorlabs's #30).
        let bits = self.as_int(x);
        let mask = self.s.m.i32(0x7fff_ffff);
        let magnitude = self.s.m.bin(c, Bin::And, bits, mask);
        let zero = self.s.m.i32(0);
        let is_zero = self.s.m.cmp(c, pred::EQ, magnitude, zero);
        let result = self.as_int(r);
        let result = self.s.m.select(c, is_zero, bits, result);
        self.as_float(result)
    }

    /// The high dword of a 32 × 32-bit product.
    pub(super) fn mul_hi(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let ext = if signed { Cast::SExt } else { Cast::ZExt };
        let a = self.s.m.cast(c, ext, a, i64t);
        let b = self.s.m.cast(c, ext, b, i64t);
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::LShr, m, k32);
        self.s.flags |= flags::INT64;
        self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32)
    }

    /// `0 × anything = 0`, as in D3D9.
    pub(super) fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        if self.x.relax.legacy_mul {
            return m;
        }
        let zero = self.s.m.f32(0);
        if self.x.relax.legacy_mul_nan {
            // Ordered and not zero: a NaN or zero product is +0.
            let number = self.s.m.cmp(c, pred::ONE, m, zero);
            return self.s.m.select(c, number, m, zero);
        }
        let az = self.s.m.cmp(c, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(c, pred::OEQ, b, zero);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        self.s.m.select(c, z, zero, m)
    }

    // ---- lane booleans ----

    pub(super) fn boolean(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        if b == Bool::TRUE || b == Bool::FALSE {
            return Ok(self.s.m.bool(b == Bool::TRUE));
        }
        if let Some(&v) = self.bools.get(&(at, b)) {
            return Ok(v);
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let a = self.atom(atom, at)?;
        let m = &mut self.s.m;
        let v = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => {
                let t = m.bool(true);
                m.bin(at, Bin::Xor, a, t)
            }
            (Bool::FALSE, h) => {
                let h = self.boolean(h, at)?;
                self.s.m.bin(at, Bin::And, a, h)
            }
            (l, Bool::TRUE) => {
                let l = self.boolean(l, at)?;
                self.s.m.bin(at, Bin::Or, a, l)
            }
            (l, h) => {
                let l = self.boolean(l, at)?;
                let h = self.boolean(h, at)?;
                self.s.m.select(at, a, h, l)
            }
        };
        self.bools.insert((at, b), v);
        Ok(v)
    }

    pub(super) fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        match atom {
            Atom::Lane(v) => self.lane_bits.get(&v).copied().ok_or_else(|| {
                self.lane_fail
                    .get(&v)
                    .copied()
                    .unwrap_or("a lane mask from a carry")
            }),
            Atom::Uniform(v) => match self.get(v)? {
                Val::Bit(x) => Ok(x),
                Val::I32(x) => {
                    let z = self.s.m.i32(0);
                    Ok(self.s.m.cmp(at, pred::NE, x, z))
                }
                Val::Missing(w) => Err(w),
            },
            Atom::Phi(id) => {
                if let Some(&v) = self.mask_phis.get(&id) {
                    return Ok(v);
                }
                let &(b, i) = self
                    .phi_at
                    .get(&id)
                    .ok_or("a lane-mask join without a phi")?;
                let blk = self.block[b].ok_or("an unreachable join")?;
                let v = self.s.m.phi(blk, self.s.t.i1);
                self.mask_phis.insert(id, v);
                self.pending_masks.push((v, b, i));
                Ok(v)
            }
            // shortcut: "some lane of the quad is set" is taken as true: the
            // lanes it adds only feed derivatives, and exports keep their own
            // masks. Wrong if such a section had a side effect.
            Atom::Quad(_) => Ok(self.s.m.bool(true)),
            // With console lanes, a mask held as a number is the console's:
            // the lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes => {
                let hi = match self.pair_high(v) {
                    Some(h) => Some(self.get(h)?),
                    None => None,
                };
                let lo = self.get(v)?;
                let word = |w: Val| match w {
                    Val::I32(x) => Ok(x),
                    _ => Err("an opaque lane mask"),
                };
                let lo = word(lo)?;
                let hi = hi.map(word).transpose()?;
                Ok(self.lane_bit(at, lo, hi))
            }
            Atom::Constant(k) if self.console_lanes => {
                let lo = self.s.m.i32(k as u32);
                let hi = self.s.m.i32((k >> 32) as u32);
                Ok(self.lane_bit(at, lo, Some(hi)))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
            // (made only where the branch is lowered, `Self::wave_decision`)
            Atom::Wave(_) => Err("a wave-wide decision as a lane's"),
        }
    }

    /// The bit of lane `lane` in the mask `lo`, `hi`.
    pub(super) fn lane_bit_of(&mut self, at: usize, lane: V, lo: V, hi: V) -> V {
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let low = self.s.m.cmp(at, pred::IULT, lane, k32);
        let word = self.s.m.select(at, low, lo, hi);
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    /// The bit position of byte LDS address `a` + `offset` in its dword.
    pub(super) fn lds_shift(&mut self, a: V, offset: u32) -> V {
        let c = self.cur;
        let (o, three) = (self.s.m.i32(offset), self.s.m.i32(3));
        let at = self.s.m.bin(c, Bin::Add, a, o);
        let byte = self.s.m.bin(c, Bin::And, at, three);
        self.s.m.bin(c, Bin::Shl, byte, three)
    }

    /// This lane's bit of the mask `lo`, `hi` (wave64).
    pub(super) fn lane_bit(&mut self, at: usize, lo: V, hi: Option<V>) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let lane = if self.halves.is_some() {
            self.lane_index()
        } else {
            self.s.op(
                at,
                "waveGetLaneIndex",
                Attr::ReadOnly,
                self.s.t.i32,
                111,
                &[],
            )
        };
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = match hi {
            Some(hi) => {
                let low = self.s.m.cmp(at, pred::IULT, lane, k32);
                self.s.m.select(at, low, lo, hi)
            }
            None => lo,
        };
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    pub(super) fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        let b = *self.def_block.get(&lo)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
        // A pair written one dword at a time (an `s_mov_b32` each): the
        // high dword its uses read with this low one.
        inst.write(pick(reg)?)
            .or_else(|| self.x.st.other_highs.get(&lo).copied())
    }
}
