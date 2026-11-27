//! Memory: shared (LDS), constant, descriptor and heap handling in the
//! lowering, as DXIL. Methods of [`super::Lower`] called from its
//! instruction dispatch and from the ALU and wave code.

use super::*;

impl<'a> Lower<'a> {
    pub(super) fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Fields::Smem {
            data,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, a);
        }
        if let Some(a) = self
            .scalar
            .get(&site)
            .copied()
            .filter(|a| a.param == USER_DATA)
        {
            return self.user_data_smem(inst, a);
        }
        if let Some(a) = self
            .scalar
            .get(&site)
            .copied()
            .filter(|a| a.param == MEMORY)
        {
            return self.memory_smem(inst, a);
        }
        let Some(rest) = name.strip_prefix("s_buffer_load_dword") else {
            return self.descriptor_data(inst, site);
        };
        let count = match rest {
            "" => 1,
            "x2" => 2,
            "x4" => 4,
            "x8" => 8,
            "x16" => 16,
            _ => return Err("a scalar buffer load"),
        };
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar buffer load");
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked descriptor")?;
        let kind = self.used_kind(src);
        let c = self.cur;
        let (base, extra) = self.built_offset(src)?;
        let offset = &((*offset as u32).wrapping_add(base) as i32);
        let mut addr = self.s.m.i32(*offset as u32);
        if let Some(x) = extra {
            addr = self.s.m.bin(c, Bin::Add, addr, x);
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            // A descriptor built around a table in the code.
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.code_dword(a)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        // Constant data through the resource heap itself is a raw buffer: the
        // host's constant buffer views need 256-byte alignment, the title's
        // 16 (module notes, "Constant buffers in the heap").
        let heap_constants = kind.is_none() && self.is_heap_direct(src);
        // So is constant data in a table: the title's slot holds a raw view
        // of it, so tables need no copies of their own per draw.
        let table_constants =
            matches!(kind, Some(u) if u.kind == ResourceType::Cbv && u.root == RootType::Table);
        let srv = heap_constants
            || table_constants
            || matches!(kind, Some(u) if u.kind == ResourceType::Srv);
        // A root UAV is read as title memory in a program that uses it.
        let root_uav = self.root_memory
            && matches!(kind, Some(u) if u.kind == ResourceType::Uav)
            && matches!(src, Source::Built { address, .. }
                if self.x.native.layout.get(address.param).is_some_and(|p| p.root == RootType::Descriptor));
        if srv || root_uav {
            let (h, addr) = self.raw_view(site, src, addr, 4 * count, (11, 0), false, None)?;
            if count > 1 && Self::wide_loads() {
                for (k, x) in self.raw_dwords(h, addr, count).into_iter().enumerate() {
                    self.write(inst, Reg::S(code + k as u8), x, None);
                }
                return Ok(());
            }
            let c = self.cur;
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let mask = self.s.m.i8(1);
                let (u, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, a, u, mask, al],
                );
                let x = self.s.m.extract(c, r, 0);
                let x = self.as_int(x);
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let h = self.handle(site, src, false, (13, 0x1_0000))?;
        let constant = match addr {
            V::Const(_)
                if matches!(soffset, Operand::Scalar { code: NULL, .. }) && extra.is_none() =>
            {
                Some(*offset as u32)
            }
            _ => None,
        };
        let mut rows: HashMap<u32, V> = HashMap::new();
        for k in 0..count {
            let x = match constant {
                Some(base) => {
                    let byte = base + 4 * k;
                    let row = match rows.get(&(byte >> 4)) {
                        Some(&r) => r,
                        None => {
                            let ri = self.s.m.i32(byte >> 4);
                            let r = self.s.op(
                                c,
                                "cbufferLoadLegacy.i32",
                                Attr::ReadOnly,
                                self.s.t.cbuf_ret,
                                59,
                                &[h, ri],
                            );
                            rows.insert(byte >> 4, r);
                            r
                        }
                    };
                    self.s.m.extract(c, row, (byte >> 2) & 3)
                }
                None => {
                    let off = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, addr, off);
                    self.cbuffer_dword(h, a)
                }
            };
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// A descriptor load (`s_load_dword*` from a table or heap) whose dwords
    /// the code also reads as numbers: the console compiler computes sizes
    /// from descriptor bits (an image's width, height, depth, levels; a
    /// buffer's record count) instead of querying them. Those fields are
    /// rebuilt from the host view's dimensions in the GFX10 layout (image
    /// dword 1 bits 30-31 and dword 2 bits 0-13 width - 1, dword 2 bits
    /// 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
    /// 28, dword 4 depth or layers - 1; buffer dword 2 the record count).
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right) and the other
    // dwords (addresses, formats, tiling) have no host value: reading them
    // stops the program.
    pub(super) fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        const OTHER: Why = "a descriptor's address or format read as data";
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data"),
        };
        // How the program uses the descriptor tells the view's kind; a load
        // read only as data takes it from a load of the same descriptor that
        // the program uses (the compiler loads a descriptor again rather than
        // keep it, and queries sizes in loops of their own).
        let use_of = |site: Site| {
            self.x
                .res
                .descriptors
                .iter()
                .filter(|d| d.role == Role::Resource)
                .find_map(|d| {
                    d.source.leaves().into_iter().find_map(|l| match *l {
                        Source::Loaded { load, first } if load == site => Some((d.site, first)),
                        _ => None,
                    })
                })
        };
        // The VGPR a waterfall loop reads the first lane of.
        let lane_source = |v: Value| {
            let i = self.def_inst(v)?;
            match i.fields {
                Fields::Valu(ref f) if i.name() == "v_readfirstlane_b32" => match f.src[..] {
                    [Operand::Vector { reg, .. }] => i.read(Reg::V(reg)),
                    _ => None,
                },
                _ => None,
            }
        };
        let same = |a: &Address, b: &Address| {
            a.param == b.param
                && a.offset == b.offset
                && (a.dynamic == b.dynamic
                    || a.dynamic.and_then(lane_source).is_some()
                        && a.dynamic.and_then(lane_source) == b.dynamic.and_then(lane_source))
        };
        let address = self.scalar.get(&site);
        let use_at = use_of(site).or_else(|| {
            self.x
                .res
                .scalar
                .iter()
                .filter(|&(s, a)| *s != site && address.is_some_and(|b| same(a, b)))
                .find_map(|&(s, _)| use_of(s))
        });
        let (first, src, props, image, uav) = if let Some((at, first)) = use_at {
            let src = Source::Loaded { load: site, first };
            let used = self.used_kind(&src);
            let ui: &Inst = &self.x.p.blocks[at.0].insts[at.1];
            let un = ui.name();
            let uav = un.contains("store")
                || un.contains("atomic")
                || used.is_some_and(|u| u.kind == ResourceType::Uav);
            let u = if uav { 0x1000 } else { 0 };
            let (props, image) = match &ui.fields {
                Fields::Image { dim, .. } => {
                    let kind = match dim {
                        0 => 1,
                        1 => 2,
                        2 => 4,
                        4 => 6,
                        5 => 7,
                        _ => return Err("an image dimension (cube or MSAA)"),
                    };
                    ((kind | u, 0x409), Some(kind))
                }
                Fields::Buffer { idxen, format, .. } => {
                    let stride = used.map_or(0, |u| u.stride);
                    let props = if un.starts_with("buffer_load_format")
                        || un.starts_with("buffer_store_format")
                    {
                        (10 | u, 0x409)
                    } else if *idxen && format.is_none() && stride >= 4 && stride.is_multiple_of(4)
                    {
                        (12 | u, stride)
                    } else {
                        (11 | u, 0)
                    };
                    if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                        return Err("a constant buffer's descriptor read as data");
                    }
                    (props, None)
                }
                // A buffer the code reads with scalar loads (`buf[buf.Length - 1]`
                // at a uniform index): structured by the used-resource entry's
                // stride, so its record count is in elements, else raw (bytes).
                Fields::Smem { .. } => {
                    let used = used.ok_or("a scalar-loaded buffer of unknown kind read as data")?;
                    if used.kind == ResourceType::Cbv {
                        return Err("a constant buffer's descriptor read as data");
                    }
                    let props = match used.stride {
                        0 | 1 => (11 | u, 0),
                        s if s.is_multiple_of(4) => (12 | u, s),
                        _ => return Err("a scalar-loaded buffer of odd stride read as data"),
                    };
                    (props, None)
                }
                _ => return Err("a scalar memory read as data"),
            };
            (first, src, props, image, uav)
        } else {
            // An image the program only measures: it reads the size bits of
            // eight dwords loaded from a table or the heap and samples other
            // images (bindless materials, by index). Its host view is taken
            // as a 2D texture.
            // shortcut: a 3D or array image reports depth or layers 1, a
            // cube one its face size.
            let src = Source::Loaded {
                load: site,
                first: 0,
            };
            let heap = self.scalar.get(&site).is_some_and(|a| {
                matches!(
                    self.x.native.layout.get(a.param).map(|p| p.root),
                    Some(RootType::Table | RootType::ResourceHeap)
                )
            });
            let used = self.used_kind(&src);
            // A structured buffer the program only counts the records of
            // (dword 2): an exact, statically addressed four-dword binding
            // that the container declares structured, never the nearest
            // earlier entry or a dynamically indexed range.
            let address = self.scalar.get(&site);
            let structured = |uav: bool| {
                used.filter(|b| {
                    count == 4
                        && address.is_some_and(|a| {
                            a.dynamic.is_none()
                                && b.table_offset_dwords.checked_mul(4) == Some(a.offset)
                        })
                        && b.dwords == 4
                        && b.kind
                            == if uav {
                                ResourceType::Uav
                            } else {
                                ResourceType::Srv
                            }
                        && (4..=2048).contains(&b.stride)
                        && b.stride.is_multiple_of(4)
                        && self.x.psv.is_some_and(|part| {
                            super::super::declares_structured_buffer(part, b.lower_bound, uav)
                        })
                })
            };
            if let Some(b) = structured(false).or_else(|| structured(true)) {
                let uav = b.kind == ResourceType::Uav;
                let u = if uav { 0x1000 } else { 0 };
                (0, src, (12 | u, b.stride), None, uav)
            } else if count == 8 && heap && used.is_none_or(|u| u.kind == ResourceType::Srv) {
                (0, src, (2, 0x409), Some(2), false)
            } else {
                return self.view_addresses(inst, site, code, count);
            }
        };
        let h = self.handle(site, &src, false, props)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = if image.is_some() && (!uav || self.x.uav_size_level0) {
            self.s.m.i32(0)
        } else {
            self.s.m.undef(i32t)
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
        let words: Vec<(u32, V)> = match image {
            Some(kind) => {
                let one = self.s.m.i32(1);
                let w = field(self, 0);
                let (height, depth) = match kind {
                    1 => (one, one),
                    2 => (field(self, 1), one),
                    6 => (one, field(self, 1)),
                    _ => (field(self, 1), field(self, 2)),
                };
                let levels = if uav { one } else { field(self, 3) };
                let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
                let (w, height, depth, levels) = (
                    less(self, w),
                    less(self, height),
                    less(self, depth),
                    less(self, levels),
                );
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (k30, k2, k14, k16, mask) = (
                    k(self, 30),
                    k(self, 2),
                    k(self, 14),
                    k(self, 16),
                    k(self, 0x3FFF),
                );
                let d1 = self.s.m.bin(c, Bin::Shl, w, k30);
                let hi = self.s.m.bin(c, Bin::LShr, w, k2);
                let hi = self.s.m.bin(c, Bin::And, hi, mask);
                let ht = self.s.m.bin(c, Bin::Shl, height, k14);
                let d2 = self.s.m.bin(c, Bin::Or, hi, ht);
                let ty = [0, 8, 9, 0, 10, 0, 12, 13][kind as usize];
                let base = k(self, 0xFAC | ty << 28);
                let last = self.s.m.bin(c, Bin::Shl, levels, k16);
                let d3 = self.s.m.bin(c, Bin::Or, base, last);
                vec![(1, d1), (2, d2), (3, d3), (4, depth)]
            }
            None => vec![(2, field(self, 0))],
        };
        for k in 0..count {
            let rel = k.wrapping_sub(u32::from(first));
            match words.iter().find(|w| w.0 == rel) {
                Some(&(_, x)) => self.write(inst, Reg::S(code + k as u8), x, None),
                None => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(OTHER));
                    }
                }
            }
        }
        Ok(())
    }

    /// `s_load_dword*` from the root signature's memory: user-data dwords.
    pub(super) fn user_data_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = smem_dwords(inst.name()).ok_or("a scalar load from the root signature")?;
        for k in 0..count {
            let index = a.offset / 4 + k;
            let held = self.x.native.layout.iter().enumerate().find_map(|(i, p)| {
                let dword = index.checked_sub(p.user_data.into())?;
                (dword < p.sgprs).then_some((i, dword))
            });
            let val = match held {
                Some((param, dword)) => self.user(param, dword),
                None => Val::Missing("a root-signature dword no parameter holds"),
            };
            if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                self.vals.insert(v, val);
            }
        }
        Ok(())
    }

    /// `s_load_dword*` from title memory at an address the program computes.
    pub(super) fn memory_smem(&mut self, inst: &Inst, _a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            base: Operand::Scalar { code: base, .. },
            offset,
            soffset,
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = smem_dwords(inst.name()).ok_or("a scalar load from title memory")?;
        let word = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst
                .read(Reg::S(base + k))
                .ok_or("a scalar memory address")?;
            l.i32_of(v)
        };
        let (lo, hi) = (word(self, 0)?, word(self, 1)?);
        // The base's high dword holds address bits 32-47.
        let mask = self.s.m.i32(0xFFFF);
        let hi = self.s.m.bin(self.cur, Bin::And, hi, mask);
        // The immediate is signed (21 bits), the offset register unsigned.
        let mut extra = vec![(self.s.m.i32(offset as u32), offset < 0)];
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            extra.push((self.operand(inst, &soffset)?, false));
        }
        let (mut lo, mut hi) = (lo, hi);
        for (x, negative) in extra {
            (lo, hi) = self.add64(lo, hi, x, negative);
        }
        let (h, off) = self.memory(lo, hi, 4 * count, false, None);
        for k in 0..count {
            let four = self.s.m.i32(4 * k);
            let at = self.s.m.bin(self.cur, Bin::Add, off, four);
            let x = self.raw_dword(h, at);
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// `(hi:lo) + x` with `x` zero-extended, or sign-extended when
    /// `negative` says its top bit means it.
    pub(super) fn add64(&mut self, lo: V, hi: V, x: V, negative: bool) -> (V, V) {
        let c = self.cur;
        let sum = self.s.m.bin(c, Bin::Add, lo, x);
        let carry = self.s.m.cmp(c, pred::IULT, sum, lo);
        let carry = self.s.m.cast(c, Cast::ZExt, carry, self.s.t.i32);
        let hi = self.s.m.bin(c, Bin::Add, hi, carry);
        let hi = if negative {
            let all = self.s.m.i32(u32::MAX);
            self.s.m.bin(c, Bin::Add, hi, all)
        } else {
            hi
        };
        (sum, hi)
    }

    /// A 64-bit image atomic `op` of `value` (cmpswap: replacing
    /// `compare`) on the texel at `cs` of the globally coherent R32G32_UINT
    /// view `h`, under a lock word (`emit`, "Atomic locks"). The wave's
    /// lanes take turns in a loop over lane indices that every lane runs
    /// alike: in its turn a lane takes its texel's lock, reads, writes and
    /// frees it. Lanes of one wave never wait for each other's locks (that
    /// never finishes on Apple GPUs), and the loop needs no wave operation
    /// in divergent code (D3DMetal's return other lanes' values there).
    /// Returns the old value's low and high dwords.
    pub(super) fn locked_atomic64(
        &mut self,
        h: V,
        cs: &[V],
        coords: usize,
        op: &str,
        value: V,
        compare: Option<V>,
    ) -> (V, V) {
        let i32t = self.s.t.i32;
        let i64t = self.s.m.int(64);
        let void = self.s.t.void;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        self.s.flags |= flags::WAVE_OPS | flags::TYPED_UAV_LOADS | flags::INT64;
        let c = self.cur;
        // The lock: a hash of the coordinates.
        let mut hash = k(self, 0);
        for (&x, m) in cs
            .iter()
            .take(coords)
            .zip([0x9E37_79B1u32, 0x85EB_CA77, 0xC2B2_AE3D])
        {
            let m = k(self, m);
            let x = self.s.m.bin(c, Bin::Mul, x, m);
            hash = self.s.m.bin(c, Bin::Xor, hash, x);
            let k15 = k(self, 15);
            let r = self.s.m.bin(c, Bin::LShr, hash, k15);
            hash = self.s.m.bin(c, Bin::Xor, hash, r);
        }
        let mask = k(self, super::super::ATOMIC_LOCKS - 1);
        let slot = self.s.m.bin(c, Bin::And, hash, mask);
        let k2 = k(self, 2);
        let slot = self.s.m.bin(c, Bin::Shl, slot, k2);
        let base = k(self, super::super::ATOMIC_LOCKS_AT);
        let lock = self.s.m.bin(c, Bin::Add, slot, base);
        let locks = self.s.view_writes_handle();
        let me = self
            .s
            .op(c, "waveGetLaneIndex", Attr::ReadNone, i32t, 111, &[]);
        let lanes = self
            .s
            .op(c, "waveGetLaneCount", Attr::ReadNone, i32t, 112, &[]);
        let mut blocks = [0; 5];
        for b in &mut blocks {
            *b = self.s.m.new_block();
        }
        let [head, spin, held, latch, done] = blocks;
        self.s.m.terminate(c, Term::Br(head));
        let iu = self.s.m.undef(i32t);
        let (zero, one) = (k(self, 0), k(self, 1));
        let fence = k(self, 2 | 8);

        // Head: lane `i`'s turn.
        let i = self.s.m.phi(head, i32t);
        let plo = self.s.m.phi(head, i32t);
        let phi_ = self.s.m.phi(head, i32t);
        let turn = self.s.m.cmp(head, pred::EQ, me, i);
        self.s.m.terminate(head, Term::CondBr(turn, spin, latch));

        // Spin, bounded (past it, on without the lock).
        let got = self.s.op(
            spin,
            "atomicCompareExchange.i32",
            Attr::NoUnwind,
            i32t,
            79,
            &[locks, lock, iu, iu, zero, one],
        );
        let won = self.s.m.cmp(spin, pred::EQ, got, zero);
        let tries = self.s.m.phi(spin, i32t);
        let more = self.s.m.bin(spin, Bin::Add, tries, one);
        self.s.m.set_phi(tries, vec![(zero, head), (more, spin)]);
        let limit = k(self, 1 << 16);
        let out = self.s.m.cmp(spin, pred::IUGE, more, limit);
        let leave = self.s.m.bin(spin, Bin::Or, won, out);
        self.s.m.terminate(spin, Term::CondBr(leave, held, spin));

        // Held: fence, read, write, fence, free.
        self.s
            .op(held, "barrier", Attr::NoDuplicate, void, 80, &[fence]);
        let mut args = vec![h, iu];
        args.extend(cs);
        args.extend([iu, iu, iu]);
        let r = self.s.op(
            held,
            "textureLoad.i32",
            Attr::ReadOnly,
            self.s.t.res_ret_i32,
            66,
            &args,
        );
        let lo = self.s.m.extract(held, r, 0);
        let hi = self.s.m.extract(held, r, 1);
        let l64 = self.s.m.cast(held, Cast::ZExt, lo, i64t);
        let h64 = self.s.m.cast(held, Cast::ZExt, hi, i64t);
        let k32 = self.s.m.int_const(i64t, 32);
        let h64 = self.s.m.bin(held, Bin::Shl, h64, k32);
        let old = self.s.m.bin(held, Bin::Or, l64, h64);
        let m = &mut self.s.m;
        let pick = |m: &mut Module, p: u8| {
            let keep = m.cmp(held, p, old, value);
            m.select(held, keep, old, value)
        };
        let new = match op {
            "add" | "sub" => m.bin(held, Bin::Add, old, value),
            "and" => m.bin(held, Bin::And, old, value),
            "or" => m.bin(held, Bin::Or, old, value),
            "xor" => m.bin(held, Bin::Xor, old, value),
            "smin" => pick(m, pred::SLT),
            "smax" => pick(m, pred::SGT),
            "umin" => pick(m, pred::IULT),
            "umax" => pick(m, pred::IUGT),
            "swap" => value,
            _ => {
                let same = m.cmp(held, pred::EQ, old, compare.expect("a compare value"));
                m.select(held, same, value, old)
            }
        };
        let nlo = self.s.m.cast(held, Cast::Trunc, new, i32t);
        let nhi = self.s.m.bin(held, Bin::LShr, new, k32);
        let nhi = self.s.m.cast(held, Cast::Trunc, nhi, i32t);
        let mut args = vec![h];
        args.extend(cs);
        // DXIL stores all four channels; an R32G32 view keeps two.
        args.extend([nlo, nhi, nlo, nhi, self.s.m.i8(0xF)]);
        self.s
            .op(held, "textureStore.i32", Attr::NoUnwind, void, 67, &args);
        self.s
            .op(held, "barrier", Attr::NoDuplicate, void, 80, &[fence]);
        // Frees it only if taken: an `and` with all ones changes nothing.
        let (and, all) = (k(self, 1), k(self, u32::MAX));
        let keep = self.s.m.select(held, won, zero, all);
        self.s.op(
            held,
            "atomicBinOp.i32",
            Attr::NoUnwind,
            i32t,
            78,
            &[locks, and, lock, iu, iu, keep],
        );
        self.s.m.terminate(held, Term::Br(latch));

        // Latch: the next lane.
        let olo = self.s.m.phi(latch, i32t);
        self.s.m.set_phi(olo, vec![(lo, held), (plo, head)]);
        let ohi = self.s.m.phi(latch, i32t);
        self.s.m.set_phi(ohi, vec![(hi, held), (phi_, head)]);
        let next = self.s.m.bin(latch, Bin::Add, i, one);
        let more = self.s.m.cmp(latch, pred::IULT, next, lanes);
        self.s.m.set_phi(i, vec![(zero, c), (next, latch)]);
        self.s.m.set_phi(plo, vec![(iu, c), (olo, latch)]);
        self.s.m.set_phi(phi_, vec![(iu, c), (ohi, latch)]);
        self.s.m.terminate(latch, Term::CondBr(more, head, done));
        self.cur = done;
        (olo, ohi)
    }

    /// The dword at byte `at` of raw buffer `h`.
    pub(super) fn raw_dword(&mut self, h: V, at: V) -> V {
        let c = self.cur;
        let (u, mask, al) = (
            self.s.m.undef(self.s.t.i32),
            self.s.m.i8(1),
            self.s.m.i32(4),
        );
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[h, at, u, mask, al],
        );
        let x = self.s.m.extract(c, r, 0);
        self.as_int(x)
    }

    /// Marks the pages a write of `bytes` bytes at byte `x` of the unordered
    /// buffer view `h` touches, in the lanes `guard` lets through (module
    /// notes, "Write marks"): the bits of its first and last byte, each set
    /// only where still clear. Nothing for handles that are no heap view, or
    /// without [`Parts::view_marks`].
    pub(super) fn mark_view_write(&mut self, h: V, x: V, bytes: u32, guard: Option<V>) {
        if !self.x.view_marks {
            return;
        }
        let Some(&index) = self.s.view_index.get(&h) else {
            return;
        };
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let table = self.s.view_marks_handle();
        let c = self.cur;
        let (eight, four) = (k(self, 8), k(self, 4));
        let at = self.s.m.bin(c, Bin::Mul, index, eight);
        let base = self.raw_dword(table, at);
        let at = self.s.m.bin(c, Bin::Add, at, four);
        let phase = self.raw_dword(table, at);
        let none = k(self, super::super::VIEW_MARK_NONE);
        let valid = self.s.m.cmp(c, pred::NE, base, none);
        let active = match guard {
            Some(g) => self.s.m.bin(c, Bin::And, g, valid),
            None => valid,
        };
        let first = self.s.m.bin(c, Bin::Add, phase, x);
        let span = k(self, bytes.max(1) - 1);
        let last = self.s.m.bin(c, Bin::Add, first, span);
        let (k16, cap) = (k(self, 16), k(self, super::super::VIEW_WRITE_BITS - 1));
        let mut bits = Vec::new();
        for byte in [first, last] {
            let page = self.s.m.bin(c, Bin::LShr, byte, k16);
            let n = self.s.m.bin(c, Bin::Add, base, page);
            let over = self.s.m.cmp(c, pred::IUGT, n, cap);
            bits.push(self.s.m.select(c, over, cap, n));
        }
        let bitmap = self.s.view_writes_handle();
        let outer = self.begin_guarded(Some(active));
        for (i, &n) in bits.iter().enumerate() {
            let c = self.cur;
            let (k5, k2, k31, one, zero) =
                (k(self, 5), k(self, 2), k(self, 31), k(self, 1), k(self, 0));
            let word = self.s.m.bin(c, Bin::LShr, n, k5);
            let word = self.s.m.bin(c, Bin::Shl, word, k2);
            let bit = self.s.m.bin(c, Bin::And, n, k31);
            let bit = self.s.m.bin(c, Bin::Shl, one, bit);
            // Most writes find their page marked: a load, not an atomic.
            let old = self.raw_dword(bitmap, word);
            let set = self.s.m.bin(c, Bin::And, old, bit);
            let mut clear = self.s.m.cmp(c, pred::EQ, set, zero);
            if i == 1 {
                let other = self.s.m.cmp(c, pred::NE, n, bits[0]);
                clear = self.s.m.bin(c, Bin::And, clear, other);
            }
            let inner = self.begin_guarded(Some(clear));
            let iu = self.s.m.undef(self.s.t.i32);
            // Operation 2 is OR.
            self.s.op(
                self.cur,
                "atomicBinOp.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                78,
                &[bitmap, k2, word, iu, iu, bit],
            );
            self.end_guarded(inner, &[]);
        }
        self.end_guarded(outer, &[]);
    }

    /// The byte stride of the buffer view at heap index `index`, from the
    /// view-stride table (`t`[`VIEW_STRIDES`]).
    pub(super) fn stride_at(&mut self, index: V) -> V {
        let word = self.view_word(index);
        let bits = self.s.m.i32(VIEW_STRIDE_BITS);
        self.s.m.bin(self.cur, Bin::And, word, bits)
    }

    /// Dword `index` of the view-stride table: the stride and the format
    /// kind bits ([`VIEW_INTEGER`], [`VIEW_SIGNED`]).
    pub(super) fn view_word(&mut self, index: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let at = self.s.m.bin(c, Bin::Mul, index, four);
        let table = self.s.root_srv_handle(VIEW_STRIDES);
        self.raw_dword(table, at)
    }

    /// Whether the typed buffer view a handle names has an integer format,
    /// and whether a signed one, from the view-stride table; `None` if the
    /// handle's heap index is not known.
    pub(super) fn integer_view(&mut self, h: V) -> Option<(V, V)> {
        let index = *self.s.view_index.get(&h)?;
        let word = self.view_word(index);
        let c = self.cur;
        let zero = self.s.m.i32(0);
        let [int, signed] = [VIEW_INTEGER, VIEW_SIGNED].map(|bit| {
            let bit = self.s.m.i32(bit);
            let x = self.s.m.bin(c, Bin::And, word, bit);
            self.s.m.cmp(c, pred::NE, x, zero)
        });
        Some((int, signed))
    }

    /// Title memory at a raw GPU address (decision 0005: the unified-memory
    /// item). Every access to an address the program computes goes through
    /// here: scalar and global loads, stores and atomics, the BVH node
    /// fetch, and in a program that uses title memory its root SRVs and UAVs
    /// ([`Self::raw_view`]). `(hi:lo)` is the 64-bit address of the access's
    /// first byte (`hi` holds bits 32-47), `bytes` its size, `write` whether
    /// it stores or does atomics, `active` the lanes that access (`None`:
    /// all; the others look up but mark nothing). Returns a raw buffer handle
    /// (read-write with `write`) and the address's byte offset in it; the
    /// access's dwords follow at `+ 4k`. Memory no host copy holds is the
    /// null view: reads give 0, writes are dropped. Contract (`emit` "Title
    /// memory"): a page table in the root SRV `t0` of
    /// [`super::super::shader::MEMORY_SPACE`]; the root UAV `u0` there takes a bit
    /// per page written and per page or 256 MiB region missed. May end in a
    /// new block: callers take [`Lower::cur`] afresh.
    // shortcut: one lookup per access, so its bytes are taken to lie in one
    // host copy of title memory (any access within a 64 KiB page does).
    pub(crate) fn memory(
        &mut self,
        lo: V,
        hi: V,
        bytes: u32,
        write: bool,
        active: Option<V>,
    ) -> (V, V) {
        let e = self.memory_entry(lo, hi, write);
        let c = self.cur;
        // Writes always mark (written or missed pages), reads only misses.
        let marks = match (write, active) {
            (true, Some(a)) => Some(a),
            (true, None) => None,
            (false, Some(a)) => Some(self.s.m.bin(c, Bin::And, e.miss, a)),
            (false, None) => Some(e.miss),
        };
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        // A miss marks its page's entry, or with no block its region's
        // level-1 index (bits 0-4095, block 0's entries, which no copy has).
        let zero = k(self, 0);
        let empty = self.s.m.cmp(c, pred::EQ, e.block, zero);
        let missed = self.s.m.select(c, empty, e.l1, e.entry);
        let first = self.s.m.select(c, e.miss, missed, e.entry);
        let mut bits = vec![first];
        if write {
            // A write marks the pages of its first and last byte (one block).
            let (last, k16) = (k(self, bytes - 1), k(self, 16));
            let end = self.s.m.bin(c, Bin::Add, lo, last);
            let a = self.s.m.bin(c, Bin::LShr, lo, k16);
            let b = self.s.m.bin(c, Bin::LShr, end, k16);
            let pages = self.s.m.bin(c, Bin::Sub, b, a);
            let next = self.s.m.bin(c, Bin::Add, e.entry, pages);
            bits.push(self.s.m.select(c, e.miss, first, next));
        }
        let bitmap = self.s.memory_dirty_handle();
        let at = self.begin_guarded(marks);
        let c = self.cur;
        let (k5, k31, k2, one) = (k(self, 5), k(self, 31), k(self, 2), k(self, 1));
        let iu = self.s.m.undef(self.s.t.i32);
        for n in bits {
            // Byte offset of the bitmap dword; operation 2 is OR.
            let word = self.s.m.bin(c, Bin::LShr, n, k5);
            let word = self.s.m.bin(c, Bin::Shl, word, k2);
            let bit = self.s.m.bin(c, Bin::And, n, k31);
            let bit = self.s.m.bin(c, Bin::Shl, one, bit);
            self.s.op(
                c,
                "atomicBinOp.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                78,
                &[bitmap, k2, word, iu, iu, bit],
            );
        }
        self.end_guarded(at, &[]);
        (e.handle, e.offset)
    }

    /// [`Self::memory`]'s lookup in the page table.
    pub(super) fn memory_entry(&mut self, lo: V, hi: V, write: bool) -> MemoryEntry {
        let c = self.cur;
        let map = self.s.memory_handle();
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        // Level 1: (hi:lo) >> 28, 4096 entries for 40-bit addresses.
        let (k4, k28, k256) = (k(self, 4), k(self, 28), k(self, 256));
        let top = self.s.m.bin(c, Bin::Shl, hi, k4);
        let low = self.s.m.bin(c, Bin::LShr, lo, k28);
        let l1 = self.s.m.bin(c, Bin::Or, top, low);
        let inside = self.s.m.cmp(c, pred::IULT, hi, k256);
        let zero = k(self, 0);
        let l1 = self.s.m.select(c, inside, l1, zero);
        let two = k(self, 2);
        let at = self.s.m.bin(c, Bin::Shl, l1, two);
        let block = self.raw_dword(map, at);
        let block = self.s.m.select(c, inside, block, zero);
        // Level 2: block b holds 4096 entries of two dwords after level 1.
        let (k16, k12, mask, k3, first) = (
            k(self, 16),
            k(self, 12),
            k(self, 0xFFF),
            k(self, 3),
            k(self, 4096 * 4),
        );
        let page = self.s.m.bin(c, Bin::LShr, lo, k16);
        let page = self.s.m.bin(c, Bin::And, page, mask);
        let entry = self.s.m.bin(c, Bin::Shl, block, k12);
        let entry = self.s.m.bin(c, Bin::Or, entry, page);
        let at = self.s.m.bin(c, Bin::Shl, entry, k3);
        let at = self.s.m.bin(c, Bin::Add, at, first);
        let view = self.raw_dword(map, at);
        let four = k(self, 4);
        let at = self.s.m.bin(c, Bin::Add, at, four);
        let start = self.raw_dword(map, at);
        // Block 0's first entry names the null pair: no copy holds the page.
        let null = self.raw_dword(map, first);
        let miss = self.s.m.cmp(c, pred::EQ, view, null);
        // Views come in pairs: the raw SRV, then the raw UAV of one copy.
        let (view, props) = if write {
            let one = k(self, 1);
            (self.s.m.bin(c, Bin::Add, view, one), (11 | 0x1000, 0))
        } else {
            (view, (11, 0))
        };
        let handle = self.s.heap_handle(c, view, false, true, props);
        // Pages written here are marked in the page table's bitmap.
        self.s.view_index.remove(&handle);
        MemoryEntry {
            handle,
            offset: self.s.m.bin(c, Bin::Sub, lo, start),
            entry,
            block,
            l1,
            miss,
        }
    }

    /// A descriptor load from a table or heap whose dwords the program only
    /// reads as numbers: the base address of a buffer view (dwords 0 and 1,
    /// as a pointer to title memory: the BVH's address in its SRV) or, for
    /// a whole image descriptor (eight dwords), also the size fields of
    /// dwords 1-3 (width, height, first and last level, type: size queries
    /// of a texture the program never samples). They come from the host's
    /// view addresses, `t1` in [`super::super::shader::MEMORY_SPACE`]: four dwords
    /// per heap index.
    // shortcut: dword 1 holds the address bits only (stride and swizzle read
    // 0) or a texture's width bits, dword 3 a texture's size fields (format,
    // tiling and swizzle choice read 0 or the identity); dword 4 on and a
    // buffer's dwords 2-3 (record count, format) stop the program when read;
    // the table offset's computed part is taken as whole descriptors
    // (32-byte multiples).
    pub(super) fn view_addresses(
        &mut self,
        inst: &Inst,
        site: Site,
        code: u8,
        count: u32,
    ) -> Result<(), Why> {
        const OTHER: Why = "a descriptor read only as data";
        let a = *self
            .scalar
            .get(&site)
            .ok_or("a descriptor read only as data")?;
        if !matches!(
            self.x.native.layout.get(a.param).map(|p| p.root),
            Some(RootType::Table | RootType::ResourceHeap)
        ) {
            return Err("a descriptor read only as data");
        }
        // The compiler loads image descriptors whole, buffer ones as four
        // dwords.
        let known = if count == 8 && a.offset.is_multiple_of(32) {
            4
        } else {
            2
        };
        let views = self.s.view_addresses_handle();
        for k in 0..count {
            let word = (a.offset / 4 + k) % 8;
            let val = if word < known {
                let index = self.heap_index(site, k as u8, false)?;
                let c = self.cur;
                let four = self.s.m.i32(4);
                let at = self.s.m.bin(c, Bin::Shl, index, four);
                let w = self.s.m.i32(4 * word);
                let at = self.s.m.bin(c, Bin::Add, at, w);
                Val::I32(self.raw_dword(views, at))
            } else {
                Val::Missing(OTHER)
            };
            if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                self.vals.insert(v, val);
            }
        }
        Ok(())
    }

    /// `s_load_dword*` from the program's constant table.
    pub(super) fn code_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar load from the code"),
        };
        let c = self.cur;
        let mut addr = self.s.m.i32(a.offset);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            addr = self.s.m.bin(c, Bin::Add, addr, d);
        }
        for k in 0..count {
            let off = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, addr, off);
            let x = self.code_dword(at)?;
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// The dword at a computed byte address of a constant buffer.
    pub(super) fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }

    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned.
    pub(super) fn format_load(
        &mut self,
        h: V,
        off: V,
        format: u8,
        channels: u32,
    ) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, at, u, mask, align],
            );
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else {
            load(self, off, total / 32)
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            // A normalized format's ends are exactly 1.0 and -1.0. The host
            // may divide through a reciprocal and land just below 1.0: RDNA
            // 4 loaded 16_16_UNORM's 65535 as the half 0x3BFF, not 0x3C00.
            let x = match numeric {
                Numeric::Unorm => {
                    let top = (1u64 << w) - 1;
                    let q = if w <= 16 {
                        let q = self.norm_quotient(field, w);
                        self.as_float(q)
                    } else {
                        let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                        let max = self.s.m.f32((top as f32).to_bits());
                        self.s.m.bin(c, Bin::SDiv, f, max)
                    };
                    let (high, one) = (self.s.m.i32(top as u32), self.s.m.f32(1.0f32.to_bits()));
                    let at_top = self.s.m.cmp(c, pred::EQ, field, high);
                    let q = self.s.m.select(c, at_top, one, q);
                    float(self, q)
                }
                Numeric::Snorm => {
                    let s = signed(self);
                    let top = (1u32 << (w - 1)) - 1;
                    let q = if w <= 17 {
                        // The magnitude's quotient, signed (codes at and
                        // below -top are -1.0 below).
                        let (zero, sign) = (self.s.m.i32(0), self.s.m.i32(0x8000_0000));
                        let neg = self.s.m.cmp(c, pred::SLT, s, zero);
                        let minus = self.s.m.bin(c, Bin::Sub, zero, s);
                        let m = self.s.m.select(c, neg, minus, s);
                        let q = self.norm_quotient(m, w - 1);
                        let sign = self.s.m.select(c, neg, sign, zero);
                        let q = self.s.m.bin(c, Bin::Or, q, sign);
                        self.as_float(q)
                    } else {
                        let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                        let max = self.s.m.f32((top as f32).to_bits());
                        self.s.m.bin(c, Bin::SDiv, f, max)
                    };
                    // The lowest value is -1.0 too, as is the one above it.
                    let (high, one) = (self.s.m.i32(top), self.s.m.f32(1.0f32.to_bits()));
                    let at_top = self.s.m.cmp(c, pred::SGE, s, high);
                    let q = self.s.m.select(c, at_top, one, q);
                    let low = self.s.m.i32(top.wrapping_neg());
                    let minus_one = self.s.m.f32((-1.0f32).to_bits());
                    let at_bottom = self.s.m.cmp(c, pred::SLE, s, low);
                    let q = self.s.m.select(c, at_bottom, minus_one, q);
                    float(self, q)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let h = match half_shift(w) {
                        0 => field,
                        k => {
                            let sh = self.s.m.i32(k);
                            self.s.m.bin(c, Bin::Shl, field, sh)
                        }
                    };
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    float(self, f)
                }
            };
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    /// The bits of the float nearest to `a / d`, `d = 2^w - 1`, for codes
    /// `0 <= a < d` and `w <= 16`: what the console's buffer unit gives for
    /// a normalized code. The host's division is not exact (the RX 6700 XT
    /// returns 8-bit codes one ulp high and some 16-bit codes one ulp low),
    /// and titles truncate `x * d` back to the code (a material index in a
    /// vertex colour's alpha), which a quotient one ulp low turns into the
    /// code below.
    ///
    /// `Q = floor(a * 2^32 / d) = a * K + floor(a * r / d)` with `K =
    /// floor(2^32 / d)` and `r = 2^32 mod d` (no division when `r` is 1:
    /// widths dividing 32). `a / d` is never exact for `0 < a < d` (`d`
    /// odd, `r` coprime to it), so `Q` rounds to 24 bits up exactly when
    /// its highest dropped bit is set, and the result is that integer (at
    /// most 2^24, so its conversion is exact on any host) scaled by a power
    /// of two; `Q` itself when it fits, the quotient's further bits then
    /// lying below half an ulp. That holds for every code of every width but
    /// 11 and 12 (checked for all codes of widths 2 to 16), which take
    /// [`Self::exact_norm`]. The host's conversion of a larger integer is
    /// not relied on: WARP rounds those above 2^31 otherwise.
    pub(super) fn norm_quotient(&mut self, a: V, w: u32) -> V {
        if matches!(w, 11 | 12) {
            return self.exact_norm(a, w);
        }
        let c = self.cur;
        let d = (1u64 << w) - 1;
        let (kk, r) = ((1u64 << 32) / d, (1u64 << 32) % d);
        let k = |l: &mut Self, v: u64| l.s.m.i32(v as u32);
        let (one, kv) = (k(self, 1), k(self, kk));
        let mut q = self.s.m.bin(c, Bin::Mul, a, kv);
        if r != 1 {
            let (rv, dv) = (k(self, r), k(self, d));
            let t = self.s.m.bin(c, Bin::Mul, a, rv);
            let more = self.s.m.bin(c, Bin::UDiv, t, dv);
            q = self.s.m.bin(c, Bin::Add, q, more);
        }
        // The bits to drop (1 to 8) from the exponent of `q >> 8`, which
        // converts exactly (fewer than 25 bits); none when `q` fits.
        let k8 = k(self, 8);
        let top = self.s.m.bin(c, Bin::LShr, q, k8);
        let top = self.s.m.cast(c, Cast::UiToFp, top, self.s.t.f32);
        let top = self.as_int(top);
        let k23 = k(self, 23);
        let e = self.s.m.bin(c, Bin::LShr, top, k23);
        let bias = k(self, 127 + 15);
        let shift = self.s.m.bin(c, Bin::Sub, e, bias);
        let (zero, b24) = (k(self, 0), k(self, 1 << 24));
        let fits = self.s.m.cmp(c, pred::IULT, q, b24);
        let shift = self.s.m.select(c, fits, zero, shift);
        // Rounded up when the highest dropped bit is set: `a / d` is never
        // exact, so a dropped half is more than half, never a tie.
        let up = self.s.m.bin(c, Bin::Shl, q, one);
        let up = self.s.m.bin(c, Bin::LShr, up, shift);
        let up = self.s.m.bin(c, Bin::And, up, one);
        let m = self.s.m.bin(c, Bin::LShr, q, shift);
        let m = self.s.m.bin(c, Bin::Add, m, up);
        // At most 2^24: exact as a float, and so its scale by 2^(shift - 32).
        let f = self.s.m.cast(c, Cast::UiToFp, m, self.s.t.f32);
        let b95 = k(self, 127 - 32);
        let scale = self.s.m.bin(c, Bin::Add, shift, b95);
        let scale = self.s.m.bin(c, Bin::Shl, scale, k23);
        let scale = self.as_float(scale);
        let f = self.s.m.bin(c, Bin::Mul, f, scale);
        self.as_int(f)
    }

    /// [`Self::norm_quotient`] in integers only, for any width up to 16: `Q`
    /// as there, 8 more bits when that leaves fewer than 25 (codes of wide
    /// formats near 0), then rounded to 24 bits to nearest even with the
    /// remainder as sticky bit, and assembled as float bits.
    pub(super) fn exact_norm(&mut self, a: V, w: u32) -> V {
        let c = self.cur;
        let d = (1u64 << w) - 1;
        let (kk, r) = ((1u64 << 32) / d, (1u64 << 32) % d);
        let k = |l: &mut Self, v: u64| l.s.m.i32(v as u32);
        let (one, zero, dv, k8) = (k(self, 1), k(self, 0), k(self, d), k(self, 8));
        let kv = k(self, kk);
        let mut q = self.s.m.bin(c, Bin::Mul, a, kv);
        let mut rem = a;
        if r != 1 {
            let rv = k(self, r);
            let t = self.s.m.bin(c, Bin::Mul, a, rv);
            let more = self.s.m.bin(c, Bin::UDiv, t, dv);
            q = self.s.m.bin(c, Bin::Add, q, more);
            rem = self.s.m.bin(c, Bin::URem, t, dv);
        }
        // The exponent's bias less the 32 (or 40) fraction bits of `q`; the
        // shift below is added.
        let mut bias = k(self, 127 + 23 - 32);
        if kk < 1 << 24 {
            // Fewer than 25 bits for the smallest codes: 8 more.
            let t = self.s.m.bin(c, Bin::Shl, rem, k8);
            let more = self.s.m.bin(c, Bin::UDiv, t, dv);
            let rest = self.s.m.bin(c, Bin::URem, t, dv);
            let wide = self.s.m.bin(c, Bin::Shl, q, k8);
            let wide = self.s.m.bin(c, Bin::Or, wide, more);
            let limit = k(self, 1 << 24);
            let small = self.s.m.cmp(c, pred::IULT, q, limit);
            q = self.s.m.select(c, small, wide, q);
            rem = self.s.m.select(c, small, rest, rem);
            let b40 = k(self, 127 + 23 - 40);
            bias = self.s.m.select(c, small, b40, bias);
        }
        // Bits to drop: the bit length less 24, 1 to 8.
        let mut shift = one;
        for bit in 25..32 {
            let at = k(self, 1 << bit);
            let ge = self.s.m.cmp(c, pred::IUGE, q, at);
            let add = self.s.m.select(c, ge, one, zero);
            shift = self.s.m.bin(c, Bin::Add, shift, add);
        }
        let mant = self.s.m.bin(c, Bin::LShr, q, shift);
        let below = self.s.m.bin(c, Bin::Sub, shift, one);
        let half = self.s.m.bin(c, Bin::LShr, q, below);
        let half = self.s.m.bin(c, Bin::And, half, one);
        let low = self.s.m.bin(c, Bin::Shl, one, below);
        let low = self.s.m.bin(c, Bin::Sub, low, one);
        let low = self.s.m.bin(c, Bin::And, q, low);
        let rest = self.s.m.bin(c, Bin::Or, low, rem);
        let sticky = self.s.m.cmp(c, pred::NE, rest, zero);
        let odd = self.s.m.bin(c, Bin::And, mant, one);
        let tie = self.s.m.select(c, sticky, one, odd);
        let up = self.s.m.bin(c, Bin::And, half, tie);
        let mant = self.s.m.bin(c, Bin::Add, mant, up);
        // Rounding up to 2^24 moves to the next exponent.
        let k24 = k(self, 24);
        let carry = self.s.m.bin(c, Bin::LShr, mant, k24);
        let mant = self.s.m.bin(c, Bin::LShr, mant, carry);
        let e = self.s.m.bin(c, Bin::Add, bias, shift);
        let e = self.s.m.bin(c, Bin::Add, e, carry);
        let k23 = k(self, 23);
        let e = self.s.m.bin(c, Bin::Shl, e, k23);
        let fraction = k(self, 0x7F_FFFF);
        let m = self.s.m.bin(c, Bin::And, mant, fraction);
        let bits = self.s.m.bin(c, Bin::Or, e, m);
        let is_zero = self.s.m.cmp(c, pred::EQ, a, zero);
        self.s.m.select(c, is_zero, zero, bits)
    }

    /// Channels of buffer format `format`, as [`Self::format_load`] gives
    /// them, as d16 data: 16 bits each, two per dword, low half first (an
    /// odd last high half 0). Integer formats keep their low 16 bits; the
    /// others become f16.
    pub(super) fn d16_channels(&mut self, format: u8, channels: Vec<V>) -> Result<Vec<V>, Why> {
        let (_, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let mut halves = Vec::new();
        for x in channels {
            halves.push(if matches!(numeric, Numeric::Uint | Numeric::Sint) {
                let m = self.s.m.i32(0xFFFF);
                self.s.m.bin(c, Bin::And, x, m)
            } else {
                let (f, i32t) = (self.as_float(x), self.s.t.i32);
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            });
        }
        let mut out = Vec::new();
        for pair in halves.chunks(2) {
            out.push(match *pair {
                [lo, hi] => {
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                }
                [lo] => lo,
                _ => unreachable!("chunks of one or two"),
            });
        }
        Ok(out)
    }

    /// The byte offset a buffer descriptor built around a root descriptor adds
    /// (constant and computed parts); none for other descriptors.
    pub(super) fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<V>), Why> {
        match src {
            Source::Built { address, .. } => {
                let x = match address.dynamic {
                    Some(v) => Some(self.i32_of(v)?),
                    None => None,
                };
                Ok((address.offset, x))
            }
            _ => Ok((0, None)),
        }
    }

    /// The byte stride of the buffer view a heap-indexed descriptor names
    /// (descriptors from the resource heap only): dword `index` of the
    /// host's view-stride table, root SRV [`VIEW_STRIDES`].
    pub(super) fn view_stride(&mut self, src: &Source) -> Result<V, Why> {
        const WHY: Why = "a structured buffer of unknown stride";
        let heap = |load: &Site| {
            self.scalar
                .get(load)
                .and_then(|a| self.x.native.layout.get(a.param))
                .is_some_and(|p| p.root == RootType::ResourceHeap)
        };
        let leaves = src.leaves();
        if leaves.is_empty()
            || !leaves.iter().all(|l| match l {
                Source::Loaded { load, .. } | Source::Fetched { load, .. } => heap(load),
                _ => false,
            })
        {
            return Err(WHY);
        }
        let index = match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, false)?,
            Source::Phi { .. } => return Err(WHY),
            _ => self.lane_heap_index(src, false)?,
        };
        let c = self.cur;
        let four = self.s.m.i32(4);
        let at = self.s.m.bin(c, Bin::Mul, index, four);
        let table = self.s.root_srv_handle(VIEW_STRIDES);
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(1);
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[table, at, iu, mask, four],
        );
        let x = self.s.m.extract(c, r, 0);
        let word = self.as_int(x);
        let bits = self.s.m.i32(VIEW_STRIDE_BITS);
        Ok(self.s.m.bin(c, Bin::And, word, bits))
    }

    /// The used-resource entry a descriptor loaded from a table falls in.
    pub(super) fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
        if let Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } =
            src
        {
            return self.used_kind(args.first()?);
        }
        if let Source::LanePhi { args, .. } = src {
            return args.iter().flatten().find_map(|a| self.used_kind(a));
        }
        if let Source::Built { address, .. } = src {
            let rp = self.x.native.layout.get(address.param)?;
            return self
                .x
                .native
                .used
                .iter()
                .find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data);
        }
        let (Source::Loaded { load, first } | Source::Fetched { load, first }) = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table {
            return None;
        }
        let dword = (a.offset / 4) + u32::from(*first);
        self.x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)
    }

    /// A handle for the descriptor `src` read at `site`; the heap index of
    /// a texture shader view the program takes from the heap itself is noted
    /// for the instruction's read mark (handles are made once per block,
    /// often before the instruction). A view read through a table needs
    /// none: the host makes the slot for the draw.
    pub(super) fn handle(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        let h = self.handle_made(site, src, sampler, props)?;
        if let Some(&index) = self.s.texture_index.get(&h)
            && self.is_heap_direct(src)
        {
            self.s.texture_reads.push(index);
        }
        Ok(h)
    }

    pub(super) fn handle_made(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        let props = if self.pops && props.0 & 0x1000 != 0 {
            self.s.flags |= flags::ROVS;
            (props.0 | 0x2000, props.1)
        } else {
            props
        };
        // A descriptor from the resource heap itself: the host cannot tell
        // which slots the program reads, or writes through unordered access.
        if !sampler && self.is_heap_direct(src) {
            self.heap[0] = true;
            self.heap[1] |= props.0 & 0x1000 != 0;
        }
        match src {
            Source::Loaded { load, first } if let Some(address) = self.entry_address(*load) => {
                // The same descriptor wherever it is loaded: the console
                // compiler reloads one before every use rather than keep it
                // (ActivateRays: 2,399 loads of one), and one handle made in
                // the entry block serves them all.
                let key = (address, *first, sampler, props.0);
                if let Some(&h) = self.entry_handles.get(&key) {
                    return Ok(h);
                }
                let here = std::mem::replace(&mut self.cur, self.s.entry);
                let index = self.entry_heap_index(address, *first, sampler);
                let index = self.buffer_guard(src, index, sampler, props);
                let h = self.s.heap_handle(self.cur, index, sampler, false, props);
                self.cur = here;
                self.entry_handles.insert(key, h);
                Ok(h)
            }
            Source::Loaded { load, first } => {
                let key = (self.cur, *load, *first, sampler, props.0);
                if let Some(&h) = self.handles.get(&key) {
                    return Ok(h);
                }
                let index = self.heap_index(*load, *first, sampler)?;
                let index = self.buffer_guard(src, index, sampler, props);
                // A waterfall loop's load at the first lane's offset reads
                // each lane's own entry (see `resource::Waterfall`).
                let per_lane = self.waterfall_loads.contains(load);
                let h = self
                    .s
                    .heap_handle(self.cur, index, sampler, per_lane, props);
                self.handles.insert(key, h);
                Ok(h)
            }
            Source::Constant(words) if sampler => {
                let words: [u32; 4] = words
                    .get(..4)
                    .and_then(|w| w.try_into().ok())
                    .ok_or("a sampler constant")?;
                let comparison = props.0 & 0x8000 != 0;
                let slot = self.samplers.iter().enumerate().position(|(n, w)| {
                    *w == words && (self.s.cmp_samplers >> n & 1 != 0) == comparison
                });
                let n = match slot {
                    Some(n) => n,
                    None => {
                        let n = self.samplers.len();
                        if n >= u64::BITS as usize {
                            return Err("more than 64 immediate sampler slots");
                        }
                        self.samplers.push(words);
                        self.s.cmp_samplers |= u64::from(comparison) << n;
                        n
                    }
                };
                Ok(self.s.sampler_handle(self.cur, n as u32))
            }
            Source::Constant(_) => Err("a constant resource descriptor"),
            Source::Built { address, .. } if address.param == MEMORY => {
                Err("a buffer descriptor built around a computed address")
            }
            Source::Built { address, .. } => {
                let rp = *self
                    .x
                    .native
                    .layout
                    .get(address.param)
                    .ok_or("a descriptor built around a table in the code")?;
                let kind = self
                    .x
                    .native
                    .used
                    .iter()
                    .find(|u| u.user_data == rp.user_data)
                    .map(|u| u.kind);
                match (rp.root, kind, props.0) {
                    (RootType::Descriptor, Some(ResourceType::Cbv) | None, 13) => {
                        Ok(self.s.root_cbv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) if !self.root_memory => {
                        Ok(self.root_view(address.param, false))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) => {
                        Ok(self.s.root_srv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Uav) | None, 0x100B)
                    | (RootType::Descriptor, Some(ResourceType::Uav), 11)
                        if !self.root_memory =>
                    {
                        Ok(self.root_view(address.param, true))
                    }
                    (RootType::Descriptor, Some(ResourceType::Uav) | None, 0x100B)
                    | (RootType::Descriptor, Some(ResourceType::Uav), 11) => {
                        Ok(self.s.root_uav_handle(rp.slot))
                    }
                    _ => Err("a root UAV or typed root view"),
                }
            }
            Source::Phi { block, args }
                if sampler && matches!(args[..], [Source::Constant(_), Source::Constant(_)]) =>
            {
                // One of two immediate samplers, chosen by uniform control
                // flow: what reads it runs with both, the edge taken picks
                // the result (static samplers cannot be indexed).
                let [a, b] = &args[..] else { unreachable!() };
                let ha = self.handle(site, a, sampler, props)?;
                let hb = self.handle(site, b, sampler, props)?;
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let mut incoming = Vec::new();
                for (j, &q) in self.x.p.blocks[*block].preds.iter().enumerate() {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail = self.tail[q].ok_or("a descriptor chosen by control flow")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    incoming.push((self.s.m.bool(j == 0), at));
                }
                let phi = self.s.m.phi(join, self.s.t.i1);
                self.s.m.set_phi(phi, incoming);
                self.s.sampler_choices.push((ha, hb, phi));
                Ok(ha)
            }
            Source::Phi { block, .. } => {
                // One handle at the join for every access the choice reaches
                // (each is in a block the join dominates).
                let _ = site;
                let key = (format!("{src:?}"), sampler, props.0);
                if let Some(&h) = self.join_handles.get(&key) {
                    return Ok(h);
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let here = std::mem::replace(&mut self.cur, join);
                let index = self.uniform_heap_index(src, sampler);
                let h = index.map(|index| {
                    let index = self.buffer_guard(src, index, sampler, props);
                    self.s.heap_handle(self.cur, index, sampler, false, props)
                });
                self.cur = here;
                let h = h?;
                self.join_handles.insert(key, h);
                Ok(h)
            }
            Source::Select { scc, args } => {
                // The heap index of each choice, picked by SCC (uniform).
                let [
                    Source::Loaded { load: a, first: fa },
                    Source::Loaded { load: b, first: fb },
                ] = args[..]
                else {
                    return Err("a descriptor chosen by a condition");
                };
                let cond = match self.get(*scc)? {
                    Val::Bit(x) => x,
                    _ => return Err("SCC without a value"),
                };
                let set = self.heap_index(a, fa, sampler)?;
                let clear = self.heap_index(b, fb, sampler)?;
                let index = self.s.m.select(self.cur, cond, set, clear);
                let index = self.buffer_guard(src, index, sampler, props);
                Ok(self.s.heap_handle(self.cur, index, sampler, false, props))
            }
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => {
                // A descriptor each lane has its own of (a waterfall loop
                // reads it from the first active lane): each lane's heap
                // index, non-uniform.
                let index = self.lane_heap_index(src, sampler)?;
                let index = self.buffer_guard(src, index, sampler, props);
                Ok(self.s.heap_handle(self.cur, index, sampler, true, props))
            }
        }
    }

    /// The heap index to read a buffer view at, for a descriptor `src` from
    /// the resource heap itself (module notes, "Buffer views at computed
    /// heap indices"): `index`, or the host's null raw buffer view when the
    /// view-stride table says the slot holds no buffer view. Indices come
    /// from data; one naming a texture slot read a texture descriptor as a
    /// buffer one (a wild address: the texture's address shifted 8 bits
    /// down) and faulted the device.
    pub(super) fn buffer_guard(
        &mut self,
        src: &Source,
        index: V,
        sampler: bool,
        props: (u32, u32),
    ) -> V {
        // Typed, raw and structured buffers (DXIL resource kinds 10-12).
        if sampler || !(10..=12).contains(&(props.0 & 0xFF)) || !self.is_heap_direct(src) {
            return index;
        }
        let c = self.cur;
        let last = self.s.m.i32(super::super::HEAP_VIEWS - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let four = self.s.m.i32(4);
        let at = self.s.m.bin(c, Bin::Mul, index, four);
        let table = self.s.root_srv_handle(VIEW_STRIDES);
        let (iu, mask) = (self.s.m.undef(self.s.t.i32), self.s.m.i8(1));
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[table, at, iu, mask, four],
        );
        let x = self.s.m.extract(c, r, 0);
        let stride = self.as_int(x);
        let zero = self.s.m.i32(0);
        let none = self.s.m.cmp(c, pred::EQ, stride, zero);
        // The null unordered access view sits just below the null SRV.
        let null = super::super::NULL_BUFFER_VIEW - u32::from(props.0 & 0x1000 != 0);
        let null = self.s.m.i32(null);
        self.s.m.select(c, none, null, index)
    }

    /// Whether a descriptor comes from the resource heap itself (root type
    /// 5, not a table) on some path.
    pub(super) fn is_heap_direct(&self, src: &Source) -> bool {
        src.leaves().iter().any(|l| match l {
            Source::Loaded { load, .. } | Source::Fetched { load, .. } => self
                .scalar
                .get(load)
                .and_then(|a| self.x.native.layout.get(a.param))
                .is_some_and(|p| p.root == RootType::ResourceHeap),
            _ => false,
        })
    }

    /// The address of the descriptor load at `load` when root parameters
    /// alone give it (a table or a heap, at a fixed offset: resource
    /// tracking found no other term), and not per lane.
    pub(super) fn entry_address(&self, load: Site) -> Option<EntryAddress> {
        let a = self.scalar.get(&load)?;
        let fixed = a.dynamic.is_none() && !self.x.res.waterfall.loads.contains(&load);
        let root = self.x.native.layout.get(a.param)?.root;
        let heap = matches!(
            root,
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap
        );
        (fixed && heap).then_some((a.param, a.offset))
    }

    /// [`Lower::heap_index`] of a descriptor at a fixed address
    /// ([`Lower::entry_address`]), in the entry block.
    pub(super) fn entry_heap_index(
        &mut self,
        (param, offset): EntryAddress,
        first: u8,
        sampler: bool,
    ) -> V {
        let entry = self.s.entry;
        let base = match self.x.native.layout[param].root {
            RootType::Table => self.user_dword(param, 0),
            _ => self.s.m.i32(0),
        };
        let k = self.s.m.i32(offset.wrapping_add(4 * u32::from(first)));
        let addr = self.s.m.bin(entry, Bin::Add, base, k);
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        self.s.m.bin(entry, Bin::LShr, addr, shift)
    }

    /// The heap index of a descriptor every lane shares, computed in the
    /// current block. One chosen by control flow is a phi at its join of the
    /// index each predecessor computes at its end; an edge may bring another
    /// join's (nested if/else).
    pub(super) fn uniform_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Phi { block, args } => {
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail = self.tail[q].ok_or("a descriptor chosen by control flow")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    // No lane comes this way (`resource` gave it another
                    // edge's source): any index will do.
                    if self.never_taken(q, *block) {
                        incoming.push((self.s.m.i32(0), at));
                        continue;
                    }
                    let here = std::mem::replace(&mut self.cur, at);
                    let index = self.uniform_heap_index(arg, sampler);
                    self.cur = here;
                    incoming.push((index?, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a descriptor chosen by control flow"),
        }
    }

    /// The heap index of a descriptor that may differ between lanes,
    /// computed in the current block (or, chosen by control flow, at the
    /// join).
    pub(super) fn lane_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Fetched { load, first } => self.fetched_index(*load, *first, sampler),
            Source::Lanes { mask, args } | Source::Select { scc: mask, args } => {
                let [a, b] = &args[..] else {
                    return Err("a descriptor chosen per lane");
                };
                let cond = if matches!(src, Source::Lanes { .. }) {
                    let &m = self
                        .x
                        .st
                        .masks
                        .get(mask)
                        .ok_or("a descriptor chosen per lane by an untracked mask")?;
                    self.boolean(m, self.cur)?
                } else {
                    match self.get(*mask)? {
                        Val::Bit(x) => x,
                        _ => return Err("SCC without a value"),
                    }
                };
                let set = self.lane_heap_index(a, sampler)?;
                let clear = self.lane_heap_index(b, sampler)?;
                Ok(self.s.m.select(self.cur, cond, set, clear))
            }
            Source::LanePhi { block, args } => {
                // Each predecessor computes its heap index at its end (what
                // it reads is there: the dwords are, on that edge); the join
                // picks the one of the edge the lane came by. An edge no
                // lane reading it takes gives index 0.
                let join =
                    self.block[*block].ok_or("a per-lane descriptor chosen by control flow")?;
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail =
                        self.tail[q].ok_or("a per-lane descriptor joined from a later block")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let index = match arg {
                        Some(arg) => {
                            let here = std::mem::replace(&mut self.cur, at);
                            let index = self.lane_heap_index(arg, sampler);
                            self.cur = here;
                            index?
                        }
                        None => self.s.m.i32(0),
                    };
                    incoming.push((index, at));
                }
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a per-lane descriptor from neither a heap load nor a choice of them"),
        }
    }

    /// The heap index of the descriptor at dword `first` of what the global
    /// load at `load` reads, in each lane: its address (base plus offsets)
    /// over the descriptor size.
    pub(super) fn fetched_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        const WHY: Why = "a per-lane descriptor load";
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            saddr,
            offset,
            ..
        } = li.fields
        else {
            return Err(WHY);
        };
        let c = self.cur;
        // The low dword of the address: SGPR base plus VGPR offset, or the
        // VGPR pair's low half (heap addresses fit in 32 bits).
        let mut at = self.i32_of(li.read(Reg::V(reg)).ok_or(WHY)?)?;
        if let Some(Operand::Scalar { code, .. }) = saddr
            && code != NULL
        {
            let base = self.i32_of(li.read(Reg::S(code)).ok_or(WHY)?)?;
            at = self.s.m.bin(c, Bin::Add, base, at);
        }
        let k = self
            .s
            .m
            .i32((offset as i32 as u32).wrapping_add(4 * u32::from(first)));
        at = self.s.m.bin(c, Bin::Add, at, k);
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, at, shift))
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block.
    pub(super) fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &li.fields
        else {
            return Err("a descriptor load");
        };
        let Operand::Scalar { code, .. } = *base else {
            return Err("a descriptor load");
        };
        let c = self.cur;
        let base = li.read(Reg::S(code)).ok_or("a descriptor load")?;
        let mut addr = self.i32_of(base)?;
        let k = self
            .s
            .m
            .i32((*offset as u32).wrapping_add(4 * u32::from(first)));
        addr = self.s.m.bin(c, Bin::Add, addr, k);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, addr, shift))
    }
}
