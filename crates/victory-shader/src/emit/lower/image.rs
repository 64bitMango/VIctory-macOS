//! Images and textures: image and buffer instructions, their view reads
//! and writes marked, gathers, and the buffer formats they load. Methods
//! of [`super::Lower`] called from its instruction dispatch.

use super::*;

impl<'a> Lower<'a> {
    /// An image instruction, its reads of texture views marked (module
    /// notes, "Read marks") after it.
    pub(super) fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        self.s.texture_reads.clear();
        let lowered = self.image_op(inst, b, i);
        for index in std::mem::take(&mut self.s.texture_reads) {
            if lowered.is_ok() {
                self.mark_view_read(index);
            }
        }
        lowered
    }

    /// Sets the read mark of resource-heap index `index` (module notes,
    /// "Read marks") where it is still clear. Nothing without
    /// [`Parts::view_marks`].
    pub(super) fn mark_view_read(&mut self, index: V) {
        if !self.x.view_marks {
            return;
        }
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let bitmap = self.s.view_writes_handle();
        let c = self.cur;
        let (base, k5, k2, k31, one, zero) = (
            k(self, super::super::VIEW_READS),
            k(self, 5),
            k(self, 2),
            k(self, 31),
            k(self, 1),
            k(self, 0),
        );
        let n = self.s.m.bin(c, Bin::Add, index, base);
        let word = self.s.m.bin(c, Bin::LShr, n, k5);
        let word = self.s.m.bin(c, Bin::Shl, word, k2);
        let bit = self.s.m.bin(c, Bin::And, n, k31);
        let bit = self.s.m.bin(c, Bin::Shl, one, bit);
        // Most reads find their slot marked: a load, not an atomic.
        let old = self.raw_dword(bitmap, word);
        let set = self.s.m.bin(c, Bin::And, old, bit);
        let clear = self.s.m.cmp(c, pred::EQ, set, zero);
        let at = self.begin_guarded(Some(clear));
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
        self.end_guarded(at, &[]);
    }

    /// Whether the image `src` is one an image atomic of the program also
    /// takes (the same descriptor, at a fixed address): DXIL's image atomics
    /// are on 32-bit integer views, so it has a 32-bit integer format.
    pub(super) fn atomic_image(&self, src: &Source) -> bool {
        let key = |s: &Source| match *s {
            Source::Loaded { load, first } => self.entry_address(load).map(|a| (a, first)),
            _ => None,
        };
        let Some(k) = key(src) else {
            return false;
        };
        self.x.res.descriptors.iter().any(|d| {
            let (b, i) = d.site;
            d.role == Role::Resource
                && self.x.p.blocks[b].insts[i]
                    .name()
                    .starts_with("image_atomic_")
                && key(&d.source) == Some(k)
        })
    }

    /// `image_store` of the channels in `dmask` alone (not all four), at
    /// coordinates `cs`, data from VGPR `reg` on (the enabled channels in
    /// order, with `d16` two per VGPR, low half first). The console leaves
    /// the other channels of the texel as they are. On an image an atomic
    /// also takes ([`Self::atomic_image`]), a 32-bit integer view, of one
    /// channel: the red one stored as an integer (`d16`: the low 16 bits,
    /// zero-extended, as for an unsigned format); any other channel is not
    /// in such a format, so stores nothing. Else (float and normalized
    /// formats, as full stores take them) the texel is read, the enabled
    /// channels replaced (`d16`: f16 values) and all four stored, which DXIL
    /// asks of a typed store.
    // shortcut: the read and the write are two accesses: another lane's
    // store to the same texel in between (of other channels) is lost.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn partial_image_store(
        &mut self,
        inst: &Inst,
        site: Site,
        src: &Source,
        kind: u32,
        cs: &[V],
        reg: u8,
        dmask: u8,
        d16: bool,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let channels: Vec<u32> = (0..4).filter(|ch| dmask >> ch & 1 != 0).collect();
        // Channel k's 32-bit data: a dword, or with d16 a half (`f16` true:
        // as f16 bits widened to f32, else zero-extended).
        let datum = |l: &mut Self, k: usize, f16: bool| -> Result<V, Why> {
            let r = if d16 { k / 2 } else { k } as u8;
            let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
            if !d16 {
                return if f16 { l.float_of(v) } else { l.i32_of(v) };
            }
            let c = l.cur;
            let mut x = l.i32_of(v)?;
            if k % 2 == 1 {
                let k16 = l.s.m.i32(16);
                x = l.s.m.bin(c, Bin::LShr, x, k16);
            }
            if f16 {
                let f32t = l.s.t.f32;
                Ok(l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]))
            } else {
                let m = l.s.m.i32(0xFFFF);
                Ok(l.s.m.bin(c, Bin::And, x, m))
            }
        };
        let void = self.s.t.void;
        if self.atomic_image(src) {
            if channels.len() != 1 {
                return Err("an image store of several channels to an integer view");
            }
            if channels[0] != 0 {
                return Ok(());
            }
            let x = datum(self, 0, false)?;
            // The atomics' view: one 32-bit integer channel.
            let h = self.handle(site, src, false, (kind | 0x1000, 0x105))?;
            let at = self.begin_guarded(guard);
            let iu = self.s.m.undef(self.s.t.i32);
            let mut args = vec![h];
            args.extend_from_slice(cs);
            args.extend([x, iu, iu, iu, self.s.m.i8(1)]);
            self.s.op(
                self.cur,
                "textureStore.i32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let mut new = Vec::new();
        for k in 0..channels.len() {
            new.push(datum(self, k, true)?);
        }
        let h = self.handle(site, src, false, (kind | 0x1000, 0x409))?;
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let iu = self.s.m.undef(self.s.t.i32);
        // A UAV has no levels: the level is undefined.
        let mut args = vec![h, iu];
        args.extend_from_slice(cs);
        args.extend([iu, iu, iu]);
        let old = self.s.op(
            c,
            "textureLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            66,
            &args,
        );
        let mut vals = Vec::new();
        for ch in 0..4u32 {
            vals.push(match channels.iter().position(|&e| e == ch) {
                Some(k) => new[k],
                None => self.s.m.extract(c, old, ch),
            });
        }
        let mut args = vec![h];
        args.extend_from_slice(cs);
        args.extend(vals);
        args.push(self.s.m.i8(0xF));
        self.s
            .op(c, "textureStore.f32", Attr::NoUnwind, void, 67, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    pub(super) fn image_op(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            dmask,
            dim,
            unorm,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        let name = inst.name();
        if name.starts_with("image_bvh") {
            return self.bvh(inst, b, i);
        }
        // A load or sample with no channel enabled returns the first.
        let dmask = if *dmask == 0 { &1 } else { dmask };
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let (kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only.
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (5, 3)
            }
            4 => (6, 2),
            5 => (7, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        // Address VGPRs in order.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        // With a16, two 16-bit components per VGPR, low half first, after
        // the offset, bias and compare value, which keep a VGPR each (an f16
        // bias in its low half).
        let extra = name
            .strip_prefix("image_sample")
            .or_else(|| name.strip_prefix("image_gather4"))
            .map_or(0, |rest| {
                rest.split('_')
                    .filter(|t| matches!(*t, "o" | "b" | "c"))
                    .count()
            });
        let addr_val = |l: &mut Self, k: usize| -> Result<V, Why> {
            let packed = *a16 && k >= extra;
            let r = *regs
                .get(if packed { extra + (k - extra) / 2 } else { k })
                .ok_or("an image address")?;
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            let x = l.i32_of(v)?;
            if !packed {
                return Ok(x);
            }
            let c = l.cur;
            let half = (k - extra) as u32 % 2;
            let (sh, mask) = (l.s.m.i32(16 * half), l.s.m.i32(0xFFFF));
            let x = l.s.m.bin(c, Bin::LShr, x, sh);
            Ok(l.s.m.bin(c, Bin::And, x, mask))
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked image")?;
        let store = name.starts_with("image_store");
        let uav = store
            || self
                .used_kind(src)
                .is_some_and(|u| u.kind == ResourceType::Uav);
        let props = (kind | if uav { 0x1000 } else { 0 }, 0x409);
        if let Some(op) = name.strip_prefix("image_atomic_") {
            // DXIL's atomic operations on a 32-bit integer texture UAV; the
            // old value returns with glc. Sub is the add of the negation.
            let code = match op {
                "add" | "sub" => 0,
                "and" => 1,
                "or" => 2,
                "xor" => 3,
                "smin" => 4,
                "smax" => 5,
                "umin" => 6,
                "umax" => 7,
                "swap" => 8,
                "cmpswap" => u32::MAX,
                _ => return Err(name),
            };
            let Fields::Image { cache, .. } = &inst.fields else {
                unreachable!()
            };
            // 16-bit coordinates (a16) are read as `addr_val` reads them.
            if *d16 {
                return Err("a 16-bit image atomic");
            }
            // dmask: the data dwords (cmpswap: twice the value's); two per
            // value is a 64-bit atomic (an R32G32_UINT view, low dword in R).
            let values = if code == u32::MAX { 2 } else { 1 };
            let wide = match dmask.count_ones() / values {
                1 => false,
                2 => true,
                _ => return Err("an image atomic of more than two dwords"),
            };
            let n = if wide { 2 } else { 1 };
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image atomic data");
            };
            let vals = self.vgprs(inst, data, values as u8 * n)?;
            let i64t = self.s.m.int(64);
            // Value k of the data: one dword, or two as an i64.
            let value_at = |l: &mut Self, k: usize| -> V {
                if !wide {
                    return vals[k];
                }
                let lo = l.s.m.cast(c, Cast::ZExt, vals[2 * k], i64t);
                let hi = l.s.m.cast(c, Cast::ZExt, vals[2 * k + 1], i64t);
                let k32 = l.s.m.int_const(i64t, 32);
                let hi = l.s.m.bin(c, Bin::Shl, hi, k32);
                l.s.m.bin(c, Bin::Or, lo, hi)
            };
            let mut value = value_at(self, 0);
            if op == "sub" {
                let z = if wide {
                    self.s.m.int_const(i64t, 0)
                } else {
                    self.s.m.i32(0)
                };
                value = self.s.m.bin(c, Bin::Sub, z, value);
            }
            let mut old = Vec::new();
            for k in 0..n {
                old.push(match inst.read(Reg::V(reg + k)) {
                    Some(v) if cache.glc => self.i32_of(v)?,
                    _ => self.s.m.i32(0),
                });
            }
            if wide && self.x.atomic_locks {
                let compare = (code == u32::MAX).then(|| value_at(self, 1));
                let h = self.handle(site, src, false, (kind | 0x1000 | 0x4000, 0x205))?;
                let at = self.begin_guarded(guard);
                let (lo, hi) = self.locked_atomic64(h, &cs, coords, op, value, compare);
                let r = self.end_guarded(at, &[(lo, old[0]), (hi, old[1])]);
                if cache.glc {
                    for (k, x) in (0..).zip(r) {
                        self.write(inst, Reg::V(reg + k), x, None);
                    }
                }
                return Ok(());
            }
            if wide {
                // SM 6.6's 64-bit atomics on typed resources in the heap: the
                // host device must support both (D3D12 options 9 and 11).
                self.s.flags |= flags::INT64 | flags::ATOMIC64_TYPED | flags::ATOMIC64_HEAP;
            }
            let h = self.handle(site, src, false, (kind | 0x1000, 0x105))?;
            let at = self.begin_guarded(guard);
            let (ty, suffix) = if wide {
                (i64t, "i64")
            } else {
                (self.s.t.i32, "i32")
            };
            let mut args = vec![h];
            let r = if code == u32::MAX {
                // cmpswap: data is (new, compare).
                args.extend(cs);
                let (new, compare) = (value, value_at(self, 1));
                args.extend([compare, new]);
                self.s.op(
                    self.cur,
                    &format!("atomicCompareExchange.{suffix}"),
                    Attr::NoUnwind,
                    ty,
                    79,
                    &args,
                )
            } else {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                self.s.op(
                    self.cur,
                    &format!("atomicBinOp.{suffix}"),
                    Attr::NoUnwind,
                    ty,
                    78,
                    &args,
                )
            };
            let r = if wide {
                let (lo, hi) = self.split64(r);
                vec![lo, hi]
            } else {
                vec![r]
            };
            let inside: Vec<_> = r.into_iter().zip(old).collect();
            let r = self.end_guarded(at, &inside);
            if cache.glc {
                for (k, x) in (0..).zip(r) {
                    self.write(inst, Reg::V(reg + k), x, None);
                }
            }
            return Ok(());
        }
        if store {
            if name != "image_store" {
                return Err("an image store to a mip level");
            }
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image store data");
            };
            if *dmask != 0xF {
                return self
                    .partial_image_store(inst, site, src, kind, &cs, reg, *dmask, *d16, guard);
            }
            // shortcut: stores are taken to write all four channels of a
            // float format (the console converts by the view's format).
            let mut vals = Vec::new();
            for k in 0..4u8 {
                // d16: two f16 channels per VGPR, low half first.
                let r = if *d16 { k / 2 } else { k };
                let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
                vals.push(if *d16 {
                    let mut x = self.i32_of(v)?;
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                } else {
                    self.float_of(v)?
                });
            }
            let h = self.handle(site, src, false, props)?;
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            args.extend(cs);
            args.extend(vals);
            args.push(self.s.m.i8(0xF));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "textureStore.f32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let (op, sample) = match name {
            "image_sample" | "image_sample_o" => (60, true),
            "image_sample_lz" | "image_sample_l" => (62, true),
            "image_sample_b" => (61, true),
            "image_sample_c_lz" => (65, true),
            "image_sample_d" | "image_sample_d_cl" => (63, true),
            "image_get_lod" if self.s.pixel => (81, true),
            // Implicit derivatives: pixel shaders only.
            "image_sample_c" if self.s.pixel => (64, true),
            "image_gather4" | "image_gather4_lz" | "image_gather4_lz_o" | "image_gather4_l"
            | "image_gather4_l_o" => (73, true),
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            _ => return Err(name),
        };
        if op == 72 {
            // Width, height, depth or layers, levels (per dmask) of the level
            // in the first address: DXIL's dimensions; a channel DXIL leaves
            // out reads 1 (UAVs: one level).
            let h = self.handle(site, src, false, props)?;
            let mip = if uav && self.x.uav_size_level0 {
                self.s.m.i32(0)
            } else if uav {
                self.s.m.undef(self.s.t.i32)
            } else {
                addr_val(self, 0)?
            };
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let one = self.s.m.i32(1);
            let present = match kind {
                1 => [true, false, false, true],
                2 | 5 | 6 => [true, true, false, true],
                _ => [true, true, true, true],
            };
            let Operand::Vector { reg, .. } = *data else {
                return Err("an image result");
            };
            for (k, ch) in (0..4u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                let x = if ch == 3 && uav || !present[ch as usize] {
                    one
                } else {
                    self.s.m.extract(c, d, ch)
                };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(site, src, false, props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        let r = if sample {
            let ss = self
                .descriptors
                .get(&(site, true))
                .copied()
                .ok_or("an untracked sampler")?;
            if matches!(name, "image_gather4_l" | "image_gather4_l_o") {
                if *a16 || *d16 || *unorm || kind != 2 {
                    return Err(
                        "a gather at a level of 16-bit values, texel coordinates or not 2D",
                    );
                }
                let n = 3 + usize::from(name.ends_with("_o"));
                let a = (0..n)
                    .map(|k| addr_val(self, k))
                    .collect::<Result<Vec<_>, _>>()?;
                let lod = inst.read(Reg::V(regs[n - 1])).ok_or("an image address")?;
                return self.gather_at_level(inst, h, ss, &a, lod, guard);
            }
            // Comparison: the reference value is the first address after
            // the bias, and the sampler is a comparison sampler.
            let cmp = matches!(op, 64 | 65);
            if cmp && (*a16 || dmask.count_ones() != 1) {
                return Err("a comparison sample of 16-bit coordinates or several channels");
            }
            let sh = self.handle(site, ss, true, (14 | if cmp { 0x8000 } else { 0 }, 0))?;
            // Texel offsets (`_o`) come first: x in bits 0-5, y in 8-13,
            // signed (programmable offsets: gathers only).
            let offset_word = name.ends_with("_o");
            let first =
                usize::from(offset_word) + usize::from(name == "image_sample_b") + usize::from(cmp);
            // Coordinates the size divides (not the array slice); explicit
            // gradients (x then y) come before the coordinates.
            let spatial = coords - usize::from(kind == 6 || kind == 7);
            // (A cube's are two per direction, on the face.)
            let grads = match op {
                63 if kind == 5 => 4,
                63 => 2 * spatial,
                _ => 0,
            };
            let mut cs = vec![fu; 4];
            // With a16, sampling coordinates and LOD are f16.
            let to_float = |l: &mut Self, x: V| {
                if *a16 {
                    let f = l.s.t.f32;
                    l.s.op(l.cur, "legacyF16ToF32", Attr::ReadNone, f, 131, &[x])
                } else {
                    l.as_float(x)
                }
            };
            // A level query takes no array slice (the code may pass anything
            // there, e.g. an unwritten register).
            let read = if op == 81 { spatial } else { coords };
            for (k, slot) in cs.iter_mut().enumerate().take(read) {
                let x = addr_val(self, first + grads + k)?;
                *slot = to_float(self, x);
            }
            if kind == 5 {
                if *unorm || op == 81 {
                    return Err("a cube sample with texel coordinates or a level query");
                }
                let dir = self.cube_direction(cs[0], cs[1], cs[2]);
                cs[..3].copy_from_slice(&dir);
            }
            if *unorm {
                // Texel coordinates: divided by the size of level 0.
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let z = self.s.m.i32(0);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                    let n = self.s.m.extract(c, d, k as u32);
                    let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                    *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
                }
            }
            if op == 81 {
                // The clamped then the unclamped level of detail.
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let f32t = self.s.t.f32;
                for (k, ch) in (0..2u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                    let clamped = self.s.m.bool(ch == 0);
                    let lod = self.s.op(
                        c,
                        "calculateLOD.f32",
                        Attr::ReadOnly,
                        f32t,
                        81,
                        &[h, sh, cs[0], cs[1], cs[2], clamped],
                    );
                    let x = self.as_int(lod);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let offsets = if kind == 5 {
                0
            } else {
                (coords - usize::from(kind == 6 || kind == 7)).min(3)
            };
            let zero = self.s.m.i32(0);
            let mut os = vec![iu; 3];
            for o in os.iter_mut().take(offsets) {
                *o = zero;
            }
            let mut args = vec![h, sh];
            args.extend(&cs);
            match op {
                60 => {
                    if offset_word {
                        // DXIL takes immediate offsets in -8..=7; the
                        // console's are 6-bit and may be computed.
                        let w = addr_val(self, 0)?;
                        let w = self
                            .s
                            .m
                            .int_value(w)
                            .ok_or("a sample offset that is not constant")?
                            as u32;
                        for (k, o) in os.iter_mut().enumerate().take(offsets) {
                            let x = (w >> (8 * k) & 0x3F) as i32;
                            let x = if x >= 32 { x - 64 } else { x };
                            if !(-8..=7).contains(&x) {
                                return Err("a sample offset past -8..7");
                            }
                            *o = self.s.m.i32(x as u32);
                        }
                    }
                    args.extend(&os);
                    args.push(fu);
                    self.s
                        .op(c, "sample.f32", Attr::ReadOnly, self.s.t.res_ret, 60, &args)
                }
                61 => {
                    let bias = addr_val(self, 0)?;
                    let bias = to_float(self, bias);
                    if self.x.bias_as_gradients && self.x.native.stage == HwStage::Cs {
                        // D3DMetal ignores SampleBias in compute shaders
                        // (found by nickorlabs, #30). Scale explicit quad
                        // gradients by 2^bias instead:
                        // log2(rho * 2^bias) = log2(rho) + bias.
                        let scale = self.unary(21, bias);
                        let mut gradients = vec![fu; 6];
                        for k in 0..spatial {
                            let dx = self.unary(83, cs[k]);
                            let dy = self.unary(84, cs[k]);
                            gradients[k] = self.s.m.bin(c, Bin::Mul, dx, scale);
                            gradients[3 + k] = self.s.m.bin(c, Bin::Mul, dy, scale);
                        }
                        args.extend(&os);
                        args.extend(gradients);
                        args.push(fu);
                        self.s.op(
                            c,
                            "sampleGrad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            63,
                            &args,
                        )
                    } else {
                        args.extend(&os);
                        args.extend([bias, fu]);
                        self.s.op(
                            c,
                            "sampleBias.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            61,
                            &args,
                        )
                    }
                }
                63 if kind == 5 => {
                    // A cube's gradients are the face's: ds, dt per step in x,
                    // then in y, in face coordinates (a face spans 1). The level
                    // the hardware picks from them, isotropically: log2 of the
                    // longer gradient in texels of a face, then the clamp.
                    // shortcut: sampled at that level, so a sampler's
                    // anisotropy is lost and its LOD bias comes after the
                    // clamp (the console adds it before); D3D12 cube sampling
                    // takes 3D gradients only. Upgrade if cube lookups blur or
                    // a biased sampler meets a binding clamp.
                    let mut g = [fu; 4];
                    for (k, x) in g.iter_mut().enumerate() {
                        let v = addr_val(self, first + k)?;
                        *x = to_float(self, v);
                    }
                    let i32t = self.s.t.i32;
                    let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                    let z = self.s.m.i32(0);
                    let d = self
                        .s
                        .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                    let w = self.s.m.extract(c, d, 0);
                    let w = self.s.m.cast(c, Cast::UiToFp, w, self.s.t.f32);
                    let mut len2 = [fu; 2];
                    for (k, l) in len2.iter_mut().enumerate() {
                        let a = self.s.m.bin(c, Bin::Mul, g[2 * k], g[2 * k]);
                        let b = self.s.m.bin(c, Bin::Mul, g[2 * k + 1], g[2 * k + 1]);
                        *l = self.s.m.bin(c, Bin::Add, a, b);
                    }
                    // FMax, Log (base 2): 0.5 log2(max(|dx|², |dy|²) w²).
                    let m = self.binary(35, len2[0], len2[1]);
                    let w2 = self.s.m.bin(c, Bin::Mul, w, w);
                    let m = self.s.m.bin(c, Bin::Mul, m, w2);
                    let half = self.s.m.f32(0.5f32.to_bits());
                    let l = self.unary(23, m);
                    let mut lod = self.s.m.bin(c, Bin::Mul, l, half);
                    if name.ends_with("_cl") {
                        let x = addr_val(self, first + grads + coords)?;
                        let clamp = to_float(self, x);
                        lod = self.binary(35, lod, clamp);
                    }
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        c,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                63 => {
                    // With a16 each direction's gradients start a VGPR.
                    if *a16 && spatial % 2 == 1 {
                        return Err("16-bit gradients of a 1D or 3D image");
                    }
                    let mut g = vec![fu; 6];
                    for k in 0..spatial {
                        let x = addr_val(self, first + k)?;
                        g[k] = to_float(self, x);
                        let y = addr_val(self, first + spatial + k)?;
                        g[3 + k] = to_float(self, y);
                    }
                    args.extend(&os);
                    args.extend(g);
                    // `_cl`: the least level of detail, after the coordinates.
                    let clamp = if name.ends_with("_cl") {
                        let x = addr_val(self, first + grads + coords)?;
                        to_float(self, x)
                    } else {
                        fu
                    };
                    args.push(clamp);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                64 | 65 => {
                    let reference = addr_val(self, first - 1)?;
                    let reference = self.as_float(reference);
                    args.extend(&os);
                    args.push(reference);
                    let name = if op == 64 {
                        args.push(fu);
                        "sampleCmp.f32"
                    } else {
                        "sampleCmpLevelZero.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
                62 => {
                    let lod = if name == "image_sample_l" {
                        let x = addr_val(self, coords)?;
                        to_float(self, x)
                    } else {
                        self.s.m.f32(0)
                    };
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        c,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                _ => {
                    if name == "image_gather4" {
                        return Err("a gather with implicit level");
                    }
                    if dmask.count_ones() != 1 {
                        return Err("a gather of several channels");
                    }
                    if offset_word {
                        let w = addr_val(self, 0)?;
                        for (k, at) in [(0, 26), (1, 18)] {
                            // Bits at..at+5 sign-extended: shift up, then down.
                            let left = self.s.m.i32(at);
                            let t = self.s.m.bin(c, Bin::Shl, w, left);
                            let k26 = self.s.m.i32(26);
                            os[k] = self.s.m.bin(c, Bin::AShr, t, k26);
                        }
                    }
                    args.extend(&os[..2]);
                    args.push(self.s.m.i32(dmask.trailing_zeros()));
                    self.s.op(
                        c,
                        "textureGather.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        73,
                        &args,
                    )
                }
            }
        } else {
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let mip = if name == "image_load_mip" {
                addr_val(self, coords)?
            } else if uav {
                // Our declarations have four channels.
                self.s.flags |= flags::TYPED_UAV_LOADS;
                iu
            } else {
                self.s.m.i32(0)
            };
            let mut args = vec![h, mip];
            args.extend(&cs);
            args.extend([iu, iu, iu]);
            self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            )
        };
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        let channels: Vec<u32> = if op == 73 {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let x = self.s.m.extract(c, r, ch);
            xs.push(if *d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                let i32t = self.s.t.i32;
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
            } else {
                self.as_int(x)
            });
        }
        if *d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// The direction a cube sample's face coordinates stand for: the
    /// console takes s and t in [1, 2] (`v_cubesc`/`v_cubetc` over
    /// `|v_cubema|`, plus 1.5) and the face id (`v_cubeid`), which the
    /// host's cube sampling turns back into the same texel.
    // shortcut: the view is taken as a cube, not a cube array (whose face
    // operand adds 8 × the slice).
    pub(super) fn cube_direction(&mut self, s: V, t: V, face: V) -> [V; 3] {
        let c = self.cur;
        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
        let (two, three) = (k(self, 2.0), k(self, 3.0));
        let u = self.s.m.bin(c, Bin::Mul, s, two);
        let u = self.s.m.bin(c, Bin::Sub, u, three);
        let v = self.s.m.bin(c, Bin::Mul, t, two);
        let v = self.s.m.bin(c, Bin::Sub, v, three);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let (one, mone) = (k(self, 1.0), k(self, -1.0));
        let f = self.s.m.cast(c, Cast::FpToUi, face, self.s.t.i32);
        // Per face +x, -x, +y, -y, +z, -z.
        let table = [
            [one, mone, u, u, u, nu],
            [nv, nv, one, mone, nv, nv],
            [nu, u, v, nv, one, mone],
        ];
        table.map(|row| {
            let mut r = row[0];
            for (n, &x) in row.iter().enumerate().skip(1) {
                let kn = self.s.m.i32(n as u32);
                let is = self.s.m.cmp(c, pred::EQ, f, kn);
                r = self.s.m.select(c, is, x, r);
            }
            r
        })
    }

    /// `image_gather4_l(_o)`: DXIL gathers only from a view's first level,
    /// so the four texels are loaded from the level. Only what the title's
    /// culling shaders use: an immediate sampler without bias whose border
    /// is transparent black (a texel outside reads as a load outside: 0),
    /// and a level of detail proven whole (which level a fraction picks is
    /// not known; a whole one is that level, clamped to the sampler's and
    /// the view's).
    pub(super) fn gather_at_level(
        &mut self,
        inst: &Inst,
        h: V,
        sampler: &Source,
        a: &[V],
        lod: Value,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let Fields::Image { data, dmask, .. } = &inst.fields else {
            unreachable!()
        };
        let Source::Constant(w) = sampler else {
            return Err("a gather at a level through a sampler from memory");
        };
        let &[w0, w1, w2, w3, ..] = &w[..] else {
            return Err("a sampler constant");
        };
        let bits = |word: u32, at: u32, n: u32| word >> at & ((1 << n) - 1);
        // Clamp to a border (x, y) of transparent black; no anisotropy,
        // unnormalized or truncated coordinates, reduction or bias; whole
        // level limits.
        if w0 & 0x3F != 0x36
            || w0 & (7 << 9 | 1 << 15 | 1 << 19 | 1 << 27 | 3 << 29) != 0
            || bits(w3, 30, 2) != 0
            || bits(w2, 0, 20) != 0
            || w1 & 0x0F_F0FF != 0
        {
            return Err("a gather at a level through a sampler other than a black border's");
        }
        // No channel enabled: the first.
        let dmask = if *dmask == 0 { 1 } else { *dmask };
        if dmask.count_ones() != 1 {
            return Err("a gather of several channels");
        }
        let mip_filter = bits(w2, 26, 2) != 0;
        if mip_filter && self.whole(lod, &mut HashSet::new()) != Some(true) {
            return Err("a gather at a level that may not be whole");
        }
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims_t = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let zero = self.s.m.i32(0);
        let level = if mip_filter {
            // The sampler's limits, then the view's last level.
            let f = self.as_float(a[a.len() - 1]);
            let lo = self.s.m.f32((bits(w1, 0, 12) as f32 / 256.0).to_bits());
            let hi = self.s.m.f32((bits(w1, 12, 12) as f32 / 256.0).to_bits());
            let f = self.binary(35, f, lo);
            let f = self.binary(36, f, hi);
            let l = self.s.m.cast(c, Cast::FpToSi, f, i32t);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims_t, 72, &[h, zero]);
            let levels = self.s.m.extract(c, d, 3);
            let one = self.s.m.i32(1);
            let last = self.s.m.bin(c, Bin::Sub, levels, one);
            let past = self.s.m.cmp(c, pred::SGT, l, last);
            self.s.m.select(c, past, last, l)
        } else {
            zero
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims_t, 72, &[h, level]);
        let offset = (a.len() == 4).then(|| a[0]);
        let mut first = [zero; 2];
        for (k, at) in [(0, 26), (1, 18)] {
            let size = self.s.m.extract(c, d, k as u32);
            first[k] = self.gather_first(a[a.len() - 3 + k], size);
            if let Some(o) = offset {
                // Bits at..at+5 of the offset word sign-extended.
                let left = self.s.m.i32(at);
                let t = self.s.m.bin(c, Bin::Shl, o, left);
                let k26 = self.s.m.i32(26);
                let o = self.s.m.bin(c, Bin::AShr, t, k26);
                first[k] = self.s.m.bin(c, Bin::Add, first[k], o);
            }
        }
        // Texels (x, y + 1), (x + 1, y + 1), (x + 1, y), (x, y).
        let one = self.s.m.i32(1);
        let next = first.map(|f| self.s.m.bin(c, Bin::Add, f, one));
        let iu = self.s.m.undef(i32t);
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        for (k, (x, y)) in [
            (first[0], next[1]),
            (next[0], next[1]),
            (next[0], first[1]),
            (first[0], first[1]),
        ]
        .into_iter()
        .enumerate()
        {
            let args = [h, level, x, y, iu, iu, iu, iu];
            let r = self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            );
            let v = self.s.m.extract(c, r, dmask.trailing_zeros());
            let v = self.as_int(v);
            self.write(inst, Reg::V(reg + k as u8), v, guard);
        }
        Ok(())
    }

    /// The first texel of a gather's footprint along an axis of `size`
    /// texels at coordinate `u` (bits of a float): floor(u × size − 0.5)
    /// with the product exact and rounded to 1/256 of a texel, half up, as
    /// the hardware does (`gather_level_tests`), i.e.
    /// (floor(u × size × 512) − 255) >> 9, in 64-bit integers. NaN is 0;
    /// beyond ±2 every texel of the footprint is outside, as at ±2.
    pub(super) fn gather_first(&mut self, u: V, size: V) -> V {
        let c = self.cur;
        let f = self.as_float(u);
        let num = self.s.m.cmp(c, pred::OEQ, f, f);
        let fz = self.s.m.f32(0);
        let f = self.s.m.select(c, num, f, fz);
        let (lo, hi) = (
            self.s.m.f32((-2.0f32).to_bits()),
            self.s.m.f32(2.0f32.to_bits()),
        );
        let f = self.binary(35, f, lo);
        let f = self.binary(36, f, hi);
        let x = self.as_int(f);
        // u = m × 2^(e − 150), e at least 1 (denormals).
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k23, kff) = (k(self, 23), k(self, 0xFF));
        let e = self.s.m.bin(c, Bin::LShr, x, k23);
        let e = self.s.m.bin(c, Bin::And, e, kff);
        let (kmant, zero, one) = (k(self, 0x7F_FFFF), k(self, 0), k(self, 1));
        let m = self.s.m.bin(c, Bin::And, x, kmant);
        let normal = self.s.m.cmp(c, pred::NE, e, zero);
        let implicit = k(self, 0x80_0000);
        let bit = self.s.m.select(c, normal, implicit, zero);
        let m = self.s.m.bin(c, Bin::Or, m, bit);
        let e = self.s.m.select(c, normal, e, one);
        // u × size × 512 = m × size × 2^(e − 141); |u| ≤ 2 keeps e ≤ 128.
        let k141 = k(self, 141);
        let shift = self.s.m.bin(c, Bin::Sub, k141, e);
        let k63 = k(self, 63);
        let far = self.s.m.cmp(c, pred::SGT, shift, k63);
        let shift = self.s.m.select(c, far, k63, shift);
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let m = self.s.m.cast(c, Cast::ZExt, m, i64t);
        let n = self.s.m.cast(c, Cast::ZExt, size, i64t);
        let p = self.s.m.bin(c, Bin::Mul, m, n);
        let z64 = self.s.m.int_const(i64t, 0);
        let neg = self.s.m.bin(c, Bin::Sub, z64, p);
        let negative = self.s.m.cmp(c, pred::SLT, x, zero);
        let p = self.s.m.select(c, negative, neg, p);
        let shift = self.s.m.cast(c, Cast::ZExt, shift, i64t);
        let g = self.s.m.bin(c, Bin::AShr, p, shift);
        let k255 = self.s.m.int_const(i64t, 255);
        let g = self.s.m.bin(c, Bin::Sub, g, k255);
        let k9 = self.s.m.int_const(i64t, 9);
        let g = self.s.m.bin(c, Bin::AShr, g, k9);
        self.s.m.cast(c, Cast::Trunc, g, self.s.t.i32)
    }

    /// Whether `v` holds, as a float, a whole number or an infinity in
    /// every lane: `Some(true)` and never NaN, `Some(false)` or NaN, `None`
    /// perhaps a fraction. Follows moves, selects, minimum and maximum,
    /// sums with constants, rounding and conversions from integers (a level
    /// of detail made with `ceil`), through phis and guarded writes' kept
    /// values.
    pub(super) fn whole(&self, v: Value, seen: &mut HashSet<Value>) -> Option<bool> {
        if !seen.insert(v) {
            // Around a loop: what enters it decides.
            return Some(true);
        }
        let p = self.x.p;
        if let Some(&(b, i)) = self.phi_at.get(&v) {
            let mut never_nan = true;
            for &arg in &p.blocks[b].phis[i].args {
                never_nan &= self.whole(arg, seen)?;
            }
            return Some(never_nan);
        }
        let Some(&b) = self.def_block.get(&v) else {
            // A VGPR no one set reads zero.
            let vgpr = p
                .inputs
                .iter()
                .any(|&(r, x)| x == v && matches!(r, Reg::V(_)));
            let io = self.x.io.inputs.iter().any(|i| i.0 == v);
            return (vgpr && !io).then_some(true);
        };
        let (i, inst) = p.blocks[b]
            .insts
            .iter()
            .enumerate()
            .find(|(_, i)| i.writes.iter().any(|w| w.1 == v))?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if f.omod != 0 || f.sdwa.is_some() || f.dpp.is_some() {
            return None;
        }
        let mut never_nan = true;
        if self.x.st.guards[b][i].is_some() {
            let Some(Operand::Vector { reg, .. }) = f.dst else {
                return None;
            };
            never_nan = self.whole(inst.read(Reg::V(reg))?, seen)?;
        }
        // Per source: (never NaN, a finite constant).
        let mut src = |k: usize| -> Option<(bool, bool)> {
            let x = match *f.src.get(k)? {
                Operand::Vector { reg, .. } => {
                    return Some((self.whole(inst.read(Reg::V(reg))?, seen)?, false));
                }
                Operand::Const(c) => crate::resource::inline(c)?,
                Operand::Literal(x) => x,
                _ => return None,
            };
            let x = f32::from_bits(x);
            (x.is_infinite() || x.fract() == 0.0).then_some((true, x.is_finite()))
        };
        let r = match inst.name() {
            "v_cvt_f32_i32" | "v_cvt_f32_u32" => true,
            "v_floor_f32" | "v_ceil_f32" | "v_trunc_f32" | "v_rndne_f32" => false,
            "v_mov_b32" => src(0)?.0,
            "v_cndmask_b32" => src(0)?.0 & src(1)?.0,
            "v_max_f32" | "v_min_f32" => src(0)?.0 | src(1)?.0,
            "v_max3_f32" | "v_min3_f32" => src(0)?.0 | src(1)?.0 | src(2)?.0,
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" => {
                // Infinities of both signs make NaN.
                let (x, y) = (src(0)?, src(1)?);
                x.0 && y.0 && (x.1 || y.1)
            }
            _ => return None,
        };
        Some(r && never_nan)
    }

    pub(super) fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if name == "buffer_gl0_inv" || name == "buffer_gl1_inv" {
            // Console cache coherence: gl0 writes back and invalidates the
            // per-CU shader L0, gl1 invalidates GL1, so one wavefront's
            // writes become visible to others. D3D12 already guarantees UAV
            // coherence across wavefronts and dispatches, DXIL has no
            // equivalent instruction, and this project's host-visible
            // writeback comes from command-stream GpuWrite ranges rather
            // than shader cache hints. Ordering is kept by the surrounding
            // s_waitcnt/s_barrier, which the translator does lower.
            return Ok(());
        }
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err(name);
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let atomic = match name.strip_prefix("buffer_atomic_") {
            None => None,
            Some("add") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap") => Some(u32::MAX),
            Some(_) => return Err(name),
        };
        // d16: two 16-bit channels per VGPR, low half first (buffer and
        // tbuffer formats alike).
        let d16 = name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            Some("ubyte" | "sbyte" | "ushort" | "sshort") if !store => (false, 1),
            Some("byte" | "short") if store => (false, 1),
            _ => return Err(name),
        };
        // Loads or stores of a byte or a half: its width and whether signed.
        let sub = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
        {
            Some("ubyte" | "byte") => Some((8, false)),
            Some("sbyte") => Some((8, true)),
            Some("ushort" | "short") => Some((16, false)),
            Some("sshort") => Some((16, true)),
            _ => None,
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        // A descriptor built around a root view has the stride the code puts
        // in dword 1 (bits 16-29) and no record limit (dword 2 is -1): an
        // index is a byte offset into the raw root view.
        let index = match (src, index) {
            (
                Source::Built {
                    address,
                    bits,
                    words,
                },
                Some(ix),
            ) if address.param < self.x.native.layout.len() && !typed => {
                if bits >> 30 != 0 || words[0] != u32::MAX {
                    return Err("a root view indexed with swizzling or a record limit");
                }
                let stride = self.s.m.i32(bits >> 16 & 0x3FFF);
                let byte = self.s.m.bin(c, Bin::Mul, ix, stride);
                off = self.s.m.bin(c, Bin::Add, off, byte);
                None
            }
            (_, ix) => ix,
        };
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let table = used.is_some_and(|u| u.root == RootType::Table);
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            if table {
                // A table's constant buffer is a raw view (scalar loads above).
                let Some(Operand::Vector { reg, .. }) = *data else {
                    return Err("a buffer result");
                };
                let (h, byte) = self.raw_view(site, src, byte, 4 * count, (11, 0), false, guard)?;
                for k in 0..count {
                    let four = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, byte, four);
                    let (mask, al) = (self.s.m.i8(1), self.s.m.i32(4));
                    let u = self.s.m.undef(self.s.t.i32);
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
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let x = self.cbuffer_dword(h, a);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        // A heap-indexed buffer read by element has no used-resource entry,
        // so no stride: its bytes are addressed as the hardware does, the
        // element times the view's stride (the host's table) plus the offset.
        let index = match index {
            Some(i) if used.is_none() && !typed => {
                let stride = self.view_stride(src)?;
                let rows = self.s.m.bin(c, Bin::Mul, i, stride);
                off = self.s.m.bin(c, Bin::Add, rows, off);
                None
            }
            index => index,
        };
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        if let Some(op) = atomic {
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
            let (h, coords, byte) = match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride < 4 || !stride.is_multiple_of(4) {
                        return Err("a structured buffer of unknown stride");
                    }
                    let h = self.handle(site, src, false, (12 | u, stride))?;
                    let s = self.s.m.i32(stride);
                    let row = self.s.m.bin(self.cur, Bin::Mul, index, s);
                    let byte = self.s.m.bin(self.cur, Bin::Add, row, off);
                    (h, [index, off, iu], byte)
                }
                None => {
                    let (h, off) = self.raw_view(site, src, off, 4, (11 | u, 0), true, guard)?;
                    (h, [off, iu, iu], off)
                }
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if returns => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let at = self.begin_guarded(guard);
            let i32t = self.s.t.i32;
            let r = if op == u32::MAX {
                // cmpswap: data is (new, compare).
                let mut args = vec![h];
                args.extend(coords);
                args.extend([vals[1], vals[0]]);
                self.s.op(
                    self.cur,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &args,
                )
            } else {
                let mut args = vec![h, self.s.m.i32(op)];
                args.extend(coords);
                args.push(vals[0]);
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
            };
            self.mark_view_write(h, byte, 4, None);
            let r = self.end_guarded(at, &[(r, old)])[0];
            if returns {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if let (true, Some((width, _))) = (store, sub) {
            // The field of the dword holding it, by two atomics on that dword
            // (clear, then set): other lanes' fields in it are left alone.
            // shortcut: the field is taken not to cross a dword.
            if index.is_some() {
                return Err("a byte or half buffer store by index");
            }
            let d = data.as_ref().ok_or("buffer store data")?;
            let x = self.vgprs(inst, d, 1)?[0];
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            let field = self.s.m.i32(if width == 8 { 0xFF } else { 0xFFFF });
            let x = self.s.m.bin(c, Bin::And, x, field);
            let x = self.s.m.bin(c, Bin::Shl, x, sh);
            let mask = self.s.m.bin(c, Bin::Shl, field, sh);
            let all = self.s.m.i32(u32::MAX);
            let keep = self.s.m.bin(c, Bin::Xor, mask, all);
            let (h, at) = self.raw_view(site, src, at, 4, (11 | 0x1000, 0), true, guard)?;
            let begin = self.begin_guarded(guard);
            let i32t = self.s.t.i32;
            for (op, v) in [(1, keep), (2, x)] {
                let args = [h, self.s.m.i32(op), at, iu, iu, v];
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args);
            }
            self.mark_view_write(h, at, 4, None);
            self.end_guarded(begin, &[]);
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            let typed_h = if typed {
                Some(self.handle(site, src, false, (10 | u, 0x409))?)
            } else {
                None
            };
            if d16 {
                // d16 data is f16 for float and normalized formats, widened
                // for the 32-bit float store; for integer formats (the view's
                // kind, from the view-stride table) the 16-bit integer, zero-
                // or sign-extended (stored through the float-typed handle as
                // loads read: see there).
                let integer = typed_h.and_then(|h| self.integer_view(h));
                let c = self.cur;
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    let mut w = self.as_int(f);
                    if let Some((int, signed)) = integer {
                        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
                        let zero_ext = self.s.m.bin(c, Bin::And, x, low);
                        let up = self.s.m.bin(c, Bin::Shl, x, k16);
                        let sign_ext = self.s.m.bin(c, Bin::AShr, up, k16);
                        let n = self.s.m.select(c, signed, sign_ext, zero_ext);
                        w = self.s.m.select(c, int, n, w);
                    }
                    wide.push(w);
                }
                vals = wide;
            }
            let mut args = Vec::new();
            // Where the write starts in the view, and how many bytes it may
            // cover (for its marks).
            let (byte, bytes);
            let (op, name) = if typed {
                // shortcut: typed stores are taken to write all four channels
                // of a float format.
                if count != 4 || *offen || *offset != 0 {
                    return Err("a typed buffer store of some channels or at an offset");
                }
                let h = typed_h.ok_or("a typed buffer store")?;
                let element = index.unwrap_or(self.s.m.i32(0));
                args.push(h);
                args.extend([element, iu]);
                for x in vals {
                    args.push(self.as_float(x));
                }
                // The element's size is the view's stride (16 bytes at most).
                (byte, bytes) = match self.s.view_index.get(&h).copied() {
                    Some(at) if self.x.view_marks => {
                        let stride = self.stride_at(at);
                        (self.s.m.bin(self.cur, Bin::Mul, element, stride), 16)
                    }
                    _ => (element, 16),
                };
                (69, "bufferStore.f32")
            } else {
                match index {
                    Some(index) => {
                        let stride = used.map_or(0, |u| u.stride);
                        if stride < 4 || !stride.is_multiple_of(4) {
                            return Err("a structured buffer of unknown stride");
                        }
                        args.push(self.handle(site, src, false, (12 | u, stride))?);
                        args.extend([index, off]);
                        let s = self.s.m.i32(stride);
                        let row = self.s.m.bin(self.cur, Bin::Mul, index, s);
                        (byte, bytes) = (self.s.m.bin(self.cur, Bin::Add, row, off), 4 * count);
                    }
                    None => {
                        let n = 4 * count;
                        let (h, off) =
                            self.raw_view(site, src, off, n, (11 | u, 0), true, guard)?;
                        args.extend([h, off, iu]);
                        (byte, bytes) = (off, n);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.mark_view_write(args[0], byte, bytes, None);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = *format {
            if index.is_some() {
                return Err("a typed-format load by index");
            }
            let (h, off) = self.raw_view(site, src, off, 4 * count, (11 | u, 0), false, guard)?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let mut xs = self.format_load(h, off, f, count)?;
            if d16 {
                xs = self.d16_channels(f, xs)?;
            }
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if let Some((width, signed)) = sub {
            // The dword holding the field, as `global_load_ubyte` and kin.
            // shortcut: the field is taken not to cross a dword.
            if index.is_some() {
                return Err("a byte or half buffer load by index");
            }
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let (h, at) = self.raw_view(site, src, at, 4, (11 | u, 0), false, guard)?;
            let c = self.cur;
            let (mask, align) = (self.s.m.i8(1), self.s.m.i32(4));
            let r = self.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                139,
                &[h, at, iu, mask, align],
            );
            let w = self.s.m.extract(c, r, 0);
            let w = self.as_int(w);
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            let x = self.s.m.bin(c, Bin::LShr, w, sh);
            let left = self.s.m.i32(32 - width);
            let x = self.s.m.bin(c, Bin::Shl, x, left);
            let x = self
                .s
                .m
                .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left);
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            self.write(inst, Reg::V(reg), x, guard);
            return Ok(());
        }
        let mut typed_h = None;
        let r = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(site, src, false, (10 | u, 0x409))?;
            typed_h = Some(h);
            let index = index.unwrap_or(self.s.m.i32(0));
            self.s.op(
                c,
                "bufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                68,
                &[h, index, iu],
            )
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride < 4 || !stride.is_multiple_of(4) {
                        return Err("a structured buffer of unknown stride");
                    }
                    let h = self.handle(site, src, false, (12 | u, stride))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, index, off, mask, align],
                    )
                }
                None => {
                    let bytes = 4 * count;
                    let (h, off) =
                        self.raw_view(site, src, off, bytes, (11 | u, 0), false, guard)?;
                    self.s.op(
                        self.cur,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, off, iu, mask, align],
                    )
                }
            }
        };
        let c = self.cur;
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if d16 {
            // Two 16-bit channels per VGPR (an odd last high half 0): f16 for
            // float and normalized formats; for integer formats (the view's
            // kind, from the view-stride table) the integer's low 16 bits.
            // shortcut: an integer view is read and written through the
            // float-typed handle, so its channels' bits travel as the driver
            // passes a mismatched component type (unchanged on the RX 6700
            // XT and WARP, `stride_tests`); an integer-typed second handle
            // for the slot would not rely on it.
            let integer = typed_h.and_then(|h| self.integer_view(h));
            let c = self.cur;
            let k16 = self.s.m.i32(16);
            let i32t = self.s.t.i32;
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let mut h = self
                        .s
                        .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
                    if let Some((int, _)) = integer {
                        let bits = self.as_int(f);
                        let low = self.s.m.i32(0xFFFF);
                        let low = self.s.m.bin(c, Bin::And, bits, low);
                        h = self.s.m.select(c, int, low, h);
                    }
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }
}
