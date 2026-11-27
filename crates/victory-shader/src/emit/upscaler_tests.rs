//! Compute patterns of an upscaler's passes, made as the translation makes
//! any program ([`super::translate_program`]) for 32-lane host waves and,
//! on Windows (under Wine: D3DMetal), run on the hardware device against
//! the console's results worked out by hand.
//!
//! - A bounded poll: a wave reloads flags until every lane sees its flag
//!   ready or a count of tries runs out, the "some lane is not ready" test
//!   being the SCC of a lane mask operation read as a scalar.
//! - Whole-wave OR reductions that end a pass: one (wave32) never restores
//!   EXEC, setting it to lane 0 alone for an atomic before the end; the
//!   other (wave64) joins its halves with a read of lane 0 inside the
//!   section and reads lane 63 after it, where the lanes differ.
//! - Image stores of some channels: 16-bit (`d16`) and 32-bit data into a
//!   float image, and the reset of a counter image an atomic also takes
//!   (a 32-bit integer view) with 16-bit data and coordinates.

use super::{HostShader, Options};
use crate::dxbc;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};

const S_ENDPGM: u32 = 0xBF81_0000;

/// A compute program of `threads` (ids in v0) with four root constants in
/// s0-s3 and a table in s4 whose first view is a raw buffer, its
/// descriptor loaded into s[8:11] before `body`.
fn native(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    native_with(body, wave64, threads, &[(0, 4)])
}

/// [`native`] with the table's views (each a UAV) at these dword offsets,
/// of these sizes in dwords.
fn native_with(body: &[u32], wave64: bool, threads: u32, views: &[(u32, u8)]) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xFA00_0000,
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    let param = |root, user_data, sgprs, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 32,
        layout: vec![
            param(RootType::Constants, 0, 4, 0),
            param(RootType::Table, 4, 1, 1),
        ],
        used: views
            .iter()
            .map(|&(table_offset_dwords, dwords)| UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords,
                lower_bound: 0,
                stride: 0,
                dwords,
                user_data: 4,
                slot: 1,
            })
            .collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `native` as the translation makes it for 32-lane host waves.
fn translated(native: &NativeProgram) -> Result<HostShader, crate::ShaderError> {
    let bytes = dxbc::build(&[]);
    let c = dxbc::Container::parse(&bytes).expect("container");
    let options = Options {
        narrow_waves: true,
        atomic_fences: true,
        ..Options::default()
    };
    super::translate_program(&c, native, options, None)
}

fn text(words: &[u32], wave64: bool) -> Vec<String> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    crate::gfx10::decode(&code)
        .unwrap_or_else(|e| panic!("{words:08x?}: {e}"))
        .iter()
        .map(|i| i.text(wave64))
        .collect()
}

/// The tries a poll makes at most, as the passes count them. A translated
/// loop also stops at the translation's own limit ([`super::loop_limit`],
/// 2^14), where its back edge is not taken: the poll goes on as when its
/// tries run out, after fewer of them.
const TRIES: u32 = 0x1_0000;

/// A count of tries below the loop limit, where the poll's own is exact.
#[cfg(windows)]
const FEW: u32 = 1000;

/// A bounded poll (after a pass that waits for the tiles an earlier pass
/// marks done): each try, every lane adds 1 to its flag (dword `id`) and
/// takes the old value; a lane is not ready while that is below its
/// threshold (`s0` for the thread numbered `s1`, else 2). The wave goes
/// round again while some lane is not ready (`s_and` of the compare with
/// EXEC, its SCC selected into an all-or-none mask) unless the tries ran
/// out (`tries` of them), then every lane stores the tries made at dword
/// 128 + `id`.
fn poll(wave64: bool, tries: u32) -> Vec<u32> {
    let (and, sel_ready, sel_out, orn2) = if wave64 {
        (0x8794_7E14, 0x8594_80C1, 0x8596_80C1, 0x8B94_1416)
    } else {
        (0x8714_7E14, 0x8514_80C1, 0x8516_80C1, 0x8B14_1416)
    };
    vec![
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x7E0C_0282, // v_mov_b32 v6, 2
        0x7E0E_0200, // v_mov_b32 v7, s0
        0x7D84_0001, // v_cmp_eq_u32 vcc, s1, v0
        0x020C_0F06, // v_cndmask_b32 v6, v6, v7, vcc
        0xBE90_0380, // s_mov_b32 s16, 0
        0x7E0A_0281, // v_mov_b32 v5, 1 (the loop's top)
        0xE0C8_5000,
        0x8002_0502, // buffer_atomic_add v5, v2, s[8:11], 0 offen glc
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xD484_0014,
        0x0002_0B06, // v_cmp_gt_i32 s[20:21], v6, v5
        and,         // s_and_b64 s[20:21], s[20:21], exec
        sel_ready,   // s_cselect_b64 s[20:21], -1, 0
        0x8010_8110, // s_add_u32 s16, s16, 1
        0xBF09_FF10,
        tries,       // s_cmp_ge_u32 s16, tries
        sel_out,     // s_cselect_b64 s[22:23], -1, 0
        orn2,        // s_orn2_b64 s[20:21], s[22:23], s[20:21]
        0xBF84_FFF2, // s_cbranch_scc0 to the loop's top
        0x7E02_0210, // v_mov_b32 v1, s16
        0xE070_1200,
        0x8002_0102, // buffer_store_dword v1, v2, s[8:11], 0 offen offset:512
    ]
}

/// [`poll`] run by the threads below s2 alone (`s_and_saveexec`), as a
/// pass polls inside a per-lane branch: a half of a console wave may have
/// no lane there; the others store nothing.
fn gated_poll(wave64: bool) -> Vec<u32> {
    let (save, restore) = if wave64 {
        (0xBE98_246A, 0xBEFE_0418)
    } else {
        (0xBE98_3C6A, 0xBEFE_0318)
    };
    let p = poll(wave64, TRIES);
    let (setup, rest) = p.split_at(5);
    let mut w = setup.to_vec();
    w.extend([
        0x7D88_0002,                     // v_cmp_gt_u32 vcc, s2, v0
        save,                            // s_and_saveexec_b64 s[24:25], vcc
        0xBF88_0000 | rest.len() as u32, // s_cbranch_execz past the poll
    ]);
    w.extend_from_slice(rest);
    w.push(restore); // s_mov_b64 exec, s[24:25]
    w
}

/// The group [`poll`] runs in: one console wave in wave64 (two console
/// waves looping apart could not meet in halves: a group of several runs
/// at the console's wave size), four in wave32.
fn poll_threads(wave64: bool) -> u32 {
    if wave64 { 64 } else { 128 }
}

/// What thread `t` of [`gated_poll`] stores, flag then tries, in console
/// waves of `wave` threads, with threshold `s0` for thread `s1` and the
/// threads below `s2` polling ([`poll`]: all of them), at most `tries`.
#[cfg(windows)]
fn poll_wants(t: u32, wave: u32, [s0, s1, s2]: [u32; 3], tries: u32) -> (u32, u32) {
    if t >= s2 {
        return (0, 0);
    }
    let most = if s1 / wave == t / wave && s1 < s2 {
        s0.max(2)
    } else {
        2
    };
    let made = (most + 1).min(tries).min(super::loop_limit());
    (made, made)
}

/// Every lane's input to the reductions: `1 << (id & 31)` for the threads
/// below s0, else 0.
const REDUCED: [u32; 7] = [
    0x3606_009F, // v_and_b32 v3, 31, v0
    0x7E0E_0281, // v_mov_b32 v7, 1
    0x340C_0F03, // v_lshlrev_b32 v6, v3, v7
    0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
    0x020C_0C80, // v_cndmask_b32 v6, 0, v6, vcc
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0xBF80_0000, // s_nop 0
];

/// A wave32 OR of [`REDUCED`] over the whole wave (inactive lanes' 0 put
/// in with `v_cndmask`, then DPP, `v_permlanex16_b32`), EXEC never
/// restored: lane 0 alone adds lane 31's result to dword `id >> 5`.
fn unrestored_reduction() -> Vec<u32> {
    let mut w = REDUCED.to_vec();
    w.extend([
        0xBE8D_3DC1, // s_or_saveexec_b32 s13, -1
        0xD501_003F,
        0x0036_0C80, // v_cndmask_b32 v63, 0, v6, s13
        0x387E_7EFA,
        0xFF04_B13F, // v_or_b32 v63, v63, v63 quad_perm:[1,0,3,2] fi:1
        0x387E_7EFA,
        0xFF04_4E3F, // v_or_b32 v63, v63, v63 quad_perm:[2,3,0,1] fi:1
        0x387E_7EFA,
        0xFF05_413F, // v_or_b32 v63, v63, v63 row_half_mirror fi:1
        0x387E_7EFA,
        0xFF05_403F, // v_or_b32 v63, v63, v63 row_mirror fi:1
        0xD778_003E,
        0x0201_013F, // v_permlanex16_b32 v62, v63, 0, 0
        0x387E_7F3E, // v_or_b32 v63, v62, v63
        0xBEFE_0381, // s_mov_b32 exec_lo, 1
        0xD760_000E,
        0x0001_3F3F, // v_readlane_b32 s14, v63, 31
        0x2C08_0083, // v_lshrrev_b32 v4, 3, v0
        0x7E0A_020E, // v_mov_b32 v5, s14
        0xE0C8_1000,
        0x8002_0504, // buffer_atomic_add v5, v4, s[8:11], 0 offen
    ]);
    w
}

/// A wave64 OR of [`REDUCED`]: each 32-lane half reduced as in
/// [`unrestored_reduction`], then lane 0's half OR-ed into every lane
/// inside the section; after the restore, lane 63's result (the whole
/// wave's) stored by every lane at its dword.
fn halves_reduction() -> Vec<u32> {
    let mut w = REDUCED.to_vec();
    w.extend([
        0xBE8C_25C1, // s_or_saveexec_b64 s[12:13], -1
        0xD501_003E,
        0x0032_0C80, // v_cndmask_b32 v62, 0, v6, s[12:13]
        0x387C_7CFA,
        0xFF04_B13E, // v_or_b32 v62, v62, v62 quad_perm:[1,0,3,2] fi:1
        0x387C_7CFA,
        0xFF04_4E3E, // v_or_b32 v62, v62, v62 quad_perm:[2,3,0,1] fi:1
        0x387C_7CFA,
        0xFF05_413E, // v_or_b32 v62, v62, v62 row_half_mirror fi:1
        0x387C_7CFA,
        0xFF05_403E, // v_or_b32 v62, v62, v62 row_mirror fi:1
        0xD778_003F,
        0x0201_013E, // v_permlanex16_b32 v63, v62, 0, 0
        0x387C_7F3E, // v_or_b32 v62, v62, v63
        0xD760_000E,
        0x0001_013E, // v_readlane_b32 s14, v62, 0
        0x387C_7C0E, // v_or_b32 v62, s14, v62
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0xD760_000E,
        0x0001_7F3E, // v_readlane_b32 s14, v62, 63
        0x7E0A_020E, // v_mov_b32 v5, s14
        0xE070_1000,
        0x8002_0502, // buffer_store_dword v5, v2, s[8:11], 0 offen
    ]);
    w
}

/// Image stores of some channels, by 16 threads (wave32) on 4x4 images,
/// thread `t` at texel (t & 3, t >> 2): into an RGBA32F image (table
/// entry 0) the four channels 10 + t, 20 + t, 30 + t, 40 + t, then red and
/// blue as f16 0.5 and -2.0 (`d16`, dmask 0x5), then alpha as 7.0 (dmask
/// 0x8); into an R32_UINT counter image (entry 1) an atomic add of 5,
/// then its reset: a `d16` store of 0x1234 (the high half 0xABCD) with
/// 16-bit coordinates (`a16`).
fn partial_stores() -> Vec<u32> {
    vec![
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xF40C_0502,
        0xFA00_0020, // s_load_dwordx8 s[20:27], s[4:5], 0x20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x3602_0083, // v_and_b32 v1, 3, v0
        0x2C04_0082, // v_lshrrev_b32 v2, 2, v0
        0x7E06_0D00, // v_cvt_f32_u32 v3, v0
        0x0608_06FF,
        0x4120_0000, // v_add_f32 v4, 10.0, v3
        0x060A_06FF,
        0x41A0_0000, // v_add_f32 v5, 20.0, v3
        0x060C_06FF,
        0x41F0_0000, // v_add_f32 v6, 30.0, v3
        0x060E_06FF,
        0x4220_0000, // v_add_f32 v7, 40.0, v3
        0xF020_1F08,
        0x0003_0401, // image_store v[4:7], v[1:2], s[12:19] dmask:0xf dim:2D unorm
        0x7E10_02FF,
        0xC000_3800, // v_mov_b32 v8, f16 -2.0 : 0.5
        0xF020_1508,
        0x8003_0801, // image_store v8, v[1:2], s[12:19] dmask:0x5 dim:2D unorm d16
        0x7E12_02FF,
        0x40E0_0000, // v_mov_b32 v9, 7.0
        0xF020_1808,
        0x0003_0901, // image_store v9, v[1:2], s[12:19] dmask:0x8 dim:2D unorm
        0x7E14_0285, // v_mov_b32 v10, 5
        0xF044_1108,
        0x0005_0A01, // image_atomic_add v10, v[1:2], s[20:27] dim:2D unorm
        0xD76F_000C,
        0x0405_2102, // v_lshl_or_b32 v12, v2, 16, v1
        0x7E16_02FF,
        0xABCD_1234, // v_mov_b32 v11, 0xabcd1234
        0xF020_1108,
        0xC005_0B0C, // image_store v11, v12, s[20:27] dim:2D unorm a16 d16
    ]
}

/// The table views of [`partial_stores`]: two image descriptors.
const IMAGES: [(u32, u8); 2] = [(0, 8), (8, 8)];

/// The OR of [`REDUCED`] over console wave `w` (of `wave` threads).
#[cfg(windows)]
fn reduced(w: u32, wave: u32, s0: u32) -> u32 {
    (w * wave..(w + 1) * wave)
        .filter(|&t| t < s0)
        .fold(0, |a, t| a | 1 << (t & 31))
}

#[test]
fn the_test_programs_are_what_their_comments_say() {
    let r = text(&[unrestored_reduction(), vec![S_ENDPGM]].concat(), false);
    for (at, want) in [
        (0, "v_and_b32 v3, 31, v0"),
        (2, "v_lshlrev_b32 v6, v3, v7"),
        (4, "v_cndmask_b32 v6, 0, v6, vcc_lo"),
        (7, "s_or_saveexec_b32 s13, -1"),
        (8, "v_cndmask_b32 v63, 0, v6, s13"),
        (9, "v_or_b32 v63, v63, v63 quad_perm:[1,0,3,2] fi:1"),
        (10, "v_or_b32 v63, v63, v63 quad_perm:[2,3,0,1] fi:1"),
        (11, "v_or_b32 v63, v63, v63 row_half_mirror fi:1"),
        (12, "v_or_b32 v63, v63, v63 row_mirror fi:1"),
        (13, "v_permlanex16_b32 v62, v63, 0, 0"),
        (14, "v_or_b32 v63, v62, v63"),
        (15, "s_mov_b32 exec_lo, 1"),
        (16, "v_readlane_b32 s14, v63, 31"),
        (19, "buffer_atomic_add v5, v4, s[8:11], 0 offen"),
    ] {
        assert_eq!(r[at], want, "instruction {at}");
    }
    let s = text(&[partial_stores(), vec![S_ENDPGM]].concat(), false);
    for (at, want) in [
        (
            10,
            "image_store v[4:7], v[1:2], s[12:19] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
        ),
        (
            12,
            "image_store v8, v[1:2], s[12:19] dmask:0x5 dim:SQ_RSRC_IMG_2D unorm d16",
        ),
        (
            14,
            "image_store v9, v[1:2], s[12:19] dmask:0x8 dim:SQ_RSRC_IMG_2D unorm",
        ),
        (
            16,
            "image_atomic_add v10, v[1:2], s[20:27] dim:SQ_RSRC_IMG_2D unorm",
        ),
        (17, "v_lshl_or_b32 v12, v2, 16, v1"),
        (
            19,
            "image_store v11, v12, s[20:27] dim:SQ_RSRC_IMG_2D unorm a16 d16",
        ),
    ] {
        assert_eq!(s[at], want, "instruction {at}");
    }
    let h = text(&[halves_reduction(), vec![S_ENDPGM]].concat(), true);
    for (at, want) in [
        (7, "s_or_saveexec_b64 s[12:13], -1"),
        (8, "v_cndmask_b32 v62, 0, v6, s[12:13]"),
        (13, "v_permlanex16_b32 v63, v62, 0, 0"),
        (15, "v_readlane_b32 s14, v62, 0"),
        (16, "v_or_b32 v62, s14, v62"),
        (17, "s_mov_b64 exec, s[12:13]"),
        (18, "v_readlane_b32 s14, v62, 63"),
        (20, "buffer_store_dword v5, v2, s[8:11], 0 offen"),
    ] {
        assert_eq!(h[at], want, "instruction {at}");
    }
    let p = text(&[poll(true, TRIES), vec![S_ENDPGM]].concat(), true);
    for (at, want) in [
        (6, "v_mov_b32 v5, 1"),
        (7, "buffer_atomic_add v5, v2, s[8:11], 0 offen glc"),
        (9, "v_cmp_gt_i32 s[20:21], v6, v5"),
        (10, "s_and_b64 s[20:21], s[20:21], exec"),
        (11, "s_cselect_b64 s[20:21], -1, 0"),
        (13, "s_cmp_ge_u32 s16, lit(0x00010000)"),
        (14, "s_cselect_b64 s[22:23], -1, 0"),
        (15, "s_orn2_b64 s[20:21], s[22:23], s[20:21]"),
        (16, "s_cbranch_scc0 label_0018"),
        (18, "buffer_store_dword v1, v2, s[8:11], 0 offen offset:512"),
    ] {
        assert_eq!(p[at], want, "instruction {at}");
    }
    let g = text(&[gated_poll(true), vec![S_ENDPGM]].concat(), true);
    for (at, want) in [
        (5, "v_cmp_gt_u32 vcc, s2, v0"),
        (6, "s_and_saveexec_b64 s[24:25], vcc"),
        (7, "s_cbranch_execz label_0068"),
        (19, "s_cbranch_scc0 label_0024"),
        (22, "s_mov_b64 exec, s[24:25]"),
    ] {
        assert_eq!(g[at], want, "instruction {at}");
    }
    let p = text(&[poll(false, TRIES), vec![S_ENDPGM]].concat(), false);
    for (at, want) in [
        (10, "s_and_b32 s20, s20, exec_lo"),
        (11, "s_cselect_b32 s20, -1, 0"),
        (15, "s_orn2_b32 s20, s22, s20"),
    ] {
        assert_eq!(p[at], want, "instruction {at}");
    }
}

/// The poll translates, wave64 (in halves, inside the per-lane branch
/// with its branches taken wave-wide) and wave32.
#[test]
fn a_bounded_poll_translates() {
    for wave64 in [true, false] {
        for (gated, words) in [(false, poll(wave64, TRIES)), (true, gated_poll(wave64))] {
            let shader = translated(&native(&words, wave64, poll_threads(wave64)))
                .unwrap_or_else(|e| panic!("wave64={wave64} gated={gated}: {e}"));
            assert_ne!(wave_size(&shader.dxil), 64, "wave64={wave64} gated={gated}");
        }
    }
}

/// Both reductions translate: the one never restored as a section ending
/// where EXEC becomes lane 0, the wave64 one in halves (no `WaveSize(64)`,
/// which D3DMetal does not run).
#[test]
fn reductions_that_end_a_pass_translate() {
    translated(&native(&unrestored_reduction(), false, 64)).expect("unrestored");
    let h = translated(&native(&halves_reduction(), true, 128)).expect("halves");
    assert_ne!(wave_size(&h.dxil), 64);
}

/// The image stores of some channels translate.
#[test]
fn image_stores_of_some_channels_translate() {
    translated(&native_with(&partial_stores(), false, 16, &IMAGES)).expect("translate");
}

/// The `WaveSize` a translation declares (PSV0's lane counts; 0 for none).
fn wave_size(dxil: &[u8]) -> u32 {
    let u = |o: usize| u32::from_le_bytes(dxil[o..o + 4].try_into().unwrap());
    for i in 0..u(28) as usize {
        let o = u(32 + 4 * i) as usize;
        if &dxil[o..o + 4] == b"PSV0" {
            return u(o + 8 + 4 + 16);
        }
    }
    panic!("no PSV0")
}

/// The root constants s0-s2 of `shader`.
#[cfg(windows)]
fn constants(shader: &HostShader, s: [u32; 3]) -> Vec<u32> {
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if u.root == RootType::Constants && u.slot == 0 && u.dword <= 2 {
            ud[u.index as usize] = s[u.dword as usize];
        }
    }
    ud
}

/// `shader` run once on the hardware device with root constants `s`
/// (s0-s2) and a raw UAV of `dwords`: the UAV after.
#[cfg(windows)]
fn run(host: &super::host::Host, shader: &HostShader, s: [u32; 3], dwords: u32) -> Vec<u32> {
    use super::host::{Dispatch, Root, View};
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let (root, pso) = host
        .compute_pipeline(shader)
        .unwrap_or_else(|e| panic!("compute_pipeline failed: {e}"));
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(constants(shader, s)))],
                views: &[View::RawUav(dwords)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    out[0][..dwords as usize].to_vec()
}

#[cfg(windows)]
fn hardware() -> Option<super::host::Host> {
    match super::host::Host::new(false) {
        Ok(host) => Some(host),
        Err(e) => {
            eprintln!("no SM 6.6 hardware device ({e}); skipped");
            None
        }
    }
}

/// On the hardware device: every console wave polls until its slowest
/// lane is ready, or the tries run out, and its lanes agree on the count.
#[cfg(windows)]
#[test]
fn a_bounded_poll_gives_the_console_s_results() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some(host) = hardware() else { return };
    for wave64 in [true, false] {
        let wave = if wave64 { 64 } else { 32 };
        let threads = poll_threads(wave64);
        let make = |w: &[u32]| translated(&native(w, wave64, threads)).expect("translate");
        let plain = make(&poll(wave64, TRIES));
        let few = make(&poll(wave64, FEW));
        let gated = make(&gated_poll(wave64));
        for (s0, s1) in [
            (0, 0),
            (5, 3),
            (9, 40),
            (40, 127),
            (FEW + 9, 2),
            (TRIES + 7, 33),
        ] {
            for (shader, s2, tries) in [
                (&plain, threads, TRIES),
                (&few, threads, FEW),
                (&gated, 0, TRIES),
                (&gated, 20, TRIES),
                (&gated, 41, TRIES),
            ] {
                let s = [s0, s1, s2];
                let got = run(&host, shader, s, 256);
                for t in 0..threads {
                    assert_eq!(
                        (got[t as usize], got[128 + t as usize]),
                        poll_wants(t, wave, s, tries),
                        "wave64={wave64} s={s:?} tries={tries}: thread {t}"
                    );
                }
            }
        }
    }
    eprintln!("ran on the hardware device: bounded polls, wave64 in halves and wave32");
}

/// On the hardware device: each console wave's OR, whatever lanes give 0.
#[cfg(windows)]
#[test]
fn reductions_that_end_a_pass_give_the_console_s_results() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some(host) = hardware() else { return };
    let unrestored = translated(&native(&unrestored_reduction(), false, 64)).expect("translate");
    let halves = translated(&native(&halves_reduction(), true, 128)).expect("translate");
    for s0 in [0, 1, 17, 31, 32, 33, 40, 63, 64, 65, 100, 128] {
        let got = run(&host, &unrestored, [s0, 0, 0], 64);
        let mut want = vec![0u32; 64];
        want[0] = reduced(0, 32, s0);
        want[1] = reduced(1, 32, s0);
        assert_eq!(got, want, "never restored, s0={s0}");
        let got = run(&host, &halves, [s0, 0, 0], 128);
        let want: Vec<u32> = (0..128).map(|t| reduced(t / 64, 64, s0)).collect();
        assert_eq!(got, want, "wave64 in halves, s0={s0}");
    }
    eprintln!("ran on the hardware device: reductions that end a pass");
}

/// On the hardware device: the channels a store leaves out keep what the
/// texel held, 16-bit data is f16 in the float image, and the counter's
/// reset stores the low 16 bits as an integer.
#[cfg(windows)]
#[test]
fn image_stores_of_some_channels_give_the_console_s_results() {
    use super::host::{Dispatch, Root, View};
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some(host) = hardware() else { return };
    let shader =
        translated(&native_with(&partial_stores(), false, 16, &IMAGES)).expect("translate");
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let (root, pso) = host
        .compute_pipeline(&shader)
        .unwrap_or_else(|e| panic!("compute_pipeline failed: {e}"));
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(constants(&shader, [0; 3])))],
                views: &[View::TextureUav(4, 4), View::UintTextureUav(4, 4)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    for t in 0..16usize {
        let texel: Vec<f32> = out[0][4 * t..4 * t + 4]
            .iter()
            .map(|&w| f32::from_bits(w))
            .collect();
        assert_eq!(texel, [0.5, 20.0 + t as f32, -2.0, 7.0], "float texel {t}");
        assert_eq!(out[1][t], 0x1234, "counter texel {t}");
    }
    eprintln!("ran on the hardware device: image stores of some channels");
}

/// A writable image's size query (`image_get_resinfo` at level 0), its
/// width and height then stored as floats into every texel of the same
/// image, of 4x4 texels (one per thread).
fn writable_size() -> Vec<u32> {
    vec![
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x3602_0083, // v_and_b32 v1, 3, v0
        0x2C04_0082, // v_lshrrev_b32 v2, 2, v0
        0x7E06_0280, // v_mov_b32 v3, 0
        0xF038_1308,
        0x0003_0403, // image_get_resinfo v[4:5], v3, s[12:19] dmask:0x3 dim:2D unorm
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E08_0D04, // v_cvt_f32_u32 v4, v4
        0x7E0A_0D05, // v_cvt_f32_u32 v5, v5
        0x7E0C_0280, // v_mov_b32 v6, 0
        0x7E0E_0280, // v_mov_b32 v7, 0
        0xF020_1F08,
        0x0003_0401, // image_store v[4:7], v[1:2], s[12:19] dmask:0xf dim:2D unorm
    ]
}

/// [`translated`], with the size query on a writable image at level 0 or
/// at the undefined level DXC writes.
fn translated_size(level0: bool) -> HostShader {
    let bytes = dxbc::build(&[]);
    let c = dxbc::Container::parse(&bytes).expect("container");
    let options = Options {
        narrow_waves: true,
        uav_size_level0: level0,
        ..Options::default()
    };
    super::translate_program(&c, &native_with(&writable_size(), false, 16, &[(0, 8)]), options, None)
        .expect("translate")
}

/// The switch changes the translation (the level of the size query), and
/// only it.
#[test]
fn a_writable_image_s_size_query_asks_for_level_0_with_the_switch() {
    let t = text(&[writable_size(), vec![S_ENDPGM]].concat(), false);
    assert_eq!(t[5], "image_get_resinfo v[4:5], v3, s[12:19] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm");
    assert_eq!(t[11], "image_store v[4:7], v[1:2], s[12:19] dmask:0xf dim:SQ_RSRC_IMG_2D unorm");
    assert_ne!(translated_size(true).dxil, translated_size(false).dxil);
    assert_eq!(translated_size(true).dxil, translated_size(true).dxil);
}

/// On the hardware device: at level 0 the size query gives the image's
/// size (4x4). What the undefined level gives is printed (D3DMetal: 0, the
/// halfwave50 fork found).
#[cfg(windows)]
#[test]
fn a_writable_image_s_size_at_level_0_is_its_size_on_the_device() {
    use super::host::{Dispatch, Root, View};
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some(host) = hardware() else { return };
    let size = |level0: bool| -> Vec<f32> {
        let shader = translated_size(level0);
        if let Ok(mut v) =
            super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil.clone())
                .expect("DXIL validation failed");
        }
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("compute_pipeline failed: {e}"));
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(constants(&shader, [0; 3])))],
                    views: &[View::TextureUav(4, 4)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        out[0][..64].iter().map(|&w| f32::from_bits(w)).collect()
    };
    let at0 = size(true);
    for t in 0..16 {
        assert_eq!(at0[4 * t..4 * t + 4], [4.0, 4.0, 0.0, 0.0], "texel {t}");
    }
    let undefined = size(false);
    eprintln!("a writable image's size at the undefined level: {:?}", &undefined[..2]);
}
