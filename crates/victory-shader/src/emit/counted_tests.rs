//! Programs made for dispatches counted in threads (`lower::counted`), run
//! on the host GPU: the threads past a count leave, the others see only
//! each other. Under Wine the host is D3DMetal, whose waves are 32 lanes:
//! the wave64 programs then run as halves of console waves.

use super::host::{Dispatch, Host, Root, View};
use super::{DISPATCH_THREADS, HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, range, resource};

const S_ENDPGM: u32 = 0xBF81_0000;
/// `v_lshlrev_b32 vdst, k, vsrc`.
const fn v_lshlrev(vdst: u32, k: u32, vsrc: u32) -> u32 {
    0x3400_0000 | vdst << 17 | vsrc << 9 | (0x80 + k)
}
/// `v_add_nc_u32 vdst, va, vb`.
const fn v_add(vdst: u32, va: u32, vb: u32) -> u32 {
    0x4A00_0000 | vdst << 17 | vb << 9 | (0x100 + va)
}
const V_MOV_V2_1: u32 = 0x7E04_0281; // v_mov_b32 v2, 1
const V_MOV_V2_S0: u32 = 0x7E04_0200; // v_mov_b32 v2, s0
const V_AND_V2_1_V0: u32 = 0x3604_0081; // v_and_b32 v2, 1, v0
const CMP_S0_V2_EQ_0: [u32; 2] = [0xD4C2_0000, 0x0002_0480]; // v_cmp_eq_u32 s[0:1], 0, v2
const BCNT_S0: u32 = 0xBE80_1000; // s_bcnt1_i32_b64 s0, s[0:1]
const BCNT_S0_EXEC: u32 = 0xBE80_107E; // s_bcnt1_i32_b64 s0, exec
/// `buffer_atomic_add v2, v1, s[8:11], 0 offen`: adds v2 to the dword at
/// byte v1 of the raw buffer.
const ATOMIC_ADD_V2_AT_V1: [u32; 2] = [0xE0C8_1000, 0x8002_0201];

/// A compute program of `threads` (thread ids in v0 and, in two dimensions,
/// v1) with four root constants in s0-s3 and a table in s4 whose first
/// view is a raw buffer, its descriptor loaded into s[8:11] before `body`.
/// Made to leave out the threads past a dispatch's counts, for 32-lane host
/// waves with `narrow`.
fn program(body: &[u32], wave64: bool, threads: [u32; 3], narrow: bool) -> HostShader {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xFA00_0000,
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
    let two = threads[1] > 1;
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![param(RootType::Constants, 0, 4, 0), param(RootType::Table, 4, 1, 1)],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: u32::from(two) << 11 | 5 << 1,
            threads,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let (p, st) = range::build(&native.code, native.wave64, native.vgprs).expect("build");
    let st = st.expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("raise");
    let lowered = lower::lower_with_layout(
        &lower::Parts {
            psv: None,
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            dropped_rows: Vec::new(),
            mesh_stride: 0,
            atomic_locks: false,
            bias_as_gradients: false,
            narrow_waves: narrow,
            view_marks: false,
        },
        false,
        false,
        false,
        lower::Extra { thread_counts: true },
    )
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    shader
}

/// The hardware device and `shader`'s pipeline on it; `None` (with a note)
/// without a device, or with `may_refuse` where the device does not take
/// the pipeline (a wave size it does not have).
fn pipeline(
    shader: &HostShader,
    may_refuse: bool,
) -> Option<(Host, windows::Win32::Graphics::Direct3D12::ID3D12RootSignature, windows::Win32::Graphics::Direct3D12::ID3D12PipelineState)> {
    let host = match Host::new(false) {
        Ok(host) => host,
        Err(e) => {
            eprintln!("no SM 6.6 hardware device ({e}); skipped");
            return None;
        }
    };
    match host.compute_pipeline(shader) {
        Ok((root, pso)) => Some((host, root, pso)),
        Err(e) if may_refuse => {
            eprintln!("the host does not take the pipeline ({e}); skipped");
            None
        }
        Err(e) => panic!("compute_pipeline failed: {e}"),
    }
}

/// Dispatches `groups` of `shader` with the thread counts `counts` (0: no
/// count on that axis): the first `words` dwords of its buffer after.
fn run(
    (host, root, pso): &(Host, windows::Win32::Graphics::Direct3D12::ID3D12RootSignature, windows::Win32::Graphics::Direct3D12::ID3D12PipelineState),
    shader: &HostShader,
    groups: [u32; 3],
    counts: [u32; 3],
    words: u32,
) -> Vec<u32> {
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match u.root {
            root if root == DISPATCH_THREADS => counts[u.dword as usize],
            _ => 0,
        };
    }
    let out = host
        .dispatch(
            root,
            pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(words)],
                samplers: &[],
                groups,
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    out[0][..words as usize].to_vec()
}

/// The count word of axis `axis`.
fn has_word(shader: &HostShader, axis: u32) -> bool {
    shader.user_data.iter().any(|u| u.root == DISPATCH_THREADS && u.dword == axis)
}

/// Each thread that runs adds 1 to the dword of its number in the group.
/// Of two groups of 64, the threads below the count run, every thread
/// without one.
#[test]
fn the_threads_past_a_count_do_not_run() {
    let body = [v_lshlrev(1, 2, 0), V_MOV_V2_1, ATOMIC_ADD_V2_AT_V1[0], ATOMIC_ADD_V2_AT_V1[1]];
    let shader = program(&body, false, [64, 1, 1], false);
    assert!(has_word(&shader, 0));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(made) = pipeline(&shader, false) else { return };
    for count in [0u32, 1, 7, 63, 64, 65, 74, 127, 128] {
        let out = run(&made, &shader, [2, 1, 1], [count, 0, 0], 64);
        let runs = if count == 0 { 128 } else { count };
        for (i, &word) in out.iter().enumerate() {
            let i = i as u32;
            assert_eq!(word, u32::from(i < runs) + u32::from(i + 64 < runs), "count {count}, thread {i}");
        }
    }
    eprintln!("ran on the hardware device: threads past a count left, in two groups of 64");
}

/// Groups of 8 x 8 threads, counted on both axes: a thread runs where it
/// is below the count on each.
#[test]
fn each_axis_is_cut_at_its_count() {
    let body = [
        v_lshlrev(3, 3, 1), // v3 = v1 * 8
        v_add(3, 0, 3),     // v3 = v0 + v3
        v_lshlrev(1, 2, 3), // v1 = v3 * 4
        V_MOV_V2_1,
        ATOMIC_ADD_V2_AT_V1[0],
        ATOMIC_ADD_V2_AT_V1[1],
    ];
    let shader = program(&body, false, [8, 8, 1], false);
    assert!(has_word(&shader, 0) && has_word(&shader, 1) && !has_word(&shader, 2));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(made) = pipeline(&shader, false) else { return };
    for counts in [[0u32, 0], [11, 13], [16, 9], [3, 0], [0, 15]] {
        let out = run(&made, &shader, [2, 2, 1], [counts[0], counts[1], 0], 64);
        for (i, &word) in out.iter().enumerate() {
            let (tx, ty) = (i as u32 % 8, i as u32 / 8);
            let below = |count: u32, at: u32| count == 0 || at < count;
            let want = (0..2u32)
                .flat_map(|gx| (0..2u32).map(move |gy| (gx, gy)))
                .filter(|&(gx, gy)| below(counts[0], gx * 8 + tx) && below(counts[1], gy * 8 + ty))
                .count() as u32;
            assert_eq!(word, want, "counts {counts:?}, thread {tx} {ty}");
        }
    }
    eprintln!("ran on the hardware device: groups of 8 x 8 cut on both axes");
}

/// What each thread of a console wave of which `live` lanes run adds in
/// [`lanes_counted_over_a_wave`]: the lanes of an even number among them.
fn even(live: u32) -> u32 {
    live.div_ceil(2)
}

/// A wave64 program in 32-lane host waves: each thread adds, to the dword
/// of its number in the group, how many lanes of its console wave have an
/// even number (a mask read as a number, which the halves exchange) or, in
/// the second program, how many lanes started (EXEC on entry). Past a count
/// a half may have no thread, or some: the number is that of the lanes the
/// console would have launched. Groups of one console wave and of four.
#[test]
fn lanes_counted_over_a_wave_are_those_below_the_count() {
    let add = [V_MOV_V2_S0, ATOMIC_ADD_V2_AT_V1[0], ATOMIC_ADD_V2_AT_V1[1]];
    let mut evens = vec![v_lshlrev(1, 2, 0), V_AND_V2_1_V0];
    evens.extend(CMP_S0_V2_EQ_0);
    evens.push(BCNT_S0);
    evens.extend(add);
    let mut started = vec![v_lshlrev(1, 2, 0), BCNT_S0_EXEC];
    started.extend(add);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    let programs: [(&str, &Vec<u32>, fn(u32) -> u32); 2] =
        [("even lanes", &evens, even), ("lanes on entry", &started, |n| n)];
    for (what, body, of) in programs {
        for (threads, groups) in [(64u32, 2u32), (256, 1)] {
            let shader = program(body, true, [threads, 1, 1], true);
            assert!(has_word(&shader, 0), "{what}, groups of {threads}");
            let Some(made) = pipeline(&shader, true) else { continue };
            let all = threads * groups;
            for count in [0u32, 1, 10, 31, 32, 33, 64, 74, 96, 100, 127, 128, 129, 200, 255, 256] {
                if count > all {
                    continue;
                }
                let out = run(&made, &shader, [groups, 1, 1], [count, 0, 0], threads);
                let runs = if count == 0 { all } else { count };
                for (i, &word) in out.iter().enumerate() {
                    let i = i as u32;
                    // Every group's thread of this number, each with the
                    // live lanes of its console wave.
                    let want: u32 = (0..groups)
                        .map(|g| g * threads + i)
                        .filter(|&at| at < runs)
                        .map(|at| of((runs - at / 64 * 64).min(64)))
                        .sum();
                    assert_eq!(word, want, "{what}, groups of {threads}, count {count}, thread {i}");
                }
            }
            ran += 1;
        }
    }
    if ran > 0 {
        eprintln!("ran on the hardware device: {ran} wave64 programs counted lanes past a count of threads");
    }
}
