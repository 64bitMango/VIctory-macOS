//! A least amount of indirect light.
//!
//! the title's indirect diffuse light comes from probes it fills
//! by tracing rays, which do not run here yet: its program that resolves
//! them (`a title shader_*`) leaves next to nothing, and
//! whatever the sun does not reach directly is black. Right after that
//! program, its output is replaced by a constant (it used to be a floor
//! under what the program wrote, but that is garbage: see the shader). The
//! constant is the same everywhere: the light in these images does not go with the title's exposure
//! (well-lit surfaces hold about the same in a shot at -5.5 EV and in one at
//! -9.75). The title's own lighting then applies it with the surfaces'
//! colors. Raised, not added: the result is the next frame's history.
//!
//! The level and colour follow the sky the title draws for its reflections
//! ([`follow`]; `VICTORY_AMBIENT_SKY=off` keeps the constant).
//! `VICTORY_AMBIENT=<level>` sets the constant (0.6, 0 for none) and, unless
//! `VICTORY_AMBIENT_SKY` is set too, has it used instead of the sky; `off`
//! keeps the title's own output.
//! shortcut: the same light everywhere, from everywhere; gone once rays run.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::{HostList, host};
use crate::object::This;

/// The programs it follows are named by `VICTORY_HOOK_AMBIENT`
/// ([`crate::pipeline::Hook`]).
/// Where they write the indirect diffuse light (R9G9B9E5 as 32-bit
/// integers): root parameter and place in its table.
pub(crate) const LIGHT: (u8, u32) = (1, 176);

/// 0.6 looked balanced in run a2. `VICTORY_AMBIENT=off` leaves what the
/// title's resolve wrote; 0 is no indirect diffuse light at all.
const DEFAULT: f32 = 0.6;

/// What `VICTORY_AMBIENT` gives: the level (negative for `off`), and
/// whether it named one.
fn set_level() -> (f32, bool) {
    static LEVEL: OnceLock<(f32, bool)> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        let set = std::env::var("VICTORY_AMBIENT").unwrap_or_default();
        if set.trim() == "off" {
            return (-1.0, false);
        }
        let named = set.trim().parse().ok().filter(|l: &f32| l.is_finite() && *l >= 0.0);
        (named.unwrap_or(DEFAULT), named.is_some())
    })
}

/// The floor; `None` if off.
pub(crate) fn level() -> Option<f32> {
    let (level, _) = set_level();
    (level >= 0.0).then_some(level)
}

/// The light's colour: warm white, or what `VICTORY_AMBIENT_TINT=r,g,b`
/// gives. With `VICTORY_AMBIENT_FILE=<file>` (diagnostics builds) the file's
/// `level r g b` is read again twice a second, to tune both by eye in a
/// running game.
fn tint() -> [f32; 3] {
    let live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, tint)) = *live {
        return tint;
    }
    static SET: OnceLock<[f32; 3]> = OnceLock::new();
    *SET.get_or_init(|| {
        let set: Vec<f32> = std::env::var("VICTORY_AMBIENT_TINT")
            .unwrap_or_default()
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect();
        match set[..] {
            [r, g, b] => [r, g, b],
            _ => [1.0, 0.98, 0.94],
        }
    })
}

/// The level and colour last read from `VICTORY_AMBIENT_FILE`, if any.
static LIVE: std::sync::Mutex<Option<(f32, [f32; 3])>> = std::sync::Mutex::new(None);

/// The program that writes the title's reflections for the last time in a
/// frame, by the start of its name, and where it writes them (R9G9B9E5 as
/// 32-bit integers): root parameter and place in its table. The lighting
/// programs read that image next.
///
/// Without rays the reflections are the sky's, unoccluded, wherever a
/// surface is glossy: indoors a wooden floor, a counter and glass tops wore
/// a white glaze five to ten times brighter than anything in the room (the
/// fork's run ciel1, 2026-11-25; it did not move with the ambient level).
/// They are held to a part of the light around ([`keep`]). The program is
/// named by `VICTORY_HOOK_REFLECTED` ([`crate::pipeline::Hook`]).
pub(crate) const REFLECTIONS: (u8, u32) = (1, 40);

/// The most a reflection may hold, as a part of the ambient light's level:
/// `VICTORY_AMBIENT_REFLECT=<part>` (0.5 by default; 0 leaves the title's
/// reflections alone); a ninth number in the file sets it in a running game.
/// shortcut: stands in for what the title's traced reflections would see.
static KEEP: std::sync::Mutex<Option<f32>> = std::sync::Mutex::new(None);

/// The least a reflection holds, as a part of the light around: a tenth
/// number in the file (0, the default, leaves dark reflections dark).
/// shortcut: without rays, what should mirror the sky (still water) sees
/// nothing and is black; this gives it a part of the ambient light instead.
static LEAST: std::sync::Mutex<f32> = std::sync::Mutex::new(0.0);

fn keep() -> f32 {
    if let Some(live) = *KEEP.lock().unwrap_or_else(|e| e.into_inner()) {
        return live;
    }
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| {
        std::env::var("VICTORY_AMBIENT_REFLECT")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|k: &f32| k.is_finite() && *k >= 0.0)
            .unwrap_or(0.5)
    })
}

/// The program that draws the sky as the title reflects it, named by
/// `VICTORY_HOOK_SKY` ([`crate::pipeline::Hook`]): a small image of the whole
/// sky, in the light's own units.

type Copy = std::sync::Arc<crate::resource::HostTexture>;

/// The sky dome that program last drew into: the host copy, held, since
/// the title's object may go at any time. Held too while the title draws no
/// sky: the light stays what the dome last was.
///
/// The program draws twice a frame, on two lists recorded at once: the dome
/// (128x128, R16G16B16A16_FLOAT) and another image of the sky (R11G11B10).
/// Following whichever was recorded last made the light jump between the
/// two, three times apart, from frame to frame (the fork's run don1,
/// 2026-11-25).
static SKY_IMAGE: std::sync::Mutex<Option<Copy>> = std::sync::Mutex::new(None);

/// The sky program draws into the title texture `image`.
pub(crate) fn sky_drawn(image: usize) {
    if let Some(copy) = crate::resource::host_copy(image as This) {
        hold_sky(&SKY_IMAGE, copy);
    }
}

/// `copy` becomes the sky `held`, if it is the dome.
fn hold_sky(held: &std::sync::Mutex<Option<Copy>>, copy: Copy) {
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    // The host may have made the dome typeless (castable formats).
    let dome = [DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_TYPELESS].contains(&d.Format);
    if !dome || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
        return;
    }
    let mut held = held.lock().unwrap_or_else(|e| e.into_inner());
    if !held.as_ref().is_some_and(|h| std::sync::Arc::ptr_eq(h, &copy)) {
        *held = Some(copy);
    }
}

/// The light follows the sky: its level and colour are worked out from the
/// sky image's mean. Hand-set levels that looked right on the fork were
/// 0.07 at night, 0.3 at dusk and 1.4 at noon, with sky means of 0.001 at
/// night and 0.65 at dawn; one level for all hours either drowned the
/// title's own shadows or left what the sun does not reach black. On by
/// default; `VICTORY_AMBIENT_SKY=off` keeps the constant, and so does a
/// level named by `VICTORY_AMBIENT` alone; `gain,least,power` sets it; a
/// file of two or three numbers, or six or seven with the shading's, does
/// so in a running game. Before the title has drawn a sky the constant
/// stands.
/// shortcut: the sky's mean stands in for the light the title's traced
/// probes would gather; indoors it is the sky's too.
static FOLLOW: std::sync::Mutex<Option<Option<[f32; 3]>>> = std::sync::Mutex::new(None);

/// The power the level takes of the sky's mean unless set.
const POWER: f32 = 0.4;

/// Gain, least level and power: the level is `gain * mean^power + least`.
/// The power is under one because the title's exposure follows the light:
/// the sky's mean goes from 0.001 at night to several units by day, the
/// levels that look right only from 0.07 to 1.4.
fn follow() -> Option<[f32; 3]> {
    if let Some(live) = *FOLLOW.lock().unwrap_or_else(|e| e.into_inner()) {
        return live;
    }
    static SET: OnceLock<Option<[f32; 3]>> = OnceLock::new();
    *SET.get_or_init(|| follow_set(std::env::var("VICTORY_AMBIENT_SKY").ok().as_deref(), set_level().1))
}

/// [`follow`] for what `VICTORY_AMBIENT_SKY` holds, `level_named` if
/// `VICTORY_AMBIENT` names a level.
fn follow_set(set: Option<&str>, level_named: bool) -> Option<[f32; 3]> {
    match set.map(str::trim) {
        Some("off") => return None,
        // A level asked for is the level: the fixed light, as before.
        None if level_named => return None,
        _ => {}
    }
    let mut follow = [0.36, 0.05, POWER];
    for (to, v) in follow.iter_mut().zip(set.unwrap_or_default().split(',')) {
        if let Ok(v) = v.trim().parse() {
            *to = v;
        }
    }
    Some(follow)
}

/// Where the resolve reads the scene's depth (reversed, D32 with stencil):
/// root parameter and place in its table.
pub(crate) const DEPTH: (u8, u32) = (0, 464);

/// How much less of the light reaches creases and corners, worked out from
/// the depth image alone: strength (0 for none), the radius looked at in
/// metres, the near plane's distance, and the tangent of half the vertical
/// field of view (the last two are guesses the picture is not sensitive
/// to). `VICTORY_AMBIENT_SHADE=s,r,n,t` sets them; the file's values after
/// its first four replace them.
/// shortcut: stands in for the occlusion the title's traced probes carry.
static SHADE: std::sync::Mutex<Option<[f32; 4]>> = std::sync::Mutex::new(None);

fn shade() -> [f32; 4] {
    if let Some(live) = *SHADE.lock().unwrap_or_else(|e| e.into_inner()) {
        return live;
    }
    static SET: OnceLock<[f32; 4]> = OnceLock::new();
    *SET.get_or_init(|| {
        let mut shade = [1.0, 0.6, 0.1, 0.45];
        let set = std::env::var("VICTORY_AMBIENT_SHADE").unwrap_or_default();
        for (to, v) in shade.iter_mut().zip(set.split(',')) {
            if let Ok(v) = v.trim().parse() {
                *to = v;
            }
        }
        shade
    })
}

/// The level from `VICTORY_AMBIENT_FILE`, if it gave one; starts the thread
/// that reads the file.
fn live_level() -> Option<f32> {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        if !cfg!(feature = "diagnostics") {
            return;
        }
        let Some(file) = std::env::var_os("VICTORY_AMBIENT_FILE") else { return };
        let _ = std::thread::Builder::new().name("victory-ambient".into()).spawn(move || {
            loop {
                let read: Vec<f32> = std::fs::read_to_string(&file)
                    .unwrap_or_default()
                    .split_whitespace()
                    .filter_map(|v| v.parse().ok())
                    .collect();
                // Two, three, six or seven numbers: gain, least, a power if
                // their count is odd, then the shading's four.
                if let [gain, least, ref rest @ ..] = read[..]
                    && matches!(rest.len(), 0 | 1 | 4 | 5)
                {
                    let (power, rest) = match rest.len() % 2 {
                        1 => (rest[0], &rest[1..]),
                        _ => (POWER, rest),
                    };
                    if let [s, r, n, t] = rest[..] {
                        *SHADE.lock().unwrap_or_else(|e| e.into_inner()) = Some([s, r, n, t]);
                    }
                    let new = Some(Some([gain, least, power]));
                    let mut follow = FOLLOW.lock().unwrap_or_else(|e| e.into_inner());
                    if *follow != new {
                        tracing::info!(gain, least, power, "ambient light follows the sky, set from the file");
                        *follow = new;
                    }
                } else if let [level, r, g, b, ref rest @ ..] = read[..] {
                    *FOLLOW.lock().unwrap_or_else(|e| e.into_inner()) = Some(None);
                    if let [s, r, n, t, ref last @ ..] = rest[..] {
                        *SHADE.lock().unwrap_or_else(|e| e.into_inner()) = Some([s, r, n, t]);
                        if let [k, ref floor @ ..] = last[..] {
                            *KEEP.lock().unwrap_or_else(|e| e.into_inner()) = Some(k.max(0.0));
                            *LEAST.lock().unwrap_or_else(|e| e.into_inner()) =
                                floor.first().map_or(0.0, |f| f.max(0.0));
                        }
                    }
                    let new = Some((level.max(0.0), [r, g, b]));
                    let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
                    if *live != new {
                        tracing::info!(level, r, g, b, "ambient light set from the file");
                        *live = new;
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        });
    });
    LIVE.lock().unwrap_or_else(|e| e.into_inner()).map(|(level, _)| level)
}

/// Works the light out from the sky, once for the whole image: its three
/// channels into the three texels of `around`.
///
/// The fork's pass takes the sky's mean again at every texel of the light
/// (64 reads each, at 2560x1440); one group of one thread does it here.
/// The sky's size comes in the constants (no `GetDimensions` under Wine).
const SKY_SOURCE: &str = r#"
cbuffer Sky : register(b0) {
    float gain; float least; float power;
    uint sw; uint sh;
};
Texture2DArray<float4> sky : register(t0);
RWTexture2D<float> around : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    float3 sum = 0.0;
    const int N = 8;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            // The sun's own texels are held down: it lights by itself.
            float3 there = sky.Load(int4((x * 2 + 1) * sw / (N * 2), (y * 2 + 1) * sh / (N * 2), 0, 0)).rgb;
            sum += clamp(there, 0.0, 8.0);
        }
    float3 mean = sum / (N * N);
    float m = dot(mean, float3(0.2126, 0.7152, 0.0722));
    // The sky's colour, half washed out (the ground lights too); a dark
    // sky has none worth following.
    const float3 NIGHT = float3(0.6, 0.75, 1.0);
    float3 hue = lerp(NIGHT, lerp(float3(1.0, 1.0, 1.0), mean / max(m, 1e-6), 0.5), saturate(m / 0.05));
    float3 is = hue * (gain * pow(max(m, 0.0), power) + least);
    around[uint2(0, 0)] = is.r;
    around[uint2(1, 0)] = is.g;
    around[uint2(2, 0)] = is.b;
}
"#;

const SOURCE: &str = r#"
cbuffer Floor : register(b0) {
    float level; float3 tint;
    float strength; float radius; float near; float tan_y;
    uint width; uint height;
    uint follows;
    float held; float floor_part;
};
RWTexture2D<uint> light : register(u0);
// The light worked out from the sky (`SKY_SOURCE`), when it follows it.
RWTexture2D<float> around : register(u1);
float3 from_around() {
    if (follows == 0) return tint * level;
    return float3(around[uint2(0, 0)], around[uint2(1, 0)], around[uint2(2, 0)]);
}
Texture2D<float> depth : register(t0);
// The point seen at a texel, in metres from the eye (x right, y up, z away).
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), int2(size) - 1), 0));
    float z = near / max(d, 1e-7);
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}
// How much of the light from around reaches the texel: 1 in the open.
float open_to_light(int2 p, float2 size) {
    float3 c = seen(p, size);
    if (c.z > 400.0) return 1.0;
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx));
    if (dot(n, c) > 0.0) n = -n;
    float px = radius * size.y / (2.0 * tan_y * c.z);
    if (px < 1.5) return 1.0;
    px = min(px, 128.0);
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
    float sum = 0.0;
    const int N = 16;
    for (int i = 0; i < N; i++) {
        float a = i * 2.399963 + noise * 6.2831853;
        float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
        float3 v = seen(p + int2(round(o)), size) - c;
        float vv = dot(v, v);
        float over = dot(v, n) - 0.004 * c.z - 0.005;
        sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
    }
    return saturate(1.0 - strength * 2.0 * sum / N);
}
float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float most = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(most, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (most * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    // The size comes in the constants: `GetDimensions` on the image does
    // not build under Wine (vkd3d-shader and D3DMetal).
    if (at.x >= width || at.y >= height) return;
    // What the resolve left is not kept: without rays its inputs are never
    // written and it holds whatever their memory did (bright magenta on
    // translucent surfaces, blocks of colours indoors: run v6, 2026-11-25).
    if (held > 0.0) {
        // Reflections: no brighter than a part of the light around, and no
        // darker than another.
        const float3 LUMA = float3(0.2126, 0.7152, 0.0722);
        float3 seen_there = unpack(light[at.xy]);
        float3 all_round = from_around();
        float most = dot(all_round, LUMA) * held;
        float has = dot(seen_there, LUMA);
        float3 floor_light = all_round * floor_part;
        if (has > most) light[at.xy] = pack(seen_there * (most / has));
        else if (has < dot(floor_light, LUMA)) light[at.xy] = pack(floor_light);
        return;
    }
    float open = strength > 0.0 ? open_to_light(int2(at.xy), float2(width, height)) : 1.0;
    light[at.xy] = pack(from_around() * open);
}
"#;

/// Views, used in turn: more than the passes of every list in flight.
const SLOTS: u32 = 1024;

/// Views a pass takes: the light, the light from the sky and the depth for
/// the main program, then the light from the sky and the sky for the one
/// that works it out.
const VIEWS: u32 = 5;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// The program that works the light out from the sky.
    sky: (ID3D12RootSignature, ID3D12PipelineState),
    /// What it writes: three texels of 32-bit floats, always in the
    /// unordered-access state (each list writes it before it reads it).
    around: ID3D12Resource,
    views: ID3D12DescriptorHeap,
    step: u32,
    next: AtomicU32,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

/// Built by [`crate::compute::pipeline`], whose root signature is made
/// apart from the shader: Wine's compiler (vkd3d-shader) writes none for
/// `[RootSignature]`.
fn make(device: &ID3D12Device) -> windows::core::Result<Pass> {
    use crate::compute::{Param, srv, uav};
    let params = [Param::Constants { register: 0, count: 13 }, Param::Table(&[uav(0, 2), srv(0, 1)])];
    let (root, pso) = crate::compute::pipeline(SOURCE, s!("ambient"), 0, &params)?;
    let params = [Param::Constants { register: 0, count: 5 }, Param::Table(&[uav(0, 1), srv(0, 1)])];
    let sky = crate::compute::pipeline(SKY_SOURCE, s!("ambient sky"), 0, &params)?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 3,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut around: Option<ID3D12Resource> = None;
    // SAFETY: plain creations.
    unsafe {
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        let views = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: SLOTS * VIEWS,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })?;
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut around,
        )?;
        Ok(Pass {
            root,
            pso,
            sky,
            around: around.ok_or_else(windows::core::Error::empty)?,
            views,
            step: device.GetDescriptorHandleIncrementSize(kind),
            next: AtomicU32::new(0),
        })
    }
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        ..Default::default()
    };
    // SAFETY: an open list; the barrier names no resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Raises what `light` holds to the floor, in `list`, after the program that
/// wrote it as an image. Whether it recorded anything (the list's bindings
/// are then ours).
pub(crate) fn raise(list: &HostList, light: This, depth: Option<This>) -> bool {
    let Some(level) = level() else {
        return false;
    };
    LIGHT_RESOURCE.store(light as usize, Ordering::Relaxed);
    let Some(light) = crate::resource::host_resource(light) else {
        return false;
    };
    let sky = sky();
    record(list, &light, depth, level, sky.as_ref().map(|(copy, f)| (copy, *f)), None)
}

/// The sky dome with its gain, least level and power, if the light follows
/// it and the title has drawn it.
fn sky() -> Option<(Copy, [f32; 3])> {
    let follow = follow()?;
    Some((SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner()).clone()?, follow))
}

/// Holds the reflections in `image` to a part of the ambient light, in
/// `list`, after the program that wrote them last. Whether it recorded
/// anything (the list's bindings are then ours).
pub(crate) fn hold(list: &HostList, image: This) -> bool {
    let (held, Some(level)) = (keep(), level()) else {
        return false;
    };
    if held <= 0.0 {
        return false;
    }
    let Some(image) = crate::resource::host_resource(image) else {
        return false;
    };
    let floor = *LEAST.lock().unwrap_or_else(|e| e.into_inner());
    let sky = sky();
    if !record(list, &image, None, level, sky.as_ref().map(|(copy, f)| (copy, *f)), Some([held, floor])) {
        return false;
    }
    static HELD: std::sync::Once = std::sync::Once::new();
    HELD.call_once(|| tracing::info!(held, floor, "reflections held to the ambient light"));
    true
}

/// The title's indirect diffuse light image the pass last wrote
/// (diagnostics: `draw::probe_constants` lists the programs that read it).
pub(crate) static LIGHT_RESOURCE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The pass, made once.
fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        make(&host()?.device)
            .inspect_err(|e| tracing::warn!(error = %e, "no ambient light"))
            .ok()
    })
    .as_ref()
}

/// Records the pass over the host texture `light`: at `level`, or as the
/// held `sky` gives with its gain, least level and power. With
/// `reflections`, the image is the title's reflections instead, kept
/// between the second and the first of these parts of that light.
fn record(
    list: &HostList,
    light: &ID3D12Resource,
    depth: Option<This>,
    level: f32,
    sky: Option<(&Copy, [f32; 3])>,
    reflections: Option<[f32; 2]>,
) -> bool {
    let (Some(pass), Some(host)) = (pass(), host()) else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { light.GetDesc() };
    // Said once for the light and once for the reflections.
    static SAID: [std::sync::Once; 2] = [std::sync::Once::new(), std::sync::Once::new()];
    let once = &SAID[usize::from(reflections.is_some())];
    // A texture other views read as R9G9B9E5 may be made typeless on the
    // host (castable formats): a 32-bit unsigned view of it is the same.
    if d.Format != DXGI_FORMAT_R32_UINT && d.Format != DXGI_FORMAT_R32_TYPELESS {
        once.call_once(|| {
            tracing::warn!(format = d.Format.0, reflections = reflections.is_some(), "no ambient light: another format")
        });
        return false;
    }
    once.call_once(|| {
        tracing::info!(
            level,
            follows_sky = sky.is_some(),
            reflections = reflections.is_some(),
            width = d.Width,
            height = d.Height,
            "ambient light"
        );
    });
    let slot = pass.next.fetch_add(1, Ordering::Relaxed) % SLOTS * VIEWS;
    // The sky's size, and it made readable: the list holds the copy.
    let sky = sky.map(|(copy, f)| {
        // SAFETY: a live resource.
        let d = unsafe { copy.texture.GetDesc() };
        static SEEN: std::sync::Once = std::sync::Once::new();
        SEEN.call_once(|| {
            tracing::info!(
                format = d.Format.0,
                width = d.Width,
                height = d.Height,
                slices = d.DepthOrArraySize,
                "the sky image the ambient light would follow"
            );
            tracing::info!(gain = f[0], least = f[1], power = f[2], "ambient light follows the sky");
        });
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0;
        if let Some(barrier) = list.move_texture(copy, read as u32) {
            // SAFETY: an open list and a live texture the list holds.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
        }
        (&copy.texture, f, [d.Width as u32, d.Height])
    });
    list.keep(light);
    // The depth the resolve just read, in the state it read it in; of the
    // light's size, or the shading is left out.
    let depth = depth.and_then(crate::resource::host_resource).filter(|t| {
        // SAFETY: a live resource.
        let t = unsafe { t.GetDesc() };
        t.Format == DXGI_FORMAT_R32G8X24_TYPELESS && (t.Width, t.Height) == (d.Width, d.Height)
    });
    if let Some(depth) = &depth {
        list.keep(depth);
    }
    let l = &list.list;
    uav_barrier(l);
    // SAFETY: an open list; a live texture the list keeps, in the
    // unordered-access state as the program before left it; a view in a slot
    // no list in flight reads.
    unsafe {
        let cpu = pass.views.GetCPUDescriptorHandleForHeapStart();
        let gpu = pass.views.GetGPUDescriptorHandleForHeapStart();
        let at = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + (slot * pass.step) as usize,
        };
        let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        host.device
            .CreateUnorderedAccessView(light, None::<&ID3D12Resource>, Some(&view), at);
        let nth = |n: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at.ptr + (n * pass.step) as usize };
        let worked_out = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        for n in [1, 3] {
            host.device
                .CreateUnorderedAccessView(&pass.around, None::<&ID3D12Resource>, Some(&worked_out), nth(n));
        }
        let seen = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
            },
        };
        host.device.CreateShaderResourceView(depth.as_ref(), Some(&seen), nth(2));
        l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        let table = |n: u32| D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu.ptr + u64::from((slot + n) * pass.step),
        };
        if let Some((sky, [gain, least, power], [sw, sh])) = sky {
            // One slice of the dome, in its float format whatever the host
            // made it (typeless, if other views cast it).
            let dome = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() },
                },
            };
            host.device.CreateShaderResourceView(sky, Some(&dome), nth(4));
            l.SetComputeRootSignature(&pass.sky.0);
            l.SetPipelineState(&pass.sky.1);
            let constants = [gain.to_bits(), least.to_bits(), power.to_bits(), sw, sh];
            l.SetComputeRoot32BitConstants(0, 5, constants.as_ptr().cast(), 0);
            l.SetComputeRootDescriptorTable(1, table(3));
            l.Dispatch(1, 1, 1);
            uav_barrier(l);
        }
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        let (width, height) = (d.Width as u32, d.Height);
        let [r, g, b] = tint();
        let level = live_level().unwrap_or(level);
        let [strength, radius, near, tan_y] = shade();
        let strength = if depth.is_some() { strength } else { 0.0 };
        let [held, floor] = reflections.unwrap_or([0.0, 0.0]);
        let constants = [
            level.to_bits(), r.to_bits(), g.to_bits(), b.to_bits(),
            strength.to_bits(), radius.to_bits(), near.to_bits(), tan_y.to_bits(),
            width, height,
            u32::from(sky.is_some()),
            held.to_bits(), floor.to_bits(),
        ];
        l.SetComputeRoot32BitConstants(0, 13, constants.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(1, table(0));
        l.Dispatch(width.div_ceil(8), height.div_ceil(8), 1);
    }
    uav_barrier(l);
    true
}

#[cfg(test)]
mod tests {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    use crate::host::Host;

    /// A texture of `format` for `flags`, in `state`.
    fn texture(
        host: &Host,
        (width, height, slices): (u32, u32, u16),
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: slices,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: flags,
            ..Default::default()
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut texture)
                .unwrap();
        }
        texture.unwrap()
    }

    /// A light image as the title's, in the state its programs leave it in.
    fn light(host: &Host, width: u32, height: u32) -> ID3D12Resource {
        let flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        texture(host, (width, height, 1), DXGI_FORMAT_R32_UINT, flags, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
    }

    /// A sky of one colour in `format`, as a host copy the title drew.
    fn sky(host: &Host, format: DXGI_FORMAT, slices: u16, colour: [f32; 3]) -> super::Copy {
        let state = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let sky = texture(host, (128, 128, slices), format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, state);
        let list = host.new_list().unwrap();
        let [r, g, b] = colour;
        // SAFETY: an open list and a live render target in that state.
        unsafe { list.list.ClearRenderTargetView(host.rtv(&sky, None), &[r, g, b, 1.0], None) };
        list.close();
        host.execute(&[&list]);
        std::sync::Arc::new(crate::resource::HostTexture::new(sky, state.0 as u32, Vec::new()))
    }

    /// Runs `list` after reading `image` back in it: its texels.
    fn texels(host: &Host, list: &crate::host::HostList, image: &ID3D12Resource, width: u32, height: u32) -> Vec<u32> {
        let pitch = (4 * width).next_multiple_of(256);
        let (out, data) = host.shared_buffer(u64::from(pitch * height)).unwrap();
        // SAFETY: an open list, live resources; the mapped buffer is read
        // after the list ran.
        unsafe {
            list.list.ResourceBarrier(&[
                crate::host::transition(image, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
                crate::host::transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
            ]);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&out),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R32_UINT,
                            Width: width,
                            Height: height,
                            Depth: 1,
                            RowPitch: pitch,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(image),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            list.list.ResourceBarrier(&[crate::host::transition(
                image,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            list.close();
            host.execute(&[list]);
            let mut all = Vec::new();
            for y in 0..height {
                for x in 0..width {
                    all.push(data.add((y * pitch + 4 * x) as usize).cast::<u32>().read_volatile());
                }
            }
            all
        }
    }

    /// R9G9B9E5 as the shader packs it.
    fn pack(c: [f32; 3]) -> u32 {
        let most = c[0].max(c[1]).max(c[2]);
        let mut e = (most.max(1e-9).log2().floor() as i32 + 16).clamp(0, 31);
        if most * 2f32.powi(24 - e) >= 511.5 {
            e = (e + 1).min(31);
        }
        let m = c.map(|c| ((c * 2f32.powi(24 - e) + 0.5) as u32).min(511));
        m[0] | m[1] << 9 | m[2] << 18 | (e as u32) << 27
    }

    fn unpack(v: u32) -> [f32; 3] {
        [v & 511, v >> 9 & 511, v >> 18 & 511].map(|m| m as f32 * 2f32.powi((v >> 27) as i32 - 24))
    }

    /// The light the sky program works out from a sky of one colour.
    fn from_sky(colour: [f32; 3], [gain, least, power]: [f32; 3]) -> [f32; 3] {
        let mean = colour.map(|c| c.clamp(0.0, 8.0));
        let m = 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2];
        let night = [0.6, 0.75, 1.0];
        let by = (m / 0.05).clamp(0.0, 1.0);
        let level = gain * m.max(0.0).powf(power) + least;
        [0, 1, 2].map(|i| {
            let washed = 1.0 + (mean[i] / m.max(1e-6) - 1.0) * 0.5;
            (night[i] + (washed - night[i]) * by) * level
        })
    }

    /// Every texel of `all` holds `want`, to the format's last step and a
    /// little of the host's `pow`.
    fn assert_light(all: &[u32], want: [f32; 3], what: &str) {
        for (at, &v) in all.iter().enumerate() {
            let has = unpack(v);
            for i in 0..3 {
                assert!(
                    (has[i] - want[i]).abs() <= want[i] * 0.01 + 1e-4,
                    "{what}: texel {at} holds {has:?}, not {want:?}"
                );
            }
        }
    }

    /// The passes compile with the system's compiler (Wine's too) and the
    /// host takes them.
    #[test]
    fn ambient_pass_builds() {
        let Some(host) = crate::host::host() else {
            return;
        };
        super::make(&host.device).expect("the ambient pass");
    }

    /// The pass writes the constant, packed as R9G9B9E5, over the whole
    /// image (also past a multiple of its 8x8 groups).
    #[test]
    fn ambient_pass_writes_the_constant() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let (width, height) = (20u32, 12u32);
        let image = light(host, width, height);
        let list = host.new_list().unwrap();
        assert!(super::record(&list, &image, None, 0.6, None, None));
        // 0.6 * (1, 0.98, 0.94): exponent 15 (scale 2^9), mantissas
        // 307, 301, 289.
        let want = 307 | 301 << 9 | 289 << 18 | 15 << 27;
        assert_eq!(pack([0.6, 0.6 * 0.98, 0.6 * 0.94]), want);
        for (at, v) in texels(host, &list, &image, width, height).into_iter().enumerate() {
            assert_eq!(v, want, "texel {at}");
        }
    }

    /// With a sky the light is `gain * mean^power + least` in the sky's
    /// hue, half washed out, whatever the level: by day, and bluish under a
    /// dark sky; the sun's texels count for 8 at most; a dome that is an
    /// array or typeless on the host is read the same.
    #[test]
    fn ambient_pass_follows_the_sky() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let (width, height) = (20u32, 12u32);
        let follow = [0.36, 0.05, 0.4];
        let float = DXGI_FORMAT_R16G16B16A16_FLOAT;
        for (colour, format, slices) in [
            ([1.0, 0.5, 0.25], float, 1),
            ([0.001, 0.001, 0.002], float, 1),
            ([100.0, 50.0, 2.0], float, 1),
            ([0.25, 0.5, 1.0], float, 2),
            ([0.5, 0.5, 0.25], DXGI_FORMAT_R16G16B16A16_TYPELESS, 1),
        ] {
            let image = light(host, width, height);
            let dome = sky(host, format, slices, colour);
            let list = host.new_list().unwrap();
            assert!(super::record(&list, &image, None, 0.6, Some((&dome, follow)), None));
            let all = texels(host, &list, &image, width, height);
            assert_light(&all, from_sky(colour, follow), &format!("sky {colour:?} in {format:?} x{slices}"));
        }
        // A hand check of the first: 0.36 * 0.58825^0.4 + 0.05 = 0.3412.
        let has = from_sky([1.0, 0.5, 0.25], follow);
        assert!((has[1] - 0.3412 * (1.0 + 0.5 / 0.58825) / 2.0).abs() < 1e-3, "{has:?}");
    }

    /// The dome is held: another image the sky program draws does not take
    /// its place, and with no sky drawn since (and the title's own object
    /// gone) the light stays what the dome gave, list after list.
    #[test]
    fn ambient_pass_holds_the_sky() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let (width, height) = (20u32, 12u32);
        let (colour, follow) = ([0.75, 0.5, 0.25], [0.36, 0.05, 0.4]);
        let held = std::sync::Mutex::new(None);
        let dome = sky(host, DXGI_FORMAT_R16G16B16A16_FLOAT, 1, colour);
        let dome_at = std::sync::Arc::as_ptr(&dome);
        super::hold_sky(&held, dome);
        // The sky's other image, three times brighter.
        super::hold_sky(&held, sky(host, DXGI_FORMAT_R11G11B10_FLOAT, 1, colour.map(|c| c * 3.0)));
        let image = light(host, width, height);
        for frame in 0..3 {
            let dome = held.lock().unwrap().clone().expect("a held dome");
            assert_eq!(std::sync::Arc::as_ptr(&dome), dome_at);
            let list = host.new_list().unwrap();
            assert!(super::record(&list, &image, None, 0.6, Some((&dome, follow)), None));
            let all = texels(host, &list, &image, width, height);
            assert_light(&all, from_sky(colour, follow), &format!("frame {frame}"));
        }
    }

    /// Reflections brighter than a part of the light around are brought
    /// down to it in their own colour, darker ones than the least part are
    /// given it, and those between are left: against the constant, and
    /// against the light a sky gives.
    #[test]
    fn reflections_are_held_to_the_ambient_light() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let (width, height) = (20u32, 12u32);
        let tint = [1.0, 0.98, 0.94];
        let follow = [0.36, 0.05, 0.4];
        let dome = sky(host, DXGI_FORMAT_R16G16B16A16_FLOAT, 1, [1.0, 0.5, 0.25]);
        let luma = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        // What the image holds before, as a level of the tint; the most and
        // the least part; what it holds after, as a part of the light.
        for (before, [most, least], after) in [
            (4.0, [0.5, 0.0], Some(0.5)),
            (0.0, [0.5, 0.25], Some(0.25)),
            (0.0, [0.5, 0.0], None),
            (0.2, [0.9, 0.1], None),
        ] {
            for sky in [None, Some((&dome, follow))] {
                let image = light(host, width, height);
                let list = host.new_list().unwrap();
                // The reflections the title left: a constant of ours.
                assert!(super::record(&list, &image, None, before, None, None));
                assert!(super::record(&list, &image, None, 0.6, sky, Some([most, least])));
                let around = sky.map_or(tint.map(|t| t * 0.6), |_| from_sky([1.0, 0.5, 0.25], follow));
                let was = tint.map(|t| t * before);
                let want = match after {
                    // Brought down: its own colour at the light's brightness.
                    Some(part) if before > 0.0 => was.map(|c| c * luma(around) * part / luma(was)),
                    Some(part) => around.map(|c| c * part),
                    None => was,
                };
                let all = texels(host, &list, &image, width, height);
                let what = format!("{before} held to {most} and {least}, sky {}", sky.is_some());
                assert_light(&all, want, &what);
            }
        }
    }

    /// `VICTORY_AMBIENT_SKY`: on unless `off`, its numbers over the
    /// defaults; a level named by `VICTORY_AMBIENT` alone keeps the constant.
    #[test]
    fn the_sky_is_followed_unless_a_level_is_named() {
        assert_eq!(super::follow_set(None, false), Some([0.36, 0.05, 0.4]));
        assert_eq!(super::follow_set(None, true), None);
        assert_eq!(super::follow_set(Some("off"), false), None);
        assert_eq!(super::follow_set(Some(" off "), true), None);
        assert_eq!(super::follow_set(Some(""), true), Some([0.36, 0.05, 0.4]));
        assert_eq!(super::follow_set(Some("0.5"), false), Some([0.5, 0.05, 0.4]));
        assert_eq!(super::follow_set(Some("0.5, 0.1,0.3"), true), Some([0.5, 0.1, 0.3]));
    }
}
