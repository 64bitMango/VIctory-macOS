//! The frame rate and frame time written in the picture's top left corner,
//! as rectangles cleared on the back buffer: no shader, texture or font.
//! `VICTORY_FPS=1` turns it on (the Belobug fork, where it comes from,
//! shows it unless `VICTORY_FPS=0`). It is drawn on the swap chain's buffer
//! after the picture is copied there, so captures never hold it.

use std::sync::OnceLock;

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

/// Whether the readout is drawn.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_FPS").is_ok_and(|v| v.trim() == "1"))
}

/// Writes `fps` frames a second and `ms` milliseconds a frame in the top
/// left corner of a render target of `width` by `height` pixels.
///
/// # Safety
/// `list` is open and `view` a view of a texture of that size, which is in
/// the render target state on it.
pub(crate) unsafe fn draw(
    list: &ID3D12GraphicsCommandList,
    view: D3D12_CPU_DESCRIPTOR_HANDLE,
    (width, height): (u32, u32),
    (fps, ms): (f64, f64),
) {
    let text = format!("{fps:.1} FPS  {ms:.0} MS");
    // Dots that keep their share of the picture at any size.
    let dot = (height as i32 / 216).max(2);
    let at = (dot * 3, dot * 3);
    let (w, h) = size(&text);
    let behind = RECT {
        left: at.0,
        top: at.1,
        right: at.0 + w * dot,
        bottom: at.1 + h * dot,
    };
    if behind.right > width as i32 || behind.bottom > height as i32 {
        return;
    }
    let marks = rects(&text, at, dot);
    // SAFETY: the caller's list and view; rectangles inside the picture.
    unsafe {
        list.ClearRenderTargetView(view, &BEHIND, Some(&[behind]));
        list.ClearRenderTargetView(view, &MARKS, Some(&marks));
    }
}

/// The colours of what is behind the text and of its dots.
const BEHIND: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const MARKS: [f32; 4] = [1.0, 0.9, 0.1, 1.0];

/// A glyph's seven rows, top first, of five dots each (bit 4 is the left).
fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e],
        '1' => [0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e],
        '2' => [0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f],
        '3' => [0x1f, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0e],
        '4' => [0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02],
        '5' => [0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e],
        '6' => [0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e],
        '7' => [0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e],
        '9' => [0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c],
        'F' => [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10],
        'P' => [0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10],
        'S' => [0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e],
        'M' => [0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11],
        _ => [0; 7],
    }
}

/// The text's width and height in dots, with a dot of margin all round.
pub(crate) fn size(text: &str) -> (i32, i32) {
    (text.chars().count() as i32 * 6 + 1, 9)
}

/// The rectangles of `text` with its top left margin at `at` and dots of
/// `dot` pixels: one for each run of dots in a row.
pub(crate) fn rects(text: &str, at: (i32, i32), dot: i32) -> Vec<RECT> {
    let mut all = Vec::new();
    for (i, c) in text.chars().enumerate() {
        let left = at.0 + (1 + i as i32 * 6) * dot;
        for (row, bits) in glyph(c).into_iter().enumerate() {
            let top = at.1 + (1 + row as i32) * dot;
            let mut x = 0;
            while x < 5 {
                if bits & (0x10 >> x) == 0 {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < 5 && bits & (0x10 >> x) != 0 {
                    x += 1;
                }
                all.push(RECT {
                    left: left + start * dot,
                    top,
                    right: left + x * dot,
                    bottom: top + dot,
                });
            }
        }
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_of_dots_is_one_rectangle() {
        // The top row of "F" is five dots.
        let all = rects("F", (10, 20), 4);
        assert_eq!(
            all[0],
            RECT {
                left: 14,
                top: 24,
                right: 34,
                bottom: 28
            }
        );
        assert!(rects(" ", (0, 0), 4).is_empty());
        assert_eq!(size("12 FPS"), (37, 9));
    }

    /// The readout is drawn where expected on a render target of the host
    /// (D3DMetal under Wine): dots in the marks' colour, black between
    /// them, and the picture untouched beside it.
    #[test]
    fn the_readout_is_drawn_in_the_corner() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        let Some(host) = crate::host::host() else {
            return;
        };
        let (width, height) = (640u32, 432u32);
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut picture: Option<ID3D12Resource> = None;
        let target = D3D12_RESOURCE_STATE_RENDER_TARGET;
        // SAFETY: plain creation.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, target, None, &mut picture)
                .unwrap();
        }
        let picture = picture.unwrap();
        let pitch = (4 * width).next_multiple_of(256);
        let (out, data) = host.shared_buffer(u64::from(pitch * height)).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: an open list, live resources in the states named; the
        // mapped buffer is read after the list ran.
        let at = unsafe {
            let view = host.rtv(&picture, None);
            list.list.ClearRenderTargetView(view, &[0.0, 0.0, 1.0, 1.0], None);
            draw(&list.list, view, (width, height), (7.5, 133.0));
            list.list.ResourceBarrier(&[
                crate::host::transition(&picture, target, D3D12_RESOURCE_STATE_COPY_SOURCE),
                crate::host::transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
            ]);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&out),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            Width: width,
                            Height: height,
                            Depth: 1,
                            RowPitch: pitch,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&picture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            list.close();
            host.execute(&[&list]);
            move |x: u32, y: u32| data.add((y * pitch + 4 * x) as usize).cast::<[u8; 4]>().read_volatile()
        };
        const BLUE: [u8; 4] = [0, 0, 255, 255];
        const BLACK: [u8; 4] = [0, 0, 0, 255];
        const MARK: [u8; 4] = [255, 230, 26, 255];
        // Dots of 2 pixels, the text's margin at (6, 6): "7.5 FPS  133 MS".
        let text = "7.5 FPS  133 MS";
        let (w, h) = size(text);
        let mut marked = 0;
        for y in 0..height {
            for x in 0..width {
                let has = at(x, y);
                let inside = (6..6 + 2 * w as u32).contains(&x) && (6..6 + 2 * h as u32).contains(&y);
                if !inside {
                    assert_eq!(has, BLUE, "the picture at {x},{y}");
                    continue;
                }
                // Which dot of which glyph: its bit says the colour.
                let (dx, dy) = ((x - 6) / 2, (y - 6) / 2);
                let (column, row) = ((dx + 5) % 6, dy.wrapping_sub(1));
                let bits = text.chars().nth((dx.div_ceil(6)).wrapping_sub(1) as usize).map(glyph);
                let on = dx >= 1 && column < 5 && row < 7 && bits.is_some_and(|b| b[row as usize] & (0x10 >> column) != 0);
                let want = if on { MARK } else { BLACK };
                for i in 0..4 {
                    assert!(has[i].abs_diff(want[i]) <= 1, "{x},{y} holds {has:?}, not {want:?}");
                }
                marked += u32::from(on);
            }
        }
        // The top row of the 7 is five dots, at (7, 7) in dots.
        assert_eq!(at(14, 14), at(23, 15));
        assert!(marked > 4 * 100, "{marked} pixels marked");
    }
}
