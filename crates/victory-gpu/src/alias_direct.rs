//! Direct format-cast copies of aliased textures (`VICTORY_ALIAS_DIRECT=1`;
//! off by default). `texture::alias_copy` copies a texture the host writes
//! into one the title placed at the same memory in another format through
//! a buffer, part by part. Direct3D copies straight between one pair of
//! formats of other families (the format reinterpretation of Direct3D
//! 10.1): 32-bit integers and the shared-exponent format, the texels' bits
//! as they are. The title writes packed light as integers from compute and
//! reads it as the shared-exponent format. After SlaktarnVI / avatarmanz
//! e823263 (`ALIAS_DIRECT`).

use std::sync::OnceLock;

use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32_UINT,
};

/// Whether the switch is on (`VICTORY_ALIAS_DIRECT` `1`, `on`, `true` or
/// `yes`). Read once.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse(std::env::var("VICTORY_ALIAS_DIRECT").ok().as_deref()))
}

/// `VICTORY_ALIAS_DIRECT`'s value. Pure, for testing.
fn parse(value: Option<&str>) -> bool {
    value.is_some_and(|v| {
        matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes")
    })
}

/// Whether textures of `written` and `read` copy into each other as they
/// are, the texels' bits unchanged (Direct3D's one cast across families).
pub(crate) fn casts(written: DXGI_FORMAT, read: DXGI_FORMAT) -> bool {
    let integer = |f: DXGI_FORMAT| f == DXGI_FORMAT_R32_UINT || f == DXGI_FORMAT_R32_SINT;
    let shared = |f: DXGI_FORMAT| f == DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    (integer(written) && shared(read)) || (shared(written) && integer(read))
}

/// Whether `alias_copy` copies `written` into `read` straight: the switch
/// is on and the two cast ([`casts`]).
pub(crate) fn direct(written: DXGI_FORMAT, read: DXGI_FORMAT) -> bool {
    casts(written, read) && on()
}

#[cfg(test)]
mod tests {
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::core::Interface;

    use super::*;

    #[test]
    fn only_integers_and_the_shared_exponent_cast() {
        assert!(casts(DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R9G9B9E5_SHAREDEXP));
        assert!(casts(DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R32_SINT));
        assert!(!casts(DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R9G9B9E5_SHAREDEXP));
        assert!(!casts(DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_FLOAT));
        assert!(!casts(DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R9G9B9E5_SHAREDEXP));
        assert!(!parse(None));
        assert!(!parse(Some("0")));
        assert!(parse(Some("1")));
    }

    const W: u32 = 64;
    const H: u32 = 8;
    /// Bytes a row in the copy buffers (aligned as copies need).
    const PITCH: u32 = 256;

    fn texture(device: &ID3D12Device, format: DXGI_FORMAT) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(W),
            Height: H,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let mut t: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut t,
                )
                .expect("texture");
        }
        t.expect("texture")
    }

    fn buffer(device: &ID3D12Device, kind: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: u64::from(PITCH * H),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut b: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut b)
                .expect("buffer");
        }
        b.expect("buffer")
    }

    fn placed(r: &ID3D12Resource, format: DXGI_FORMAT) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format,
                        Width: W,
                        Height: H,
                        Depth: 1,
                        RowPitch: PITCH,
                    },
                },
            },
        }
    }

    fn sub(r: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        }
    }

    fn barrier(r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    // SAFETY: a borrowed pointer the barrier never releases.
                    pResource: unsafe { std::mem::transmute_copy(r) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        }
    }

    /// Texels of `from` uploaded, copied straight into a texture of `to`,
    /// read back: the bytes read, or `None` without a device.
    fn cast_copy(texels: &[u32], from: DXGI_FORMAT, to: DXGI_FORMAT) -> Option<Vec<u32>> {
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: plain creation on the default adapter.
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut device) }.is_err() {
            eprintln!("no D3D12 device; not run");
            return None;
        }
        let device = device?;
        let (a, b) = (texture(&device, from), texture(&device, to));
        let up = buffer(&device, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let down = buffer(&device, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        // SAFETY (the rest): live objects of our own; mappings of the
        // buffers' `PITCH * H` bytes; the list waited for before reading.
        unsafe {
            let mut data = std::ptr::null_mut();
            up.Map(0, None, Some(&mut data)).expect("map");
            for y in 0..H as usize {
                let row = data.cast::<u8>().add(y * PITCH as usize).cast::<u32>();
                for x in 0..W as usize {
                    row.add(x).write(texels[y * W as usize + x]);
                }
            }
            up.Unmap(0, None);
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("queue");
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("allocator");
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .expect("list");
            list.CopyTextureRegion(&sub(&a), 0, 0, 0, &placed(&up, from), None);
            list.ResourceBarrier(&[barrier(&a, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            // The copy under test: one format into the other, straight.
            list.CopyTextureRegion(&sub(&b), 0, 0, 0, &sub(&a), None);
            list.ResourceBarrier(&[barrier(&b, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyTextureRegion(&placed(&down, to), 0, 0, 0, &sub(&b), None);
            list.Close().expect("close");
            queue.ExecuteCommandLists(&[Some(list.cast().expect("list"))]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            queue.Signal(&fence, 1).expect("signal");
            // No event: waits until the fence reaches the value.
            fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).expect("wait");
            if let Err(e) = device.GetDeviceRemovedReason() {
                panic!("device removed: {e}");
            }
            let mut data = std::ptr::null_mut();
            down.Map(0, None, Some(&mut data)).expect("map");
            let mut out = Vec::with_capacity(texels.len());
            for y in 0..H as usize {
                let row = data.cast::<u8>().add(y * PITCH as usize).cast::<u32>();
                out.extend((0..W as usize).map(|x| row.add(x).read()));
            }
            down.Unmap(0, None);
            Some(out)
        }
    }

    /// On the host device (under Wine: D3DMetal), a 32-bit integer texture
    /// copied straight into a shared-exponent one and back keeps every bit
    /// of every texel: the direct copy may be used there.
    #[test]
    fn integer_and_shared_exponent_textures_copy_into_each_other_bit_for_bit() {
        // Every pattern of bits kind: the exponent's five bits through all
        // values, mantissas full and empty, and mixed words.
        let texels: Vec<u32> = (0..W * H)
            .map(|i| match i % 4 {
                0 => i.wrapping_mul(0x9E37_79B9),
                1 => (i % 32) << 27 | 0x07FF_FFFF,
                2 => (i % 32) << 27,
                _ => !i.wrapping_mul(0x0101_0101),
            })
            .collect();
        let Some(shared) = cast_copy(&texels, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R9G9B9E5_SHAREDEXP) else {
            return;
        };
        assert_eq!(shared, texels, "R32_UINT into R9G9B9E5_SHAREDEXP");
        let back = cast_copy(&texels, DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R32_UINT).expect("device");
        assert_eq!(back, texels, "R9G9B9E5_SHAREDEXP into R32_UINT");
    }
}
