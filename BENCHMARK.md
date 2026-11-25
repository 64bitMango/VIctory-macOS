# Radeon runtime testing, 25 November 2026

The Windows diagnostics release at `e061c69032b84ec6e2ec17340469cd689500dafd` built with the pinned toolchain. It includes GPU readback changes from `6a4e2b1` and descriptor allocation changes from `e061c69`.

| Check | Result |
| --- | --- |
| GPU library tests on the D3D12 WARP software adapter | 163 passed, 0 failed, 3 ignored |
| Pixel shaders previously refused in the mission capture | All 11 translated, passed DXIL validation and produced test pipelines |
| Upstream repository asset-policy scan | 182 tracked files passed |

In a preliminary Chapter 1 City comparison, the old and new readback paths used the same checkpoint, shader cache and renderer settings. Across 93 matched indoor presents, frame rate was 3.52 and 4.05 FPS respectively. The 95th-percentile frame time was 432 and 352 ms. Two comparisons of the outdoor camera gave different results, ranging from an 18% loss to an 8% gain. The indoor figure currently has one run of the new build.

The City interior can render with an overexposed doorway and a silhouetted character. A mesh and pixel pipeline used for the windshield wiper mask was refused during the City sequence. The ray-tracing structure builder is disabled by default. Whole-game shader coverage and correct indoor lighting remain open.
