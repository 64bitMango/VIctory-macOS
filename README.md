# DexNom Runtime

DexNom Runtime is a community fork of Marvisak's VIctory for Windows x64. It runs an Xbox Series executable from your own game installation and translates its GPU work for Direct3D 12. This initial public version uses the Radeon runtime tested at `e061c69032b84ec6e2ec17340469cd689500dafd`, with DexNom branding. Game files, executables, saves and generated shader caches stay on your own computer.

## Build and run

Install Git, Rust through rustup, and Visual Studio Build Tools with **Desktop development with C++**, x64 MSVC tools and a Windows 10/11 SDK. Use a Direct3D 12 graphics driver for gameplay. Open PowerShell in the extracted source directory.

1. Obtain the BCPack format tables and check their SHA-256. The script fetches the pinned upstream copy by default. To use a local VIctory checkout instead, run `.\Initialize-FormatTables.ps1 -FromCheckout (Read-Host 'Path to existing VIctory checkout')`.

   ```powershell
   .\Initialize-FormatTables.ps1
   ```

2. Build the diagnostics-enabled release runtime and its launchers. Cargo reads the pinned Rust 1.98.1 toolchain and dependency lockfile.

   ```powershell
   cargo build --locked --release -p victory --features diagnostics --bins
   ```

3. Copy your three locally built programs to a folder outside the checkout so the launcher can create its save and cache directories. Supply the path to your own Series X game executable.

   ```powershell
   $runtimeDir = Join-Path $env:LOCALAPPDATA 'DexNom Runtime'
   New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
   Copy-Item .\target\release\victory.exe, .\target\release\victory-play.exe, .\target\release\victory-warm.exe -Destination $runtimeDir
   $gameExe = (Get-Item -LiteralPath (Read-Host 'Full path to your game executable')).FullName
   & (Join-Path $runtimeDir 'victory-play.exe') $gameExe
   ```

The launcher writes saves to `state` and diagnostic logs to `logs` beside the launcher. For repeat launches, put the full game executable path on one line in `game-path.txt` beside `victory-play.exe`. To precompile recognized shaders from that installation, run `victory-warm.exe` with the same game path. Graphics pipeline creation also uses draw state supplied during play. The command-line runtime supports separate data and cache directories. [Configuration example](victory.example.toml)

## Changes in this snapshot

- GPU readback identifies changed pages and combines adjacent copy ranges. This targets CPU/GPU synchronization and transfer overhead.
- Texture handling searches for contiguous descriptor slots and admits already-resident textures under memory pressure. Earlier changes address publication order, placed-texture ownership and supported render-XOR layouts.
- Shader translation preserves the guest wave width for lane permutations and handles heap-indexed texture-size queries. Save handling flushes copied blobs before publication and can recover a complete checkpoint.

The diagnostics release built on Windows. A WARP GPU test run passed 163 tests with three ignored. Eleven previously refused pixel shaders translated, passed DXIL validation and produced test pipelines. See the dated report for gameplay results and remaining issues. [Test and benchmark report](BENCHMARK.md)

## Credits and licensing

- Marvisak and the VIctory contributors created the original runtime. [Marvisak/VIctory](https://github.com/Marvisak/VIctory)
- ZSlayerHQ and the contributors to the victory-world line provided the rendering baseline used here, including the Radeon work from triippywhale and asdaasdsaddf. [victory-world-runtime](https://github.com/ZSlayerHQ/victory-world-runtime) · [Integrated revision](https://github.com/ZSlayerHQ/victory-world-runtime/commit/99f8834aa16b98d7f3943b8de262a8d4857bda7b)
- The XCL contributors published an independent Xbox Series compatibility layer. Their work informed our investigation of shader translation, CPU/GPU memory synchronization and performance. [xcl-dev/xcl](https://github.com/xcl-dev/xcl)

The upstream README preserves the original credits and historical test results. [Upstream README](README.upstream.md)

VIctory source is licensed under GPL-3.0-or-later. The upstream project identifies the required BCPack format tables as separately supplied data outside that license. This source archive omits that file. The setup script verifies a local or upstream copy before use. [License](LICENSE) · [Upstream asset policy and table notice](docs/asset-policy.md)
