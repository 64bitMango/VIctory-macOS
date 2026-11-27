# Review rules for VIctory for macOS

- **Nothing from the application is ever committed.** That means no executables, DLLs, shaders, textures, audio, keys, dumps, decrypted files or screenshots of application content. It also covers hex or base64 blobs inside text files. `cargo xtask policy` enforces this with a file-name allowlist. Flag any PR that adds such data or widens the allowlist or its `EXCEPTIONS`.
- Nothing from Apple's Game Porting Toolkit (D3DMetal and the like) may be committed.
- Every `unsafe` block needs a `// SAFETY:` comment that says why it is sound.
- Behaviour changes need a test. GPU and shader behaviour is best tested under Wine with D3DMetal; Windows-only tests skip themselves without a device.
- A deliberate simplification is marked with a `shortcut:` comment. Removing one should implement what it describes, not just delete the comment.
- Lock-free tables that reuse slots must not let an old record resolve to a new entry: check generation widths and wraparound.
- A feature that depends on host support (for example D3D12 options) needs a fallback when the host reports it unsupported.
- Commit subjects are short and say what changed, such as `fix(gpu): keep the clamp's budget per list`.
