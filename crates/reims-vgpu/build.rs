//! Product staticlib is pure Rust for protocol utilities.
//!
//! Pre-rewrite C packages lived under `host/utils/` (deleted; git history only).
//!
//! It computes two things. Whether the wire crate's oracle fixtures are on
//! disk, so `tests/wire_fixtures_reach_the_decoders.rs` stands down as `ignored`
//! rather than as `ok` when they are not. See `../reims-vgpu-wire/oracle/fixture_presence.rs`
//! for why the answer is shared with the wire crate instead of decided twice.
//! And which metal2vulkan this build links, for `runtime::m2v_disk`'s key.

include!("../reims-vgpu-wire/oracle/fixture_presence.rs");
include!("src/runtime/m2v_disk/sha256.rs");

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../reims-vgpu-wire/oracle/fixture_presence.rs");
    println!("cargo:rerun-if-changed=src/runtime/m2v_disk/sha256.rs");
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    probe_wire_fixtures(&format!("{manifest}/../reims-vgpu-wire/fixtures"));
    describe_translator(Path::new(&manifest));
}

// MARK: - Translator identity

/// Export what `runtime::m2v_disk` keys a translation on that only the build
/// knows: a SHA-256 over metal2vulkan's manifest and every file under its
/// `src/`, plus the workspace `Cargo.lock` that pins its dependencies; its git
/// commit; and the target triple.
///
/// The digest rather than the commit decides, because an uncommitted edit to
/// the translator changes its output without changing its commit. An empty
/// digest is how a build that could not read the source says so, and the
/// runtime turns the disk cache off for it rather than key on nothing.
fn describe_translator(manifest: &Path) {
    // The same relative path Cargo.toml gives the dependency.
    let translator = manifest.join("../../../metal2vulkan");
    let lock = manifest.join("../../Cargo.lock");
    for watched in [
        translator.join("src"),
        translator.join("Cargo.toml"),
        lock.clone(),
    ] {
        println!("cargo:rerun-if-changed={}", watched.display());
    }
    let digest = match translator_source_digest(&translator, &lock) {
        Ok(digest) => digest,
        Err(error) => {
            println!(
                "cargo:warning=metal2vulkan source unreadable at {} ({error}); \
                 the m2v disk cache is off in this build",
                translator.display()
            );
            String::new()
        }
    };
    let commit = std::process::Command::new("git")
        .arg("-C")
        .arg(&translator)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    let target = std::env::var("TARGET").unwrap_or_default();
    println!("cargo:rustc-env=REIMS_M2V_SOURCE_DIGEST={digest}");
    println!("cargo:rustc-env=REIMS_M2V_COMMIT={commit}");
    println!("cargo:rustc-env=REIMS_M2V_TARGET={target}");
}

fn translator_source_digest(translator: &Path, lock: &Path) -> std::io::Result<String> {
    let mut files = vec![translator.join("Cargo.toml")];
    collect_files(&translator.join("src"), &mut files)?;
    files.sort();
    let mut hasher = Sha256::default();
    for path in &files {
        let name = path.strip_prefix(translator).unwrap_or(path);
        digest_file(&mut hasher, name, path)?;
    }
    digest_file(&mut hasher, Path::new("Cargo.lock"), lock)?;
    Ok(sha256_hex(&hasher.finish()))
}

/// The name as well as the bytes, so a rename is a different translator, and
/// the length first, so no two files' bytes can run together.
fn digest_file(hasher: &mut Sha256, name: &Path, path: &Path) -> std::io::Result<()> {
    let bytes = std::fs::read(path)?;
    hasher.update(name.to_string_lossy().as_bytes());
    hasher.update(&[0]);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(&bytes);
    Ok(())
}

fn collect_files(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}
