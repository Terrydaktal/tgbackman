use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=TGBACKMAN_BUILD_REVISION");
    println!("cargo:rerun-if-env-changed=TGBACKMAN_BUILD_DIRTY");
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/index");
    // Git's index is unchanged by ordinary unstaged edits.  Watch the crate
    // sources so a rebuild recomputes the dirty flag for the binary actually
    // being produced, and watch ref files for commits that do not rewrite
    // .git/HEAD itself.
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=../.git/refs");
    println!("cargo:rerun-if-changed=../.git/packed-refs");
    // The GUI is shipped with the Python backup engine; edits there must also
    // invalidate a clean-build marker even though Cargo does not compile them.
    println!("cargo:rerun-if-changed=../src");
    println!("cargo:rerun-if-changed=../docs");
    println!("cargo:rerun-if-changed=../README.md");
    let revision = std::env::var("TGBACKMAN_BUILD_REVISION").ok().or_else(|| {
        Command::new("git")
            .args(["rev-parse", "--short=12", "HEAD"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
    });
    let dirty = std::env::var("TGBACKMAN_BUILD_DIRTY").ok().or_else(|| {
        Command::new("git")
            .args(["status", "--porcelain=v1", "--untracked-files=all"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| {
                if output.stdout.is_empty() {
                    "0".to_string()
                } else {
                    "1".to_string()
                }
            })
    });
    println!(
        "cargo:rustc-env=TGBACKMAN_BUILD_REVISION={}",
        revision
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "cargo:rustc-env=TGBACKMAN_BUILD_DIRTY={}",
        dirty.unwrap_or_else(|| "unknown".to_string())
    );
}
