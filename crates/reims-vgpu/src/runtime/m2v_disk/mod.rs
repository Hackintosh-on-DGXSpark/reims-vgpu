//! metal2vulkan translations kept on disk, so a boot loads what an earlier boot
//! translated instead of translating it again.
//!
//! # Why
//!
//! [`super::m2v_cache`] holds translations for the life of one process, and
//! every process started it empty. On a macOS 27.2 guest the first frame's
//! fragment shader (AIR ~94 KB, SPIR-V ~1.47 MB) takes ~8.3 s to translate on
//! every boot, and while it does, the translation-order hold keeps its sibling
//! channels — the display channel among them — waiting for 9–18 s. The guest's
//! display driver gives up on its first swap after 5 s. The Vulkan pipeline
//! cache beside it in `$TMPDIR` already persisted; the translation in front of
//! it did not, because its only directory was named after the process id.
//!
//! This is only the persistence. The hold, the queue and the order in which
//! translations resolve stay `m2v_cache`'s: an entry found here resolves the
//! same task the translator would have, sooner.
//!
//! # What is keyed
//!
//! An entry is the translator's own output — the SPIR-V and the reflection that
//! `metal2vulkan::translate_reflected_with_options` returned — not the
//! [`super::m2v_cache::CachedShader`] Reims derives from it. The caller runs the
//! same post-processing on a loaded entry as on a fresh one, so a change on the
//! Reims side of that line needs no key change. What reaches the translator is
//! keyed:
//!
//! - the AIR, stored in the entry and compared whole on load. The file name is
//!   a SHA-256 of the identity and only finds the file; as in `m2v_cache`, the
//!   retained bytes decide the hit.
//! - the stage, and the whole `TransformOptions` the caller passes (which holds
//!   a kernel's LocalSize) by its `Debug` text, so a field added to the options
//!   is keyed without anyone remembering to.
//! - the translator, [`translator_fingerprint`]: a digest of metal2vulkan's
//!   source and of `Cargo.lock` taken by `build.rs`, its commit, the target,
//!   whether debug assertions are on, every `METAL2VULKAN_*` variable, and the
//!   path, size, mtime and `--version` of the `llvm-dis` and `spirv-val` it runs.
//!
//! Nothing about the Vulkan device reaches the translator on this path. The
//! options are the default, plus LocalSize for a kernel; metal2vulkan derives
//! the rest (denormals, 32-lane SIMD) from the AIR; and the paravirt stream
//! carries no function-constant values, so the translator folds those from the
//! AIR too. No device capability or specialization state is keyed because none
//! is an input.
//!
//! # On disk
//!
//! `$TMPDIR/reims-vgpu-m2v-cache/<translator>/` holds one `<identity>.m2v` per
//! translation, and a `translator.txt` spelling out what `<translator>` stands
//! for. A new translator starts an empty directory; the old one stays until an
//! operator deletes it.
//!
//! An entry is written to a temporary file in its directory, synced, and renamed
//! over its name. A reader sees no entry or a whole one, and two processes that
//! publish one key leave whichever renamed last — the same translation either
//! way. Every entry carries its lengths and ends in a SHA-256 of everything
//! before it. One that fails either check, or whose identity is not the one
//! asked for, is refused with a line naming the check, translated again, and
//! republished.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, UNIX_EPOCH};

use metal2vulkan::passes::{Stage, TransformOptions};
use metal2vulkan::reflect::ShaderReflection;

use crate::env::Switch;

mod sha256;
#[cfg(test)]
pub(crate) mod tests;

use sha256::{sha256, sha256_hex, Sha256};

/// Changes whenever the entry layout in [`DiskCache::encode`] does.
const FORMAT: u32 = 1;
const MAGIC: &[u8; 8] = b"RM2VDISK";
const ROOT: &str = "reims-vgpu-m2v-cache";
const ENTRY_SUFFIX: &str = ".m2v";
const CHECKSUM_LEN: usize = 32;
const SPIRV_MAGIC: u32 = 0x0723_0203;

/// The programs metal2vulkan runs during a translation (`metal2vulkan::tools`).
const TOOLS: [&str; 2] = ["llvm-dis", "spirv-val"];

// MARK: - Identity

/// One translation's inputs, short of the translator itself.
pub struct Key<'a> {
    stage: Stage,
    options: String,
    air: &'a [u8],
}

impl<'a> Key<'a> {
    pub fn new(stage: Stage, options: &TransformOptions, air: &'a [u8]) -> Self {
        Self {
            stage,
            options: format!("{options:?}"),
            air,
        }
    }
}

fn stage_tag(stage: Stage) -> u8 {
    match stage {
        Stage::Vertex => 1,
        Stage::Fragment => 2,
        Stage::Kernel => 3,
    }
}

/// Everything about the translator that a translation depends on and the AIR
/// does not say, one fact a line. Its SHA-256 names the cache directory and is
/// checked in every entry.
pub fn translator_fingerprint() -> String {
    let mut text = format!(
        "format {FORMAT}\nm2v_source {}\nm2v_commit {}\ntarget {}\ndebug_assertions {}\n",
        env!("REIMS_M2V_SOURCE_DIGEST"),
        env!("REIMS_M2V_COMMIT"),
        env!("REIMS_M2V_TARGET"),
        cfg!(debug_assertions),
    );
    let mut variables: Vec<(String, String)> = std::env::vars_os()
        .map(|(name, value)| {
            (
                name.to_string_lossy().into_owned(),
                value.to_string_lossy().into_owned(),
            )
        })
        .filter(|(name, _)| name.starts_with("METAL2VULKAN_"))
        .collect();
    variables.sort();
    for (name, value) in variables {
        text.push_str(&format!("env {name}={value}\n"));
    }
    for tool in TOOLS {
        text.push_str(&format!("tool {tool} {}\n", tool_identity(tool)));
    }
    text
}

fn tool_identity(tool: &str) -> String {
    let file = match resolve_tool(tool).map(std::fs::canonicalize) {
        Some(Ok(path)) => match std::fs::metadata(&path) {
            Ok(meta) => {
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                    .map_or(0, |since| since.as_nanos());
                format!("{} size={} mtime={mtime}", path.display(), meta.len())
            }
            Err(error) => format!("{} unreadable({error})", path.display()),
        },
        Some(Err(error)) => format!("unresolved({error})"),
        None => "unresolved".to_owned(),
    };
    // Through metal2vulkan's own runner, so the version is of whichever binary
    // it resolves even if `resolve_tool` ever drifts from its search.
    let version = match metal2vulkan::tools::run(tool, &["--version"]) {
        Ok((out, err)) => one_line(&[out, err].concat()),
        Err(error) => format!("error({})", one_line(error.as_bytes())),
    };
    format!("{file} version={version}")
}

/// `metal2vulkan::tools`' search, which is private: the `METAL2VULKAN_<TOOL>`
/// override, four Homebrew and local directories, then `PATH`.
fn resolve_tool(tool: &str) -> Option<PathBuf> {
    let variable = format!(
        "METAL2VULKAN_{}",
        tool.replace('-', "_").to_ascii_uppercase()
    );
    if let Some(path) = std::env::var_os(variable) {
        return Some(PathBuf::from(path));
    }
    let fixed = [
        "/opt/homebrew/opt/llvm/bin",
        "/usr/local/opt/llvm/bin",
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ]
    .map(PathBuf::from);
    let path = std::env::var_os("PATH").unwrap_or_default();
    fixed
        .into_iter()
        .chain(std::env::split_paths(&path))
        .map(|dir| dir.join(tool))
        .find(|candidate| candidate.is_file())
}

fn one_line(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// MARK: - Cache

/// What the translator returned for one [`Key`].
pub struct Translation {
    pub spirv: Vec<u8>,
    pub reflection: ShaderReflection,
}

pub enum Lookup {
    Hit(Box<Translation>),
    Miss,
    /// An entry was there and was not used; the reason names the check it
    /// failed.
    Refused(&'static str),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub hits: u64,
    pub misses: u64,
    pub refused: u64,
    pub stored: u64,
    pub store_failed: u64,
}

impl std::fmt::Display for Counts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "hits={} misses={} refused={} stored={} store_failed={}",
            self.hits, self.misses, self.refused, self.stored, self.store_failed
        )
    }
}

#[derive(Default)]
struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    refused: AtomicU64,
    stored: AtomicU64,
    store_failed: AtomicU64,
}

/// One translator's directory of entries.
pub struct DiskCache {
    dir: PathBuf,
    translator: [u8; 32],
    entries_at_open: usize,
    counters: Counters,
    summarized: Mutex<Counts>,
}

impl DiskCache {
    /// The directory for `translator` (a [`translator_fingerprint`]) under
    /// `root`, created if it is not there.
    pub fn open(root: &Path, translator: &str) -> std::io::Result<Self> {
        let digest = sha256(translator.as_bytes());
        let dir = root.join(sha256_hex(&digest[..8]));
        std::fs::create_dir_all(&dir)?;
        if !dir.join("translator.txt").is_file() {
            publish(&dir, "translator.txt", translator.as_bytes())?;
        }
        let entries_at_open = std::fs::read_dir(&dir)?
            .filter_map(Result::ok)
            .filter(|entry| is_entry_name(&entry.file_name().to_string_lossy()))
            .count();
        Ok(Self {
            dir,
            translator: digest,
            entries_at_open,
            counters: Counters::default(),
            summarized: Mutex::new(Counts::default()),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Entries in the directory when this process opened it: what an earlier
    /// boot left to load.
    pub fn entries_at_open(&self) -> usize {
        self.entries_at_open
    }

    pub fn counts(&self) -> Counts {
        let c = &self.counters;
        Counts {
            hits: c.hits.load(Ordering::Relaxed),
            misses: c.misses.load(Ordering::Relaxed),
            refused: c.refused.load(Ordering::Relaxed),
            stored: c.stored.load(Ordering::Relaxed),
            store_failed: c.store_failed.load(Ordering::Relaxed),
        }
    }

    fn entry_name(&self, key: &Key<'_>) -> String {
        let mut hasher = Sha256::default();
        hasher.update(&self.translator);
        hasher.update(&[stage_tag(key.stage)]);
        hasher.update(&(key.options.len() as u64).to_le_bytes());
        hasher.update(key.options.as_bytes());
        hasher.update(key.air);
        format!("{}{ENTRY_SUFFIX}", sha256_hex(&hasher.finish()))
    }

    pub fn load(&self, key: &Key<'_>) -> Lookup {
        let lookup = match std::fs::read(self.dir.join(self.entry_name(key))) {
            Ok(bytes) => match self.decode(&bytes, key) {
                Ok(translation) => Lookup::Hit(Box::new(translation)),
                Err(reason) => Lookup::Refused(reason),
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Lookup::Miss,
            Err(_) => Lookup::Refused("unreadable"),
        };
        let counter = match lookup {
            Lookup::Hit(_) => &self.counters.hits,
            Lookup::Miss => &self.counters.misses,
            Lookup::Refused(_) => &self.counters.refused,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        lookup
    }

    pub fn store(
        &self,
        key: &Key<'_>,
        spirv: &[u8],
        reflection: &ShaderReflection,
    ) -> Result<(), String> {
        let stored = self.encode(key, spirv, reflection).and_then(|bytes| {
            publish(&self.dir, &self.entry_name(key), &bytes).map_err(|error| error.to_string())
        });
        let counter = match stored {
            Ok(()) => &self.counters.stored,
            Err(_) => &self.counters.store_failed,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        stored
    }

    /// The translator's output for `key`: loaded when this directory holds it,
    /// otherwise `translate`d and published. Every outcome is one line.
    pub fn load_or_translate<E>(
        &self,
        key: &Key<'_>,
        pipeline_ref: u32,
        translate: impl FnOnce() -> Result<Translation, E>,
    ) -> Result<Translation, E> {
        let stage = key.stage;
        let air = key.air.len();
        let started = Instant::now();
        let refused = match self.load(key) {
            Lookup::Hit(translation) => {
                crate::observe::off(format!(
                    "linux_m2v_disk hit pipe={pipeline_ref} stage={stage:?} air={air} spv={} load_ms={} {}",
                    translation.spirv.len(),
                    started.elapsed().as_millis(),
                    self.counts()
                ));
                return Ok(*translation);
            }
            Lookup::Miss => None,
            Lookup::Refused(reason) => {
                crate::observe::fail(format!(
                    "linux_m2v_disk refused pipe={pipeline_ref} stage={stage:?} air={air} reason={reason} entry={}",
                    self.dir.join(self.entry_name(key)).display()
                ));
                Some(reason)
            }
        };
        let started = Instant::now();
        let translated = translate();
        let translate_ms = started.elapsed().as_millis();
        let (spv, published) = match &translated {
            Ok(translation) => {
                let published = match self.store(key, &translation.spirv, &translation.reflection) {
                    Ok(()) => 1,
                    Err(error) => {
                        crate::observe::fail(format!(
                            "linux_m2v_disk store_failed pipe={pipeline_ref} stage={stage:?} air={air} error={error}"
                        ));
                        0
                    }
                };
                (translation.spirv.len(), published)
            }
            Err(_) => (0, 0),
        };
        let after = refused.map_or(String::new(), |reason| format!(" after_refused={reason}"));
        crate::observe::off(format!(
            "linux_m2v_disk miss pipe={pipeline_ref} stage={stage:?} air={air} spv={spv} translate_ms={translate_ms} translated={} published={published}{after} {}",
            u8::from(translated.is_ok()),
            self.counts()
        ));
        translated
    }

    // MARK: - Entry layout

    /// `MAGIC`, `FORMAT` (u32), the translator digest (32 bytes), the stage
    /// tag (u8), then options, AIR, SPIR-V and reflection JSON each as a u64
    /// length and its bytes, then the static-sampler LOD bits (a u64 count of
    /// three-u32 groups), then a SHA-256 of all of the above. Little-endian.
    fn encode(
        &self,
        key: &Key<'_>,
        spirv: &[u8],
        reflection: &ShaderReflection,
    ) -> Result<Vec<u8>, String> {
        let (json, lods) = encode_reflection(reflection)?;
        // A reflection that would not load as itself is not published: every
        // later boot would refuse it, translate, and say nothing new.
        let reloaded = decode_reflection(&json, &lods)?;
        if format!("{reloaded:?}") != format!("{reflection:?}") {
            return Err("the reflection does not survive a round trip".to_owned());
        }
        let mut out = Vec::with_capacity(
            128 + key.options.len() + key.air.len() + spirv.len() + json.len() + lods.len() * 12,
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT.to_le_bytes());
        out.extend_from_slice(&self.translator);
        out.push(stage_tag(key.stage));
        for section in [key.options.as_bytes(), key.air, spirv, json.as_slice()] {
            out.extend_from_slice(&(section.len() as u64).to_le_bytes());
            out.extend_from_slice(section);
        }
        out.extend_from_slice(&(lods.len() as u64).to_le_bytes());
        for bits in lods.iter().flatten() {
            out.extend_from_slice(&bits.to_le_bytes());
        }
        let checksum = sha256(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    fn decode(&self, bytes: &[u8], key: &Key<'_>) -> Result<Translation, &'static str> {
        let body_len = bytes.len().checked_sub(CHECKSUM_LEN).ok_or("truncated")?;
        let (body, checksum) = bytes.split_at(body_len);
        if sha256(body)[..] != *checksum {
            return Err("checksum");
        }
        let mut reader = Reader(body);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err("magic");
        }
        if reader.u32()? != FORMAT {
            return Err("format");
        }
        if reader.take(self.translator.len())? != self.translator {
            return Err("translator");
        }
        if reader.take(1)? != [stage_tag(key.stage)] {
            return Err("stage");
        }
        if reader.section()? != key.options.as_bytes() {
            return Err("options");
        }
        if reader.section()? != key.air {
            return Err("air");
        }
        let spirv = reader.section()?;
        let json = reader.section()?;
        let groups = reader.u64()?;
        let mut lods = Vec::new();
        for _ in 0..groups {
            lods.push([reader.u32()?, reader.u32()?, reader.u32()?]);
        }
        if !reader.0.is_empty() {
            return Err("trailing");
        }
        if spirv.len() < 20
            || !spirv.len().is_multiple_of(4)
            || spirv[..4] != SPIRV_MAGIC.to_le_bytes()
        {
            return Err("spirv");
        }
        let reflection = decode_reflection(json, &lods).map_err(|_| "reflection")?;
        Ok(Translation {
            spirv: spirv.to_vec(),
            reflection,
        })
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], &'static str> {
        if self.0.len() < len {
            return Err("truncated");
        }
        let (head, rest) = self.0.split_at(len);
        self.0 = rest;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32, &'static str> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, &'static str> {
        let low = u64::from(self.u32()?);
        let high = u64::from(self.u32()?);
        Ok(low | (high << 32))
    }

    fn section(&mut self) -> Result<&'a [u8], &'static str> {
        let len = usize::try_from(self.u64()?).map_err(|_| "truncated")?;
        self.take(len)
    }
}

/// The reflection as JSON, with its static samplers' LOD fields beside it as
/// bit patterns.
///
/// metal2vulkan decodes those three `f32`s from AIR halves, so a sampler with
/// no upper clamp carries `inf` — and JSON has no infinity: serde_json writes
/// `null` and will not read it back. The JSON carries zeros there instead.
fn encode_reflection(reflection: &ShaderReflection) -> Result<(Vec<u8>, Vec<[u32; 3]>), String> {
    let mut plain = reflection.clone();
    let mut lods = Vec::new();
    for sampler in plain
        .bindings
        .iter_mut()
        .filter_map(|binding| binding.static_sampler.as_mut())
    {
        lods.push([
            sampler.lod_min_clamp.to_bits(),
            sampler.lod_max_clamp.to_bits(),
            sampler.lod_bias.to_bits(),
        ]);
        sampler.lod_min_clamp = 0.0;
        sampler.lod_max_clamp = 0.0;
        sampler.lod_bias = 0.0;
    }
    let json = serde_json::to_vec(&plain).map_err(|error| error.to_string())?;
    Ok((json, lods))
}

fn decode_reflection(json: &[u8], lods: &[[u32; 3]]) -> Result<ShaderReflection, String> {
    let mut reflection: ShaderReflection =
        serde_json::from_slice(json).map_err(|error| error.to_string())?;
    let mut samplers = reflection
        .bindings
        .iter_mut()
        .filter_map(|binding| binding.static_sampler.as_mut());
    let mut lods = lods.iter();
    loop {
        match (samplers.next(), lods.next()) {
            (Some(sampler), Some(&[min, max, bias])) => {
                sampler.lod_min_clamp = f32::from_bits(min);
                sampler.lod_max_clamp = f32::from_bits(max);
                sampler.lod_bias = f32::from_bits(bias);
            }
            (None, None) => return Ok(reflection),
            _ => return Err("static sampler count differs from its LOD groups".to_owned()),
        }
    }
}

fn is_entry_name(name: &str) -> bool {
    name.ends_with(ENTRY_SUFFIX) && !name.starts_with('.')
}

/// Write `bytes` to `dir/name` so that no reader ever sees part of them: into a
/// temporary file in the same directory, synced, then renamed over the name.
fn publish(dir: &Path, name: &str, bytes: &[u8]) -> std::io::Result<()> {
    // Process-wide, so two caches open on one directory in one process — the
    // tests, or a second device — cannot pick the same temporary name.
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let temporary = dir.join(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let published =
        write_synced(&temporary, bytes).and_then(|()| std::fs::rename(&temporary, dir.join(name)));
    if published.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    published
}

fn write_synced(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

// MARK: - This process's cache

/// `None` until first asked for, then the answer for the rest of the process.
static SHARED: Mutex<Option<Option<Arc<DiskCache>>>> = Mutex::new(None);

/// The cache this process translates through, opened on first use. `None` when
/// `REIMS_VGPU_M2V_DISK_CACHE=off`, when the build could not digest the
/// translator's source, or when the directory cannot be made; each says so once.
pub fn shared() -> Option<Arc<DiskCache>> {
    SHARED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(open_shared)
        .clone()
}

fn open_shared() -> Option<Arc<DiskCache>> {
    // A test that wants a cache installs one with `replace_shared_for_test`;
    // the rest must not write into the host's `$TMPDIR`.
    if cfg!(test) {
        return None;
    }
    match crate::env::read(crate::env::M2V_DISK_CACHE) {
        (Switch::Off, _) => {
            crate::observe::off("linux_m2v_disk disabled reason=env");
            return None;
        }
        (Switch::Unrecognized, value) => crate::observe::fail(format!(
            "linux_m2v_disk env_unrecognized value={} kept=on",
            value.unwrap_or_default()
        )),
        (Switch::Unset | Switch::On, _) => {}
    }
    if option_env!("REIMS_M2V_SOURCE_DIGEST").is_none_or(str::is_empty) {
        crate::observe::fail("linux_m2v_disk disabled reason=translator_source_unknown");
        return None;
    }
    let root = std::env::temp_dir().join(ROOT);
    let started = Instant::now();
    let translator = translator_fingerprint();
    match DiskCache::open(&root, &translator) {
        Ok(cache) => {
            crate::observe::off(format!(
                "linux_m2v_disk open dir={} entries={} m2v_commit={} open_ms={}",
                cache.dir.display(),
                cache.entries_at_open,
                env!("REIMS_M2V_COMMIT"),
                started.elapsed().as_millis()
            ));
            Some(Arc::new(cache))
        }
        Err(error) => {
            crate::observe::fail(format!(
                "linux_m2v_disk disabled reason=open root={} error={error}",
                root.display()
            ));
            None
        }
    }
}

#[cfg(test)]
pub(crate) fn replace_shared_for_test(cache: Option<Arc<DiskCache>>) {
    *SHARED.lock().unwrap_or_else(|e| e.into_inner()) = Some(cache);
}

/// [`DiskCache::load_or_translate`] on this process's cache, or `translate()`
/// when there is none. `pipeline_ref` is telemetry, as everywhere in
/// `m2v_cache`.
pub fn load_or_translate<E>(
    key: &Key<'_>,
    pipeline_ref: u32,
    translate: impl FnOnce() -> Result<Translation, E>,
) -> Result<Translation, E> {
    match shared() {
        Some(cache) => cache.load_or_translate(key, pipeline_ref, translate),
        None => translate(),
    }
}

/// This process's totals, when they have moved since the last summary.
/// `m2v_cache`'s worker calls it whenever its queue runs dry, so a boot's burst
/// of translations ends in one line.
pub fn summarize() {
    let Some(cache) = shared() else {
        return;
    };
    let counts = cache.counts();
    {
        let mut last = cache.summarized.lock().unwrap_or_else(|e| e.into_inner());
        if *last == counts {
            return;
        }
        *last = counts;
    }
    crate::observe::off(format!(
        "linux_m2v_disk summary {counts} entries_at_open={} dir={}",
        cache.entries_at_open,
        cache.dir.display()
    ));
}
