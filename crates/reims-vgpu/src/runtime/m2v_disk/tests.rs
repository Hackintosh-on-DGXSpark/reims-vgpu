use super::*;

use std::sync::atomic::AtomicBool;

use metal2vulkan::reflect::{
    DescriptorLocation, ResourceBinding, ResourceKind, SamplerAddressMode, SamplerBorderColor,
    SamplerCompareFunction, SamplerCoordinates, SamplerFilter, SamplerMipFilter, SamplerReduction,
    ShaderStage, StaticSamplerState, REFLECTION_VERSION, RESOURCE_DESCRIPTOR_SET,
    SAMPLER_BINDING_BASE,
};

// MARK: - Support

/// A directory under the host's temp dir, removed when dropped.
pub(crate) struct TempRoot(PathBuf);

impl TempRoot {
    pub(crate) fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "reims-m2v-disk-test-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("a test root");
        Self(path)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `cache` is this process's [`shared`] cache until the guard drops.
pub(crate) struct Installed;

impl Installed {
    pub(crate) fn new(cache: Arc<DiskCache>) -> Self {
        replace_shared_for_test(Some(cache));
        Self
    }
}

impl Drop for Installed {
    fn drop(&mut self) {
        replace_shared_for_test(None);
    }
}

/// A module with a valid header and `words` words of body that `tag` makes
/// distinguishable from any other tag's.
fn spirv(tag: u32, words: usize) -> Vec<u8> {
    let mut module = vec![SPIRV_MAGIC, 0x0001_0300, 0, 1, 0];
    module.extend((0..words as u32).map(|i| i.wrapping_mul(0x9e37_79b9) ^ tag));
    module.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// A fragment reflection with one static sampler whose LOD fields are `lods`.
fn reflection(lods: [f32; 3]) -> ShaderReflection {
    let [lod_min_clamp, lod_max_clamp, lod_bias] = lods;
    ShaderReflection {
        reflection_version: REFLECTION_VERSION,
        stage: ShaderStage::Fragment,
        entry_point: Some("fragment_main".to_owned()),
        bindings: vec![ResourceBinding {
            kind: ResourceKind::Sampler,
            metal_index: 2,
            descriptor: Some(DescriptorLocation {
                set: RESOURCE_DESCRIPTOR_SET,
                binding: SAMPLER_BINDING_BASE + 2,
                count: 1,
            }),
            param_index: None,
            stage_input_location: None,
            address_space: None,
            declared_size: None,
            extent: None,
            footprint: None,
            type_layout: None,
            type_name: None,
            texture_shape: None,
            embedded_source: None,
            access: None,
            static_sampler: Some(StaticSamplerState {
                min_filter: SamplerFilter::Linear,
                mag_filter: SamplerFilter::Linear,
                mip_filter: SamplerMipFilter::None,
                address_mode_s: SamplerAddressMode::ClampToEdge,
                address_mode_t: SamplerAddressMode::ClampToEdge,
                address_mode_r: SamplerAddressMode::ClampToEdge,
                coordinates: SamplerCoordinates::Normalized,
                compare_function: SamplerCompareFunction::None,
                max_anisotropy: 1,
                lod_min_clamp,
                lod_max_clamp,
                border_color: SamplerBorderColor::TransparentBlack,
                reduction: SamplerReduction::WeightedAverage,
                lod_bias,
                raw_words: [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210],
            }),
        }],
        argument_buffer_fields: vec![],
        vertex_attributes: vec![],
        varyings: vec![],
        render_targets: vec![],
        depth_members: vec![],
        depth_qualifier: None,
        stencil_members: vec![],
        local_size: None,
        vertex_builtins: None,
        tessellation: None,
        imageblock_layouts: vec![],
        implicit_imageblock_attachments: vec![],
        fragment_imageblock: None,
        datalayout: Some("e-p:64:64:64-i64:64:64".to_owned()),
        function_constants: vec![],
    }
}

fn translation(tag: u32) -> Translation {
    Translation {
        spirv: spirv(tag, 4096),
        reflection: reflection([0.0, f32::INFINITY, 0.0]),
    }
}

/// `bytes` with its trailing checksum recomputed, so a check behind the
/// checksum can be reached.
fn resealed(mut bytes: Vec<u8>) -> Vec<u8> {
    bytes.truncate(bytes.len() - CHECKSUM_LEN);
    let checksum = sha256(&bytes);
    bytes.extend_from_slice(&checksum);
    bytes
}

fn refusal(lookup: Lookup) -> Option<&'static str> {
    match lookup {
        Lookup::Refused(reason) => Some(reason),
        Lookup::Hit(_) | Lookup::Miss => None,
    }
}

fn is_miss(lookup: Lookup) -> bool {
    matches!(lookup, Lookup::Miss)
}

// MARK: - SHA-256

#[test]
fn sha256_matches_the_published_vectors() {
    assert_eq!(
        sha256_hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(&sha256(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    // A million `a`, fed in pieces that straddle every block boundary.
    let mut hasher = Sha256::default();
    let piece = [b'a'; 997];
    let mut left = 1_000_000;
    while left > 0 {
        let n = left.min(piece.len());
        hasher.update(&piece[..n]);
        left -= n;
    }
    assert_eq!(
        sha256_hex(&hasher.finish()),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

// MARK: - Fingerprint

/// The fingerprint carries the build's digest of the translator's source and
/// a line for each tool, and any `METAL2VULKAN_*` variable moves it — which
/// moves the directory, so nothing translated under the old one is found.
#[test]
fn the_fingerprint_names_the_translator_build_and_environment() {
    const PROBE: &str = "METAL2VULKAN_REIMS_DISK_CACHE_TEST_PROBE";
    let base = translator_fingerprint();
    let source = base
        .lines()
        .find_map(|line| line.strip_prefix("m2v_source "))
        .expect("a source line");
    assert_eq!(source.len(), 64, "{base}");
    assert!(source.bytes().all(|b| b.is_ascii_hexdigit()), "{base}");
    for tool in TOOLS {
        assert!(base.contains(&format!("\ntool {tool} ")), "{base}");
    }

    // SAFETY: no other test reads or writes this variable, and metal2vulkan
    // reads only the variables it names.
    unsafe { std::env::set_var(PROBE, "1") };
    let changed = translator_fingerprint();
    unsafe { std::env::remove_var(PROBE) };
    assert!(changed.contains(&format!("\nenv {PROBE}=1\n")), "{changed}");
    assert_ne!(base, changed);

    let root = TempRoot::new("fingerprint");
    let before = DiskCache::open(root.path(), &base).expect("open");
    let after = DiskCache::open(root.path(), &changed).expect("open");
    assert_ne!(before.dir(), after.dir());
    assert_eq!(
        std::fs::read_to_string(after.dir().join("translator.txt")).expect("the note"),
        changed
    );
}

// MARK: - Across processes

/// What one process publishes, the next loads without translating. Each
/// "process" is a separately opened cache on the same directory.
#[test]
fn an_entry_published_by_one_process_is_a_hit_in_the_next() {
    let root = TempRoot::new("hit");
    let options = TransformOptions::default();
    let air = b"air the first process translated".to_vec();
    let key = Key::new(Stage::Fragment, &options, &air);
    // NaN with a payload, infinity and negative zero: every LOD value JSON
    // would lose or blur.
    let lods = [f32::from_bits(0x7fc0_1234), f32::INFINITY, -0.0];
    let made = Translation {
        spirv: spirv(7, 8192),
        reflection: reflection(lods),
    };
    let translations = AtomicU64::new(0);
    let translate = || {
        translations.fetch_add(1, Ordering::Relaxed);
        Ok::<_, ()>(Translation {
            spirv: made.spirv.clone(),
            reflection: made.reflection.clone(),
        })
    };

    let first = DiskCache::open(root.path(), "translator A").expect("open");
    assert_eq!(first.entries_at_open(), 0);
    first
        .load_or_translate(&key, 22, translate)
        .expect("translates");
    assert_eq!(
        first.counts(),
        Counts {
            misses: 1,
            stored: 1,
            ..Counts::default()
        }
    );
    drop(first);

    let second = DiskCache::open(root.path(), "translator A").expect("reopen");
    assert_eq!(second.entries_at_open(), 1);
    let loaded = second
        .load_or_translate(&key, 22, translate)
        .expect("loads");
    assert_eq!(
        translations.load(Ordering::Relaxed),
        1,
        "the second process translated"
    );
    assert_eq!(
        second.counts(),
        Counts {
            hits: 1,
            ..Counts::default()
        }
    );
    assert_eq!(loaded.spirv, made.spirv);
    assert_eq!(
        format!("{:?}", loaded.reflection),
        format!("{:?}", made.reflection)
    );
    let sampler = loaded.reflection.bindings[0]
        .static_sampler
        .expect("the sampler survives");
    assert_eq!(
        [
            sampler.lod_min_clamp.to_bits(),
            sampler.lod_max_clamp.to_bits(),
            sampler.lod_bias.to_bits()
        ],
        lods.map(f32::to_bits)
    );
}

// MARK: - Translator and configuration

/// Nothing published under one translator or one set of inputs is found under
/// another — and an entry carried to where another key would look for it is
/// refused by what it says inside, not trusted for where it sits.
#[test]
fn a_different_translator_or_input_is_never_a_hit() {
    let root = TempRoot::new("keys");
    let air = b"air keyed several ways".to_vec();
    let other_air = b"air keyed several ways, edited".to_vec();
    let render = TransformOptions::default();
    let kernel_16 = TransformOptions {
        kernel_local_size: [16, 16, 1],
        ..TransformOptions::default()
    };
    let kernel_8 = TransformOptions {
        kernel_local_size: [8, 16, 1],
        ..TransformOptions::default()
    };
    let multisampled = TransformOptions {
        raster_sample_count: Some(4),
        ..TransformOptions::default()
    };

    let a = DiskCache::open(root.path(), "translator A").expect("open A");
    let fragment = Key::new(Stage::Fragment, &render, &air);
    let kernel = Key::new(Stage::Kernel, &kernel_16, &air);
    for key in [&fragment, &kernel] {
        let made = translation(1);
        a.store(key, &made.spirv, &made.reflection).expect("store");
        assert!(matches!(a.load(key), Lookup::Hit(_)));
    }

    // Another translator: another directory, and nothing in it.
    let b = DiskCache::open(root.path(), "translator B").expect("open B");
    assert_ne!(a.dir(), b.dir());
    assert!(is_miss(b.load(&fragment)));
    assert!(is_miss(b.load(&kernel)));

    // Same translator, any other input.
    for (what, key) in [
        ("stage", Key::new(Stage::Vertex, &render, &air)),
        ("air", Key::new(Stage::Fragment, &render, &other_air)),
        ("local size", Key::new(Stage::Kernel, &kernel_8, &air)),
        ("options", Key::new(Stage::Fragment, &multisampled, &air)),
    ] {
        assert!(is_miss(a.load(&key)), "{what}");
    }

    // A's entry where B would look for the same key.
    let entry = std::fs::read(a.dir().join(a.entry_name(&fragment))).expect("A's entry");
    std::fs::write(b.dir().join(b.entry_name(&fragment)), &entry).expect("plant");
    assert_eq!(refusal(b.load(&fragment)), Some("translator"));

    // A's fragment entry where A would look for other inputs.
    for (expected, key) in [
        ("stage", Key::new(Stage::Vertex, &render, &air)),
        ("options", Key::new(Stage::Fragment, &multisampled, &air)),
        ("air", Key::new(Stage::Fragment, &render, &other_air)),
    ] {
        std::fs::write(a.dir().join(a.entry_name(&key)), &entry).expect("plant");
        assert_eq!(refusal(a.load(&key)), Some(expected));
    }
}

// MARK: - Damage

/// A short, empty, flipped or padded entry is refused rather than loaded, and
/// the next translation replaces it with a whole one.
#[test]
fn a_damaged_entry_is_refused_and_replaced() {
    let root = TempRoot::new("damage");
    let options = TransformOptions::default();
    let air = b"air whose entry gets damaged".to_vec();
    let key = Key::new(Stage::Fragment, &options, &air);
    let cache = DiskCache::open(root.path(), "translator A").expect("open");
    let made = translation(3);
    cache
        .store(&key, &made.spirv, &made.reflection)
        .expect("store");
    let path = cache.dir().join(cache.entry_name(&key));
    let whole = std::fs::read(&path).expect("the entry");

    let flipped = |at: usize| {
        let mut bytes = whole.clone();
        bytes[at] ^= 0x10;
        bytes
    };
    let mut padded = whole.clone();
    padded.push(0);
    // Behind a checksum that matches, the header still has to.
    let magic = resealed(flipped(0));
    let format = resealed(flipped(MAGIC.len()));
    let damaged = [
        ("empty", Vec::new(), "truncated"),
        ("half", whole[..whole.len() / 2].to_vec(), "checksum"),
        (
            "one byte short",
            whole[..whole.len() - 1].to_vec(),
            "checksum",
        ),
        ("a body bit", flipped(whole.len() / 2), "checksum"),
        ("a checksum bit", flipped(whole.len() - 1), "checksum"),
        ("a byte past the end", padded, "checksum"),
        ("resealed magic", magic, "magic"),
        ("resealed format", format, "format"),
    ];
    for (what, bytes, expected) in damaged {
        std::fs::write(&path, &bytes).expect("damage");
        assert_eq!(refusal(cache.load(&key)), Some(expected), "{what}");
    }

    let republished = cache
        .load_or_translate(&key, 9, || Ok::<_, ()>(translation(3)))
        .expect("translates again");
    assert_eq!(republished.spirv, made.spirv);
    assert_eq!(std::fs::read(&path).expect("republished"), whole);
    assert!(matches!(cache.load(&key), Lookup::Hit(_)));
    assert_eq!(cache.counts().refused, 9);
    assert_eq!(cache.counts().stored, 2);
}

/// A module that does not start with the SPIR-V magic word is refused even
/// when everything around it is consistent.
#[test]
fn an_entry_that_is_not_spirv_is_refused() {
    let root = TempRoot::new("spirv");
    let options = TransformOptions::default();
    let air = b"air with a bad module".to_vec();
    let key = Key::new(Stage::Fragment, &options, &air);
    let cache = DiskCache::open(root.path(), "translator A").expect("open");
    let mut not_spirv = spirv(4, 64);
    not_spirv[0] ^= 0xff;
    cache
        .store(&key, &not_spirv, &reflection([0.0; 3]))
        .expect("store");
    assert_eq!(refusal(cache.load(&key)), Some("spirv"));
}

// MARK: - Concurrency

/// Two QEMUs translating the same shader publish the same key at once. However
/// the renames interleave, the directory ends with one whole entry and no
/// temporary files, and a reader during the race sees either nothing or a
/// whole entry — never part of one.
#[test]
fn writers_racing_on_one_key_leave_one_whole_entry() {
    const WRITERS: u32 = 6;
    const ROUNDS: u32 = 12;
    let root = TempRoot::new("race");
    let options = TransformOptions::default();
    let air = b"air two processes translate at once".to_vec();
    let candidates: Vec<Vec<u8>> = (0..WRITERS)
        .map(|writer| spirv(writer, 2048 + writer as usize))
        .collect();
    let reflection = reflection([0.0, f32::INFINITY, 0.0]);
    let start = std::sync::Barrier::new(WRITERS as usize + 2);
    let done = AtomicBool::new(false);
    let reads = AtomicU64::new(0);

    std::thread::scope(|scope| {
        let mut writers = Vec::new();
        for writer in 0..WRITERS {
            let (start, candidates, reflection, air, options) =
                (&start, &candidates, &reflection, &air, &options);
            let root = root.path();
            writers.push(scope.spawn(move || {
                let cache = DiskCache::open(root, "translator A").expect("open");
                let key = Key::new(Stage::Fragment, options, air);
                start.wait();
                for _ in 0..ROUNDS {
                    cache
                        .store(&key, &candidates[writer as usize], reflection)
                        .expect("publish");
                }
            }));
        }
        for _ in 0..2 {
            let (start, candidates, air, options, done, reads) =
                (&start, &candidates, &air, &options, &done, &reads);
            let root = root.path();
            scope.spawn(move || {
                let cache = DiskCache::open(root, "translator A").expect("open");
                let key = Key::new(Stage::Fragment, options, air);
                start.wait();
                while !done.load(Ordering::Acquire) {
                    match cache.load(&key) {
                        Lookup::Hit(hit) => assert!(candidates.contains(&hit.spirv)),
                        Lookup::Miss => {}
                        Lookup::Refused(reason) => {
                            panic!("a reader saw part of an entry: {reason}")
                        }
                    }
                    reads.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        for writer in writers {
            writer.join().expect("a writer");
        }
        done.store(true, Ordering::Release);
    });
    assert!(reads.load(Ordering::Relaxed) > 0);

    let cache = DiskCache::open(root.path(), "translator A").expect("reopen");
    let mut names: Vec<String> = std::fs::read_dir(cache.dir())
        .expect("the directory")
        .map(|entry| {
            entry
                .expect("an entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name != "translator.txt")
        .collect();
    names.sort();
    let key = Key::new(Stage::Fragment, &options, &air);
    assert_eq!(names, vec![cache.entry_name(&key)]);
    let Lookup::Hit(hit) = cache.load(&key) else {
        panic!("the surviving entry does not load");
    };
    assert!(candidates.contains(&hit.spirv));
    assert_eq!(cache.counts().refused, 0);
}
