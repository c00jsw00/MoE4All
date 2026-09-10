//! Vision input support: mmproj GGUF parsing and image preprocessing for the
//! Qwen3-VL-style CLIP vision tower (`general.architecture == "clip"`,
//! `clip.projector_type == "qwen3vl_merger"`).
//!
//! Reference implementation: llama.cpp `tools/mtmd/clip.cpp` and `tools/mtmd/models/qwen3vl.cpp`.
//! Where this crate pins observable behavior (patch ordering, position-embedding interpolation),
//! the doc comment on the pinning item cites the reference.
//!
//! # Supported scope
//!
//! Vision support is NARROW and gated at every layer; anything outside this matrix fails with an
//! explicit error rather than silently degrading:
//!
//! * **Projector**: `clip.projector_type == "qwen3vl_merger"` only (both the CPU tower and the
//!   Vulkan tower reject other projector types at load).
//! * **Deepstack**: `clip.is_deepstack_layers` must be ALL ZERO. Any nonzero entry bails at load
//!   (`VitEngine::load_cpu` / `VkVit::load_on`) — deepstack multi-layer feature injection into the
//!   trunk is NOT implemented.
//! * **Image inputs**: `data:` base64 URIs and bare base64 strings only
//!   ([`decode_image_input`]). Plain `http(s)://` URLs are NOT fetched — they bail with a clear
//!   error (a fetcher would add network + SSRF surface to the server).
//! Execution backends are deliberately kept outside this foundation module. This lets the parser,
//! tensor catalog and preprocessing tests stay backend-neutral while the production tower follows
//! the service's unified VRAM lifecycle.

mod config;
mod preprocess;
mod weights;

pub use config::ClipConfig;
pub use preprocess::{
    decode_image_input, merge_major_pos, prepare_image_bytes, smart_resize, PreparedImage,
    MAX_PATCHES,
};
pub use weights::{BlockWeights, VisionWeights};
