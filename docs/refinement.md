# Controlled refinement audit

This document records the earlier cleanup scope and proposed verification procedure. The final
accuracy pass is documented separately in [evaluation.md](evaluation.md). Repository verification
commits the reviewed cleanup after `fc1265f` and `22d1b5a`; it does not change perception logic or
reprocess footage. The final snapshot has 12 algorithm-validated measurements across ten videos;
the five originally zero-result videos remain unresolved, and Deck-2 is also zero after demotion.

Scope: `feature/challenge-2`, starting at `9d6fec7`. Existing commits, thresholds, learned models,
CUDA environments and generated evidence are preserved. This pass changes organization and
failure reporting, not the perception algorithm. No results were manually edited.

## Issues found and decisions

| Area | Finding | Refinement / decision |
|---|---|---|
| `reekon/events.py` | Event recognition, endpoint consensus, reference selection and classification shared one 419-line module | Extract `EndpointEstimate`, clustering and endpoint voting/correction to `endpoints.py`; event policy remains in `events.py` |
| Shared defaults | Sampling, detector processing widths, round-trip tolerance, NCC and JPEG quality repeated across modules | `config.py` gives shared values names and units; all original numeric values retained |
| Detector | `_robust_line` accepted center, direction and width arguments it never read; `as_dict` used a redundant temporary | Remove unused private parameters after checking every caller; simplify serialization |
| CLI | Missing paths produced tracebacks; nonpositive and nonfinite sampling rates were accepted; a directory with a video suffix could be selected | Check input/rate and require files in directory discovery; construct the index replacement-key set once |
| Pipeline | Ignored image encoder/writer success flags; an empty decode could reach later stages with zero dimensions | Raise clear errors and avoid publishing new JSON that claims failed images were saved |
| Interfaces | PTS versus header FPS, work-scale versus original pixels, boolean versus mask registration results were easy to confuse | Document coordinate/time and registration contracts; add focused endpoint type hints while preserving conditional JSON fields |
| `cvbench/` | Independent hardware experiment could be mistaken for production GPU acceleration; benchmark documentation mixed historical and later claims | Keep its code and reports unchanged; clarify optional PyTorch/psutil and historical baseline context |
| `tools/` | Video sampling and Jetson model loading resemble production code | Retain intentionally: inspection sampling, hand metadata and live-display overlays differ; merging them would change outputs or require a new abstraction |
| `viewer/` | Uses original result schema and old optional fields; candidate images and proxies are ignored artifacts | Retain viewer code; verify all indexed data and validated/scene images, and exercise all ten videos in WebKit |
| `outputs/`, `results/` | Mixed historical revisions and software environments; three videos predate labels | Preserve every file and provenance field; compare fresh runs separately rather than replacing evidence |
| Dependencies | Core NumPy/OpenCV and test pytest are used; benchmark Torch and optional psutil are isolated; Jetson imports are lazy | No confirmed unused package to remove; do not change installed environments or flatten CPU/GPU requirements |
| Documentation | README combined usage, full evaluation, environment troubleshooting and design discussion | Keep a concise entrypoint; separate architecture, environments, evaluation and benchmark methodology |
| Ignore rules | Only named venvs and specific data locations were covered | Ignore environment variants, raw video extensions, model/runtime weights, caches and temporary files without ignoring generated JSON/JPEG evidence |

The shared model default remains `~/jetson-inference/data`, configurable through the detector's
constructor and inspection tool flags. Changing working directory during model construction is
still process-global. Display `:0` and constant-rate export in `tools/live_view.py` are inspection
assumptions, not production video timing. Registration's 15-sample bridge horizon only means
approximately 1.5 seconds at the default 10 Hz. Local coefficients remain beside their original
calibration comments rather than being flattened into an opaque global threshold object.

## Earlier validation plan (not a new execution record)

The procedures below are retained as the earlier plan. Their description is not evidence that
every planned before/after or browser check was completed. No raw-video reprocessing, browser
playback run or expensive benchmark is performed during final repository verification. Actual
final-run checks and recorded test outcomes are in [evaluation.md](evaluation.md#regression-verification-and-preservation).

Run the existing suite before editing, after extraction, after boundary handling and before final
commits. CPU tests use `.venv`; CUDA parity uses the existing `.venv-jetson`. The opt-in Cabinet
pytest case is supplemented by full before/after classical runs of all ten raw videos.

Fresh results are written under `/tmp/reekon-refinement/{before,after}/<video>/`, with proxies
disabled. Compare every JSON field except `created_utc`, `pipeline_seconds`,
`pipeline_seconds_excluding_proxy` and `stage_timings`, which necessarily change between runs.
This includes all annotations, counts, observations, source frames, event timestamps, endpoint
coordinates, classifications, labels and scene diagnostics. Compare all generated JPEGs,
including candidate images, with SHA-256. These runs are correctness checks under concurrent
load, not replacement performance measurements.

Separately hash every existing file beneath `outputs/` and `results/`, including ignored local
proxies and diagnostic images. Check all 14 index entries (ten videos), result JSON, and every
validated/scene image reference. Ignored diagnostics/proxies are available locally but are not
promised in a fresh clone.

Browser checks use the installed WebKitGTK under Xvfb with a local server before and after changes.
Select each video, check measurement-card counts against JSON, load gallery images, and confirm
all ten proxies reach decoded video state without a media error. This does not establish Safari
coverage or human-verified overlay accuracy. Local HTTP Range behavior is covered by the suite.
Both strategies are independently smoke-tested through the CLI on a synthetic tape clip.

## Remaining issues, ranked

1. **Correctness evidence:** no labelled held-out ground truth; first-four-video threshold tuning
   limits generalization. The original Deck-2 snapshot has a known false validated result, demoted to uncertain in
   `outputs/accuracy-final/`; Deck-1 still has occluded-hook errors. Historical evidence is preserved.
2. **Coverage and geometry:** blade-less measurements are unsupported; missing hooks, parallax,
   lens distortion and colour-based occlusion limit endpoint localization and scene registration.
3. **Failure recovery and serving:** output replacement is not transactional; a late failure can
   leave partially replaced images. Existing index JSON is read after processing. The local
   server is not a hardened public deployment (mount symlinks can escape configured directories).
4. **Memory and reproducibility:** full-resolution JPEG caches grow with video duration; requirements
   are unpinned, and outputs span code/OpenCV versions. Source/environment provenance must accompany
   any new evaluation. Performance comparisons across the two environments are confounded.
5. **Optional-runtime ergonomics:** model loading changes process CWD; model location is not exposed
   by the production CLI. The cached TensorRT engine emits an existing device-model warning.
   No engine or CUDA package was changed.
6. **Testing/portability:** no Safari automation; the inspector has a different sampling loop and live
   export uses constant FPS. Restricted Jetson execution may prepend a diagnostic to ffprobe JSON;
   the existing timestamp test passes with normal device/socket access. No nominal-FPS fallback
   was added to hide such failures.
