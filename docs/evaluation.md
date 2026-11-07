# Ten-video evaluation and measured performance

## Final accuracy pass (2026-11-07)

This pass starts at `bf29ea8` on `feature/challenge-2`, with the prior uncommitted error-handling
and documentation cleanup already present. Original `outputs/` runs and `results/` reports are
preserved. Evaluation is read-only visual inspection of sampled raw frames, endpoint close-ups,
and generated images, not manual annotations supplied to the algorithm or labelled ground truth.
The ten classical runs are the regression set; the four historical hand-assisted runs remain
historical evidence, not a new comparison of strategies. No thresholds were tuned per video.

### Failure diagnosis

| Video | Primary failure | Secondary limitation |
|---|---|---|
| Door-1 | Tape detection: only 17/351 samples previously passed. A longer door gap can win seed selection and then fail verification, suppressing the real blade. | Sparse housing sightings fail reading duration/support requirements; later hook ends are often outside the view. Curved blades also violate the straight-strip assumption. |
| Frame Pipes-1 | Endpoint evidence: the long reading sees the housing but usually cuts the hook off at the right border. | Earlier hook sightings occur in separate short episodes; only isolated sightings show both ends. Camera motion, identity filtering and the three-observation consensus requirement prevent a reliable hook estimate. Scene registration is downstream of this failure. |
| Kitchen Cabinets | Detection and event recognition: cabinet edges/handles compete with the blade, and fragmented/partly covered housings rarely produce sustained readings. | Hooks are frequently held behind fingers or outside the image. The colour occlusion check can also confuse wood with skin. Mapping does not establish an endpoint that was never reliably observed. |
| Large Window | Unsupported observation: sampled measuring poses show the device against framing with its blade retracted. Short detections of dark structural edges are not measurement evidence. | The existing hand-pose model locates hands, not device contact points or a distant measurement endpoint. Green laser traces and framing edges do not establish an A/B correspondence. No alternative endpoint strategy using the current components was validated. |
| Outdoor_Pipes | Detection plus missing hook evidence: the vertical reading shows the housing while the hook is below the frame; later horizontal placements have few accepted blade sightings. | Hands, foreshortening and competing pipe edges break detection continuity. There is no supported hook cluster in the original reading; registration is not the first failure. |

### Targeted changes and rejected probes

Evidence images: [rejected hand-as-housing result](../results/accuracy-final/rejected-seed-search.jpg),
[Deck-2 retained as uncertain](../results/accuracy-final/deck2-entry-uncertain.jpg), and
[continuous Deck-1 placement across the old split](../results/accuracy-final/deck1-continuous-placement.jpg).

The final trial addresses housing-entry localization and observability while keeping the current
pipeline. The longest wide run still identifies the housing side, but the innermost run meeting
the existing minimum run length locates the blade entry. A hand/LCD can split the housing mask;
choosing the longest body's inner edge alone can put B on the housing body. A synthetic split
housing reproduced the old 45-pixel error; the corrected entry is within 3 pixels in that fixture.
Tests cover all four image rotations. These fixture errors are not accuracy measurements on the
test videos.

Entry selection alone was insufficient: it corrected B in Deck-2's inspected frame at 13.83 s,
but the full event selected another frame with B still on the body. That version was not accepted.
The final trial additionally applies the existing two-sided strip contrast threshold locally on
the blade side of B (up to six blade widths, independent of video resolution). If the local strip
is unsupported, housing recognition still preserves the reading, but the entry is not marked
visible. Endpoint voting now honors that visibility flag, and reference selection skips frames
with unsupported entries. An unsupported body fragment can no longer vote as an observed entry
merely through repeated agreement. Passing local contrast is still not proof of an exact physical
junction; some Countertop-1 housing placement remains imprecise. Synthetic tests cover both exclusion from voting and retention of
an uncertain reading, including a minority of actual observed entries among many body positions.

A broader seed-search experiment tried blade-shaped components longest first until one passed
all existing tape checks. It recovered real blade observations in Door-1 samples, but the Deck-1
full run introduced a false validated event at 33.3–35.2 s: a hand pressing the blade became the
housing endpoint. Cabinet-1 stayed at four and Countertop-1 at two; Deck-1 changed from 16 events /
6 validated to 16 / 5, including that new false result. This experiment was **reverted**, including
its synthetic test. More detections alone did not establish better measurement correctness. Its
intermediate runs are not the final outputs.

Hook-continuation and local tick-alternation probes were inspected but not retained. They did
not reliably distinguish all partial hooks from genuine tips and could reject legitimate nearby
structure or blade ends. A stricter border margin was not justified: a raw close-up of Deck-2's
old reference frame actually shows its brass hook. The historical assessment below calls that
hook a frame-edge error; the independently visible problem is B on the housing body. The entry
correction moves B to the visible blade junction in that inspected frame, but that frame-level
result alone is not evidence of a correct final event.

No hand model, model training, dependency, CUDA configuration, event-duration threshold, endpoint
consensus threshold, or registration acceptance threshold was changed. Unobserved/unsupported
endpoints remain missing and events remain diagnostic candidates. This does **not** solve all
over-exposed-finger hooks: temporal agreement can consistently agree on an occlusion boundary.

Scene mapping retains its bidirectional, in-frame and appearance checks and existing grouping.
No clean frame is invented by weakening correspondence requirements.

### Final before/after results

All ten classical videos were reprocessed at 10 Hz into `outputs/accuracy-final/`, with proxy
encoding disabled. Original results remain at their original paths. Machine-readable counts,
software versions, source hashes and artifact checks are in
[`results/accuracy-final/comparison.json`](../results/accuracy-final/comparison.json).
Rejected experiments are recorded separately in
[`experiments.json`](../results/accuracy-final/experiments.json).

| Video | Tape observations | Candidate events | Validated | Uncertain / rejected | Seconds excluding proxy |
|---|---:|---:|---:|---:|---:|
| Cabinet-1 | 253 → 253 | 4 → 4 | 4 → 4 | 0/0 → 0/0 | 86.38 → 85.31 |
| Countertop-1 | 439 → 439 | 8 → 8 | 2 → 2 | 6/0 → 6/0 | 156.74 → 153.67 |
| Deck-1 | 386 → 393 | 16 → 15 | 6 → 5 | 10/0 → 10/0 | 181.84 → 183.48 |
| Deck-2 | 61 → 61 | 3 → 3 | 1 → 0 | 2/0 → 3/0 | 50.22 → 36.34 |
| Door-1 | 17 → 17 | 0 → 0 | 0 → 0 | 0/0 → 0/0 | 56.47 → 54.55 |
| Frame Pipes-1 | 90 → 88 | 1 → 1 | 0 → 0 | 1/0 → 1/0 | 37.38 → 37.77 |
| Kitchen Cabinets | 242 → 246 | 2 → 2 | 0 → 0 | 1/1 → 1/1 | 361.62 → 325.55 |
| Kitchen Sink | 104 → 104 | 1 → 1 | 1 → 1 | 0/0 → 0/0 | 92.55 → 75.06 |
| Large Window | 21 → 21 | 0 → 0 | 0 → 0 | 0/0 → 0/0 | 80.25 → 80.94 |
| Outdoor_Pipes | 44 → 44 | 1 → 1 | 0 → 0 | 1/0 → 1/0 | 48.57 → 48.15 |
| **Total** | **1657 → 1666** | **36 → 35** | **14 → 12** | **21/1 → 22/1** | |

These are recorded wall times, not a speedup benchmark: the before runs are historical and system
load was not controlled. The final ten-video pass was sequential; the discarded experiments ran
concurrently and their times are not comparable. Optional ffprobe container-rate fields are null
in restricted execution because of the existing EGL-prefixed JSON problem. Decoder frame counts,
sample counts and every sampled PTS timestamp match the baseline; no nominal-FPS fallback was used.

### Visual endpoint comparison and limitations

The following are qualitative judgements of the generated images and selected raw frames. They
are not precision/recall, pixel-error measurements, or an independently labelled endpoint set.
"Plausible" means visually consistent with tape endpoints, not proven physical measurement accuracy.

| Video | Before → after visual finding | False validation / coverage assessment |
|---|---|---|
| Cabinet-1 | Four plausible pairs → four retained. #1, #3 and #4 keep the same canvas and endpoint coordinates; #2 uses another clear frame 0.2 s later. | Mandatory four-measurement regression passes; no new visible endpoint defect. |
| Countertop-1 | Two plausible measurement events → two retained; housing placement remains imprecise in the saved lines. | No new validated event; exact B localization is not solved by local contrast alone. Six candidates remain uncertain. |
| Deck-1 | Three plausible lines, two partial and one ambiguous → three plausible, one partial and one ambiguous. | Old #2/#3, 46.9–50.1 s, show one continuous placement in the inspected sequence and now form one event. The earlier partial-hook duplicate is removed. Old #4 (new #3) remains partial; #1 remains ambiguous. No new hand-as-housing event appears in the retained run. |
| Deck-2 | One line with B on the housing body → no validated line; its hook remains observed and B is missing in candidate C2. | One known bad validation is demoted to uncertainty. Three candidates are retained. This is reduced false validation, not a recovered measurement. |
| Door-1 | No complete pair → none. | Detection/reading coverage remains unresolved; no validated false positive in either saved run. |
| Frame Pipes-1 | No complete pair → none; housing localized, hook missing in the retained candidate. | Candidate starts 0.2 s earlier; two fewer tape observations. No recovery of the physical hook, and no validated false positive in either run. |
| Kitchen Cabinets | No complete pair → none; one candidate lacks both endpoints and one lacks the hook. | Four more tape observations do not establish a complete measurement. One uncertain and one rejected event remain; no validated false positive in either run. |
| Kitchen Sink | One plausible pair → one retained near the frame edge. | No loss of the existing plausible result; no new validated event. |
| Large Window | No blade-based endpoint pair → none. | Existing components cannot establish a reliable blade-less pair; no fabricated event or endpoints. |
| Outdoor_Pipes | No complete pair → none; hook remains missing. | One uncertain candidate remains; no validated false positive in either run. |

**No safe recall gain was demonstrated in the five originally zero-result videos.** Deck-2
also has zero validated measurements after demotion, so the final snapshot contains **six**
zero-result videos in total. The common
blocker is missing trustworthy hook evidence, compounded by detection fragmentation. The broader
search that recovered Door-1 observations failed cross-video correctness and was rejected. The
retained changes address a separately reproduced housing-end reliability error. They do not
establish all physical endpoints correctly, and remaining partial hooks can still be validated.

Registered scene views decrease from nine to seven and total scene canvases from thirteen to
eleven as the bad Deck-2 validation and split Deck-1 reading disappear. All twelve validated
measurements retain a scene placement through the existing grouping/fallback path. No clean-view
scene was accepted before or after. Registration code and acceptance checks remain unchanged;
no additional clear-frame correspondence was demonstrated and no transform was forced.

### Regression verification and preservation

- CPU suite: `.venv/bin/python -m pytest -q` — **39 passed, 10 skipped** (nine CUDA cases in the
  CPU environment and the opt-in footage pytest case). The full Cabinet-1 pipeline run separately
  verifies four measurements. Socket/device access is needed for the existing HTTP/ffprobe tests;
  restricted execution alone gives their previously observed environmental failures.
- Existing Jetson environment: nine focused detector/endpoint tests passed, twenty deselected,
  under Python 3.10.12 / OpenCV 4.8.0 / NumPy 1.26.4. These were CPU compatibility tests, not a new
  CUDA or hand-model evaluation. No packages, drivers, runtime settings or weights were changed.
- All ten result JSONs and index entries, referenced measurement/scene/candidate images,
  validated endpoint support, finite coordinates, frame counts and sampled PTS timelines checked.
- SHA-256 checks confirm **all 111 original files under `outputs/` and `results/` are unchanged**,
  including ignored local proxies and diagnostics. Original hashes are recorded in
  [`original-artifact-hashes.json`](../results/accuracy-final/original-artifact-hashes.json).
- At the end of the accuracy pass, the earlier cleanup and README restructuring were left
  uncommitted. The subsequent repository-verification pass reviews and commits that work
  separately, without changing predictions, detection algorithms, thresholds or generated files.
  The existing CPU suite was rerun: **39 passed, 10 skipped**, with the existing CUDA-driver
  compatibility warning. No video reprocessing or expensive benchmark was repeated.

## Preserved evaluation before the final accuracy pass

The following sections retain the earlier evaluation and timings. They describe the original
saved runs, including visual judgements that the final pass may qualify above.

The original AI-assisted evaluation judged each image visually. Those checks are retained as reported evidence, are not ground truth, and are not part of the pipeline. Nothing here is a quantitative accuracy figure: there's no ground truth. All ten were run with the classical strategy at the default settings (10 samples/s). The detection thresholds were set on the first four videos and were **not** retuned for the other six.

### Overview

| Video | Length | Tape samples | Candidate events | Validated | Uncertain / rejected | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Cabinet-1 | 33 s | 253 of 327 | 4 | 4 | 0 / 0 | ✅ 4 correct |
| Countertop-1 | 97 s | 439 of 966 | 8 | 2 | 6 / 0 | ✅ 2 correct, ⚠️ 6 uncertain |
| Deck-1 | 75 s | 386 of 749 | 16 | 6 | 10 / 0 | ⚠️ 3 plausible, 2 partial, 1 uncertain; 10 uncertain |
| Deck-2 | 20 s | 61 of 199 | 3 | 1 | 2 / 0 | ❌ the 1 validated is wrong; 2 uncertain |
| Door-1 | 35 s | 17 of 351 | 0 | 0 | 0 / 0 | ❌ missed: no measurement event detected |
| Frame Pipes-1 | 15 s | 90 of 155 | 1 | 0 | 1 / 0 | ⚠️ 1 uncertain (hook not localised) |
| Kitchen Cabinets | 180 s | 242 of 1797 | 2 | 0 | 1 / 1 | ⚠️ 1 uncertain, 1 rejected (no endpoint) |
| Kitchen Sink | 39 s | 104 of 387 | 1 | 1 | 0 / 0 | ✅ 1 plausible |
| Large Window | 57 s | 21 of 572 | 0 | 0 | 0 / 0 | ❌ none: the measuring device is held against the framing with no extended blade, so this detector has nothing to find |
| Outdoor_Pipes | 26 s | 44 of 259 | 1 | 0 | 1 / 0 | ⚠️ 1 uncertain (hook not localised) |
| **Total** | | | **36** | **14** | **21 / 1** | |

Key: ✅ validated and checked correct · ⚠️ uncertain, or validated but not clearly right · ❌ wrong or nothing found. Only validated measurements are drawn as measurements. Uncertain and rejected candidates are kept in `results.json`, and in `diagnostics/`, with their reasons.

**Width/height labels.** Each validated measurement is labelled height, width or diagonal from the direction of its line in the reading frame. A left/right or top/bottom position is added only when a validated mapping lets two measurements of the same type be compared. The labels are present for Cabinet-1 (2 heights, 2 widths) and Countertop-1 (2 diagonals). Deck-1, Deck-2 and Kitchen Sink were processed before labelling was added and weren't rerun, so their validated measurements have no labels yet. The other five videos have no validated measurements to label.

### First four videos, in detail

| Video | Candidate events | Validated | Visual check of validated | Unconfirmed candidates | Scene images (registered views) | Compared with the previous pipeline |
|---|---:|---:|---|---|---:|---|
| Cabinet-1 (33 s, indoor cabinets) | 4 | 4 | 4 correct: both door heights and both widths | none | 4 (2) | Same 4 correct measurements, no regression |
| Countertop-1 (97 s, marble counters) | 8 | 2 | 2 correct diagonal counter measurements | 6 uncertain: 4 with the housing end right but the hook not localised, 1 low-confidence, 1 correct measurement now unconfirmed (see regression) | 2 (1) | The false "carrying" event is gone. **Regression:** the 31–32 s measurement, correct before, is now uncertain |
| Deck-1 (75 s, outdoor deck) | 16 | 6 | #3, #5, #6 plausible (vertical rail-to-rail); #2, #4 partial (hook placed where the blade goes behind over-exposed fingers); #1 uncertain | 10 uncertain, mostly the housing end right and the hook not localised | 5 (4), including #4 and #5 grouped on one frame | No longer confirms the railing-gap false line, the swapped-ends case or the frame-edge hooks |
| Deck-2 (20 s, outdoor deck) | 3 | 1 | **Wrong:** A at the frame edge, B on the housing body | 2 uncertain | 1 (1) | The vegetation-edge and pole false events are gone; the remaining confirmed one is still wrong |

### Six added videos, in detail

| Video | Validated: visual check | Unconfirmed candidates | Scene images (registered views) | Why so few |
|---|---|---|---:|---|
| Door-1 (35 s) | none | none | 0 | The tape was seen in only 17 samples, too few for a reading |
| Frame Pipes-1 (15 s) | none | 1 uncertain, 6.1–14.3 s: hook end not localised | 0 | The blade was seen often, but its far end was never confirmed |
| Kitchen Cabinets (180 s) | none | 1 rejected, 101.2–102.1 s: no endpoint localised; 1 uncertain, 159.0–160.3 s: hook not localised | 0 | The tape was visible in only 13% of samples over a long video |
| Kitchen Sink (39 s) | #1 plausible: along the top edge of a cabinet carcass, from back to front. A is close to the left edge of the frame | none | 1 (1) | n/a |
| Large Window (57 s) | none | none | 0 | 21 blade-like detections in short bursts of up to 1 s, mostly without the housing. In the readings the device is pressed against the framing with its display showing and no blade extended. This is outside what a blade detector can measure |
| Outdoor_Pipes (26 s) | none | 1 uncertain, 6.1–7.0 s: hook not localised | 0 | Few tape samples (44 of 259) |

**Scene images, honestly:**
- **What they are:** the "registered views" are nearby frames chosen for less obstruction. They still show the tape and hands.
- **What was rejected:** frames that show the measured object unobstructed (approach frames) were rejected by the reliability checks on this footage. The scene-registration log in each `results.json` breaks down why: broken chains during fast head motion, round-trip error, endpoints out of view, or the appearance check. I'd rather show the reading frame than draw a line in the wrong place.
- **What the appearance check caught:** before it was added, one deck measurement was drawn on the beach instead of the railing. That mapping had passed the round-trip check, because the error was consistent in both directions.

**Previous vs new pipeline, in short:** fewer confirmed measurements, but far fewer wrong ones presented as confirmed. Everything rejected is kept as a diagnostic candidate with its reason.

## Performance

### Pipeline, per video (measured on this Jetson; wall-clock seconds; browser video encode excluded)

| Video | Strategy | Decode | Detect | Registration prep | Frame cache | Events + endpoints | Scene registration | Annotate | **Total** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cabinet-1 | classical | 23.1 | 19.0 | 4.9 | 7.7 | 14.9 | 10.9 | 0.7 | **81.2** |
| Cabinet-1 | hand-assisted | 7.0 | 45.5 | 8.0 | 9.1 | 19.5 | 12.7 | 1.1 | **124.3** |
| Countertop-1 | classical | 55.2 | 40.1 | 13.4 | 18.5 | 17.1 | 7.9 | 0.4 | **152.7** |
| Countertop-1 | hand-assisted | 13.7 | 97.1 | 20.9 | 21.8 | 21.0 | 19.3 | 0.6 | **205.4** |
| Deck-1 | classical | 44.1 | 36.0 | 10.6 | 15.8 | 21.4 | 36.9 | 1.3 | **166.1** |
| Deck-1 | hand-assisted | 10.9 | 86.2 | 16.1 | 18.8 | 27.8 | 31.1 | 1.2 | **206.4** |
| Deck-2 | classical | 12.6 | 13.5 | 2.8 | 4.1 | 3.7 | 11.9 | 0.2 | **48.8** |
| Deck-2 | hand-assisted | 2.9 | 32.9 | 4.7 | 4.9 | 3.6 | 18.9 | 0.2 | **79.2** |

How to read these numbers:
- **Stage times overlap.** FFmpeg decodes on background threads while detection runs. The two strategies also run in different environments (OpenCV 5.0 vs 4.8). Compare totals.
- **The hand-assisted strategy is 24–62% slower and doesn't improve correctness here:**
  - Cabinet-1: same 4 measurements.
  - Countertop-1: 3 validated; the third is a doubtful short-tape event.
  - Deck-1: 4 validated against 6.
  - Deck-2: 2 validated; one I can't confirm, the other has the same flaw as the classical run.

### Classical vs AI-assisted (algorithm comparison, separate from hardware)

| Video | Classical: tape samples / validated / unconfirmed | Hand-assisted: tape samples / validated / unconfirmed |
|---|---|---|
| Cabinet-1 | 253 of 327 / 4 / 0 | 262 of 327 / 4 / 0 |
| Countertop-1 | 439 of 966 / 2 / 6 | 449 of 966 / 3 / 4 |
| Deck-1 | 386 of 749 / 6 / 10 | 405 of 749 / 4 / 9 |
| Deck-2 | 61 of 199 / 1 / 2 | 80 of 199 / 2 / 0 |

### CPU vs CUDA preprocessing (`cvbench`, measured: `results/gpu_system_python.json`)

This is resize, grayscale and Gaussian blur on synthetic frames, not the measurement pipeline. It was run with JetPack's system PyTorch 2.1 / CUDA 12.2, and the CPU path is OpenCV 4.8.

| Input, batch | CPU median ms | GPU median ms | End-to-end speedup | Stage speedup (resize / gray / blur) |
|---|---:|---:|---:|---|
| 480p, 1 | 3.17 | 5.00 | 0.63× | 3.4 / 0.6 / 0.1 |
| 480p, 8 | 25.30 | 15.82 | 1.60× | 12.2 / 4.2 / 0.2 |
| 720p, 8 | 98.59 | 40.23 | 2.45× | 19.1 / 6.6 / 0.3 |
| 1080p, 1 | 20.48 | 14.27 | 1.44× | 8.2 / 4.2 / 0.2 |
| 1080p, 8 | 218.92 | 88.55 | 2.47× | 19.2 / 8.3 / 0.3 |

All 9 cases are in the JSON. A second run in `.venv-jetson` (`results/gpu_venv_jetson.json`, with `psutil`) agreed:
- **Parity:** all 9 cases were within tolerance.
- **Speed:** 0.83× at 480p batch 1, up to 2.51× at 1080p batch 8.
- **Variation:** single small inputs vary most between runs.

Key findings:
- **Parity:** every case is within the 2×10⁻⁵ tolerance (maximum difference 1.8×10⁻⁷).
- **Small inputs:** the GPU loses on small single frames, because of overhead.
- **Batched HD:** it wins about 2.4×.
- **Blur:** the blur, as two PyTorch convolutions, is 3–10× slower than OpenCV's blur on the CPU.

### GPU compatibility on this Jetson

| Item | Finding |
|---|---|
| Platform | JetPack 6 (L4T R36.3) ships CUDA 12.2 |
| Cause of the mismatch | The `.venv` PyTorch 2.15 is a CUDA 13.2 build, so it fails with "driver too old" (`results/cuda-diagnostics.log`) |
| Already working | The system `python3` has NVIDIA's JetPack PyTorch 2.1.0 (CUDA 12.2). `cvbench` runs there even without `psutil`, using standard-library fallbacks |
| Isolated GPU environment: what was done | `.venv-jetson` created with `python3 -m venv --system-site-packages --without-pip` (plain `venv` fails: JetPack lacks `ensurepip`). It inherits, from system paths: PyTorch 2.1.0 (`/usr/local/lib/python3.10/dist-packages`), OpenCV 4.8.0 (`/usr/lib/python3.10/dist-packages`), NumPy 1.26.4 (`~/.local`) and `jetson_inference`/`jetson_utils` (`~/jetson-inference`). Only `pytest` and `psutil` (plus their small dependencies) were installed, into the venv |
| Isolated GPU environment: verified | <ul><li>CUDA available on the Orin.</li><li>A 2048×2048 fp32 matrix multiply ran on `cuda:0`: 66.6 ms/iteration (258 GFLOP/s), relative error 1.4×10⁻⁶ against a float64 CPU reference.</li><li>TensorRT 8.6.2 hand-pose inference ran on the GPU at 18.0 ms/frame on a real frame.</li><li>`cvbench`: all 9 cases within tolerance.</li><li>Full test suite: 34 passed, 1 opt-in skip.</li></ul> |
| Not available | TensorRT's own Python module (`import tensorrt`) isn't installed; `jetson_inference` reaches TensorRT itself. OpenCV 4.8 has no CUDA module. `pip check` reports `pynacl` missing `cffi` in the plain system Python too, so it predates this environment |
| Alternative | NVIDIA's `l4t-pytorch` container for r36 (`~/jetson-containers`). It needs the user added to the `docker` group, which wasn't done |
| Changed? | Nothing in drivers, JetPack, system packages or `.venv` was modified |

## Limitations

- **Cached TensorRT engine:** the hand-pose engine cached under `~/jetson-inference/data/networks` triggers a TensorRT warning that the plan was built for a different device model. Inference ran correctly on the GPU. Rebuilding the engine (delete the `.engine` file and let jetson-inference rebuild it) would remove the warning, but that changes a shared model folder, so it wasn't done.

- **Ten videos, no ground truth.** Correctness is a visual judgement. The thresholds were set on the first four videos and don't generalise well to the other six: those six gave 1 validated measurement out of 5 candidate events, and 2 of them (Door-1, Large Window) gave no event at all.
- **Blade-less measuring.** If the device is used without extending its blade (Large Window), there is nothing for the blade detector to find.
- **Labels missing on three videos.** Deck-1, Deck-2 and Kitchen Sink would need a rerun to get width/height labels.
- **Hidden hook ends.** A blade end hidden behind over-exposed fingers looks like a white surface, so it can be taken for the hook (Deck-1 #2, #4).
- **Colour-based hand test.** It can mistake beige wood next to a hook for skin. That caused the Countertop-1 regression. Hand keypoints from the hand-pose model would be the principled replacement.
- **Downward-pointing tapes.** Strongly foreshortened tapes, pointing away from the camera and held in hand, are often not detected at all.
- **Long measurements.** Hooks seen many seconds and large camera moves away from the housing are often not mappable, so those events stay `uncertain`.
- **Scene images aren't truly clear views yet.** Lens distortion isn't modelled. Truly clear frames registered unreliably, so the accepted views are nearby frames.
- **Endpoint B** is where the blade enters the housing, not the back of the housing. On vertical deck shots it can land on the housing body (Deck-2).
- **Browser coverage.** The original evaluation verified Range serving and H.264 timestamps without browser playback. During refinement, WebKitGTK under Xvfb loaded all ten proxies to decoded video state and rendered result galleries. Safari and other browsers still require separate testing.

## Alternatives considered

| Option | Outcome |
|---|---|
| Colour (yellow) blade mask | Rejected after seeing the footage: the blade is black with yellow numbers. |
| SIFT / ORB matching for whole registration chains | 0–17 inliers across 2–4 s; now used only to bridge failed tracking links. |
| Whole-frame or very local (77 px) point motion | Whole-frame fits followed the most textured depth layer. Too few features at 77 px. Replaced by a corridor around the measured segment. |
| Wider foreground mask for registration | Removed so many features that every link failed; rejected. |
| Round-trip check alone | Passed a consistently wrong mapping (railing onto beach); an appearance check was added. |
| "Hand at an end ⇒ housing" rule | All 5 firings were wrong (hands press the hook); replaced by a small confidence boost. |
| Pretrained COCO detector (YOLOv8n) for the housing | Script in `tools/experiment_yolo_housing.py`. **Not run:** the downloaded `.pt` weights are a pickle (they can execute code when loaded), and loading them was blocked pending approval. An ONNX export would be the safer route. No result is claimed. |
| Open-vocabulary detectors (YOLO-World, OWL-ViT) | Would need new Python packages (CLIP / transformers); not installed without approval. |
| Training a detector from scratch | Not done, as instructed, and with no labels. |

## AI usage and tuning disclosure

- **In the pipeline:** NVIDIA's pretrained Pose-ResNet18-Hand (jetson-inference / TensorRT), used as-is in the optional hand-assisted strategy. No other learned model is used. YOLOv8n was prepared but not run (see above).
- **Original implementation:** the original code, tests and README were written with Claude Code (an AI coding assistant) working on the Jetson. It inspected the footage frame by frame, ran every experiment and timing above, and checked every output image. Those visual judgements should be reviewed by a person.

This refinement pass used OpenAI Codex for repository review, code organization, error handling,
documentation and regression verification. It did not inspect footage to relabel results, tune
thresholds, train models or manually change annotations. The visual judgements above are retained
from the original evaluation, not independently relabelled ground truth.

## With two more weeks

1. **Ground truth.** A small hand-checked endpoint set on held-out videos, to report pixel error and precision/recall instead of visual checks.
2. **Hand-keypoint occlusion.** Use the hand model's keypoints, rather than colour, to decide whether a blade end is hidden. This fixes the wood-as-skin regression and the over-exposed-finger hooks.
3. **Lens self-calibration from the blade,** which is physically straight. Registering undistorted points would make clear-view scene images feasible.
4. **A learned blade/hook/housing detector,** fine-tuned on frames auto-labelled by the classical detector (high-confidence only), safely exported (ONNX/TensorRT). Compare it with the classical strategy on held-out videos.
5. **Speed.** Hardware HEVC decoding (NVDEC via GStreamer), fewer full-resolution JPEG caches, and detection only around tape episodes.
6. **Read the length (stretch goal).** OCR of the housing's LCD.

## Final repository verification

The concise README links the full architecture, recorded stack/setup, measured CPU/CUDA
comparison, AI usage disclosure and limitations. The final index preserves ten classical result
JSONs with 12 algorithm-validated measurements; the original 14-entry index and historical
benchmark/evaluation files are retained. Raw videos and browser proxies, model weights/runtime
engines, virtual environments, caches and temporary files remain ignored. JSON/JPEG evaluation
evidence stays tracked; ignored candidate diagnostics are local artifacts, with selected important
examples preserved under `results/accuracy-final/`.

This verification uses OpenAI Codex for review, documentation consistency checks, artifact/hash
checks, existing automated tests and local commits. It performs no further accuracy development,
manual annotation correction, model training, dependency/environment modification or video
reprocessing. Requirements remain unpinned; recorded software versions and source hashes describe
the evaluated environments, not a cross-platform reproducibility guarantee. No push is performed.
