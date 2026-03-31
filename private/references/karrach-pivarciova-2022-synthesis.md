# Synthesis: Karrach & Pivarčiová (2022)

**Full citation**: Karrach, L. & Pivarčiová, E. (2022). *Location and Recognition of Data
Matrix and QR Codes in Images*. Applied Sciences, vol. 8, pp. 22–1.
File: `private/references/vol-8-022-1_location_and_recognition.pdf`

---

## 1. Publication Overview

The paper describes a complete pipeline for locating and decoding Data Matrix and QR codes
from images, benchmarked on the same **Dataset-QR-2020** (99 JPEG images, 595 QR codes)
that gqreader uses for integration testing.

**Their results on Dataset-QR-2020 (Table 10)**:

| Tool | QR codes decoded | Notes |
|------|-----------------|-------|
| ZBar standalone | 505/595 (84.9%) | No preprocessing, no localization |
| Proposed method | 592/595 (99.5%) | Full pipeline incl. perspective + per-region threshold |
| **gqreader (CoreML+ZBar)** | **558/595 (93.8%)** | 96/99 images (97.0%) — as of 2026-03-31 |
| Python QReader | 581/595 (97.6%) | 96/99 images (97.0%) |

gqreader sits between ZBar standalone and the Karrach system. The gap to their 99.5% is
almost entirely in QR4/QR6 (dense industrial images, 180–250 small QR codes per image)
where their per-region adaptive threshold excels. For single-QR images our accuracy matches
or exceeds theirs.

---

## 2. Key Techniques and gqreader Applicability

### 2.1 Adaptive per-region thresholding — §3.2.9 · LOW PRIORITY

After perspective correction, the QR region is divided into a 3×3 grid of tiles, each
thresholded independently using its own average intensity, then bilinearly interpolated
across tile boundaries. This compensates for uneven illumination within the ROI.

**Window sizing rule** (§3.2.9): the adaptive threshold window must be ≥ 5× the expected
module size. Karrach uses a 35×35 px window for their dataset (≈ 5× the 7 px module size).
Alternative formulas evaluated: Niblack, Sauvola, Bradley-Roth, and a custom formula
`T(x,y) = m(x,y) − I(x,y)/k1 − s²(x,y)/k2`.

**Verdict for gqreader**: ZBar's internal adaptive binarization already handles per-region
thresholding implicitly. This is only relevant if we build our own decoder or integrate a
custom binarization step before passing the image to ZBar. **No action needed.**

### 2.2 Cylindrical distortion — §3.2.7 + p.62 · CONTEXT ONLY

The paper explicitly states (§3.2.7, p.62): perspective transformation **cannot correct
cylindrical distortion**. Cylindrical surfaces (bottles, cans) cause barrel-like warping
that an affine/projective transform cannot undo. The paper proposes detecting cylindrical
distortion by analyzing edge orientations in three horizontal thirds of the image, then
applying a reverse-cylinder model — this is listed as **future work** (§4.3).

**Verdict for gqreader**: This validates the approach applied for QR3c (photo of QR on a
bottle). The fix (`grayscale` strategy — raw gray → ZBar's own adaptive binarization) is
the best achievable without a custom cylindrical undistortion step. The paper confirms no
standard preprocessing technique can recover this case reliably. **QR3c is at the limit
of what the current architecture can achieve.**

### 2.3 Histogram equalization / CLAHE — §4.3 + Appendix A · MEDIUM PRIORITY

For images with poor contrast, spreading the histogram before thresholding significantly
improves binarization quality. Two variants:

- **Full histogram equalization** (`gocv.EqualizeHist`): maps pixel values so the
  cumulative histogram is approximately uniform. Can amplify noise in low-information regions.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization, `gocv.NewCLAHE`): operates
  on local tiles with a clip limit that prevents over-amplification. Recommended for images
  with local illumination gradients.

**Existing strategy**: `HighContrastStrategy()` in `imgprep/presets.go` already implements
CLAHE. It is used in `TestAllPresetStrategies` but is **not wired into**
`QRDecodeStrategies()` or `QRDecodeFallbackStrategies()`.

**Proposal**: Add as a Phase 2 fallback. Likely impact on QR3e (low contrast + perspective
distortion). Implementation: one-line addition to `QRDecodeFallbackStrategies()`.

### 2.4 Contrast stretching — Appendix A · MEDIUM PRIORITY

Linear normalization that clips the extreme percentiles and rescales the remainder to [0, 255]:

```
g(x,y) = (f(x,y) − a) × 255 / (b − a)
```

where `a` = 5th percentile of pixel values, `b` = 95th percentile. Gentler than full
histogram equalization; avoids amplifying noise from extreme outlier pixels.

**Existing strategy**: Not present in `imgprep/`. Could be implemented as a new strategy
step in `imgprep/strategies.go` using `gocv.Normalize` with `NORM_MINMAX` (approximation)
or via direct percentile computation.

**Proposal**: Implement `ContrastStretch` and wire into `QRDecodeFallbackStrategies()`.
Potential impact on QR3q (compressed URL, small modules, marginal contrast). Cost: ~2h.

### 2.5 Morphological closing — §4.3 · LOW PRIORITY

After binarization, apply morphological closing (dilation followed by erosion) with a 3×3
structuring element. This bridges small gaps in the finder pattern lines caused by noise
or print degradation — critical for the QR6 category (damaged finder patterns).

`gocv.MorphologyEx(src, &dst, gocv.MorphClose, kernel)`

**Verdict**: The current pipeline already passes all 6 QR6 images. This technique is a
defensive measure against regressions if dataset composition changes. **Low priority.**

### 2.6 Multi-scale rescaling — §4.3 · ALREADY IMPLEMENTED

The paper recommends trying 4:3 and 2:1 scale factors. gqreader's `DefaultConfig`
`ScaleFactors: []float64{1.5, 2.0, 0.5}` already covers this range. **No action needed.**

### 2.7 Fixed global threshold sweep — §4.3 · NOT APPLICABLE

Karrach sweeps thresholds T ∈ {30, 40, 50, 60, 70, 80, 90} as a robustness fallback when
adaptive thresholding fails. ZBar's internal adaptive binarization renders this redundant
for our use case — ZBar selects the threshold adaptively per scan line. **No action needed.**

---

## 3. Mapping to Remaining Failing Images

Current state (2026-03-31, master): **3 images fail** out of 99.

| Image | Failure reason | Paper insight | Status |
|-------|---------------|---------------|--------|
| QR2c | ZBar-specific regression (passes with OpenCV decoder) | None applicable | Irreducible; decoder trade-off |
| QR3e | Low contrast + perspective distortion | CLAHE / histogram equalization (§4.3) | **Fixed** — `HighContrastStrategy()` added to Phase 2 fallback (2026-03-31) |
| QR3k | Japanese URL, encoding edge case | None applicable | Irreducible; also fails Python QReader |
| QR3q | Compressed URL, small module size | Contrast stretching (Appendix A) | Still failing; `ContrastStretch` not yet implemented |

**QR3k and QR3q also fail Python QReader** — they represent hard cases where the image
quality is genuinely below the decodable threshold for any standard decoder.

---

## 4. Concrete Improvement Proposals (Ranked)

### P1 — Wire `HighContrastStrategy()` into Phase 2 fallback · ✅ Done (2026-03-31)

**Commit**: `577edcc`

`HighContrastStrategy()` added as position 2 in `QRDecodeFallbackStrategies()`.
Also added `gray_otsu_blur5` (gray → Otsu → Blur(5)) at position 1, restoring a strategy
that was present before the two-phase split but was accidentally dropped.

**Actual impact**: QR3e recovered. 95/99 → 96/99 (97.0%), tied with Python QReader.

### P2 — Implement `ContrastStretch` strategy · Medium

**File**: `imgprep/strategies.go` (new type `ContrastStretch`)

```go
// ContrastStretch linearly normalizes pixel values to [0, 255] using
// the image's actual min/max (or percentile clipping for robustness).
type ContrastStretch struct{}
```

Internal implementation: `gocv.Normalize(src, &dst, 0, 255, gocv.NormMinMax)`.
This is a min-max approximation; full percentile clipping requires sorting pixel values.

Wire into `QRDecodeFallbackStrategies()` after `HighContrastStrategy`.

**Expected impact**: May assist QR3q. Independently useful for images with very narrow
dynamic range (e.g., slightly overexposed or underexposed crops after perspective correction).

### P3 — Morphological closing flavor · Future / Low

Add a `MorphClose{KernelSize: 3}` strategy for use in Phase 2. Only relevant if QR6-category
accuracy regresses. Defer until there is a concrete regression to fix.

---

## 5. Architecture Insights from the Paper

### Why YOLO preprocessing beats ZBar standalone

Table 10 directly confirms the value of the detection layer. ZBar standalone on raw images:
505/595 (84.9%). With localization (their method): 592/595 (99.5%). The 14.6 pp gap is
entirely attributable to cropping + perspective correction before decoding. This validates
the gqreader architecture: YOLO detection → crop → preprocess → ZBar decode.

### Why ZBar's internal binarization matters

The paper notes (§3.2.9) that for cylindrically distorted QR codes, module sizes vary
spatially — making any single global or even tile-based threshold insufficient. ZBar's
internal scan-line adaptive threshold handles this gracefully because it operates per-row
at the byte level. This is why `grayscale` (pass raw gray to ZBar, no preprocessing) fixed
QR3c: any preprocessing we apply destroys the gradient information ZBar needs.

**Rule of thumb**: For Phase 1 strategies, prefer strategies that pass minimally processed
images to ZBar. Heavy preprocessing (Otsu, blur) belongs in Phase 2 fallback, not Phase 1.

---

## 6. Summary of Actionability

| Finding | Priority | Effort | Status |
|---------|----------|--------|--------|
| Wire `HighContrastStrategy()` into Phase 2 | Medium | 15 min | ✅ Done — commit `577edcc` |
| Implement `ContrastStretch` strategy | Medium | 2h | Open — targets QR3q |
| Morphological closing | Low | 1h | Defer |
| Per-region (9-tile) thresholding | Low | N/A | ZBar handles this internally |
| Cylindrical distortion correction | None | N/A | Requires custom decoder |
| Multi-scale | — | — | Already in DefaultConfig |
| Threshold sweep | — | — | ZBar handles this internally |
