# QRdet Model Format Documentation

## ⚠️ CRITICAL: Single-Class Detection Model

This package is currently designed for **single-class QR code detection only**.

The QRdet model (v2.0) from GitHub releases has been specifically trained for QR codes and produces a fundamentally different output format compared to generic multi-class YOLOv8 models.

## Model Output Format

### Current: QRdet-s ONNX (Single-Class)

```
Input:  [batch=1, channels=3, height=640, width=640]
Output0: [batch=1, channels=37, predictions=8400]
Output1: [batch=1, channels=32, height=160, width=160]
```

#### Output0 Channel Layout (37 total)

| Channels | Purpose | Indices |
|----------|---------|---------|
| 0-3 | Bounding box (cx, cy, w, h) | 0:4 |
| 4 | Confidence score | 4 |
| 5-36 | Mask coefficients | 5:37 |

**Key Difference**: Confidence is at **channel index 4**, NOT after all classes like multi-class YOLO.

#### Output1: Segmentation Prototypes

- Shape: `[1, 32, 160, 160]`
- 32 prototype feature maps at 160×160 resolution
- Multiplied by mask coefficients to generate instance masks
- Formula: `mask = sigmoid(coefficients @ prototypes) > 0.5`

## Comparison: Standard YOLOv8-seg vs QRdet

### Standard YOLOv8-seg (Multi-Class)

```
Output0: [batch=1, channels=116, predictions=8400]
         ↑ 4 bbox + 80 COCO classes + 32 mask coefficients
```

Layout per prediction:
```
[cx, cy, w, h, class_score_0, class_score_1, ..., class_score_79, mask_coeff_0, ..., mask_coeff_31]
```

### QRdet-seg (Single-Class)

```
Output0: [batch=1, channels=37, predictions=8400]
         ↑ 4 bbox + 1 confidence + 32 mask coefficients
```

Layout per prediction:
```
[cx, cy, w, h, confidence, mask_coeff_0, ..., mask_coeff_31]
```

## Implementation Impact

### Current Code Hardcoded Assumptions

These constants in `types.go` assume single-class detection:

```go
const modelOutputChannels = 37           // 4 + 1 + 32
const qrClassIndex = 4                   // confidence at index 4
const modelMaskCoefficients = 32
const modelPrototypeSize = 160
const modelOutputPredictions = 8400
```

### Parser Function: `parseOutput0()`

Located in `postprocess.go`, this function **assumes**:
1. Confidence at channel index 4
2. Mask coefficients at channels 5-36
3. No class selection logic (single class)

## Future: Multi-Class Barcode Detection

### Required Changes for Multi-Format Support

If you want to detect QR codes, DataMatrix, Code128, I2of5, Codabar, etc., you need:

#### 1. New Model

**DO NOT reuse QRdet**. Instead:

**Option A** (Recommended): Train new YOLOv8-seg with multiple barcode classes
```bash
# Train with N=5 classes: QR, DataMatrix, Code128, I2of5, Codabar
# Result: Output0 will have shape [1, 4+5+32, 8400] = [1, 41, 8400]
```

**Option B**: Use separate single-class models per format
- Less efficient but simpler initially
- Requires model routing logic

#### 2. Code Changes Required

| Component | Change | Effort |
|-----------|--------|--------|
| `types.go` | Parameterize `modelOutputChannels`, `modelNumClasses` | 30 min |
| `parseOutput0()` | Handle variable class counts dynamically | 2-4 hours |
| `Detection` struct | Add `DetectedClass string` field | 1 hour |
| `postprocess.go` | Class-aware coordinate extraction | 2-4 hours |
| `detector.go` | Class mapping/configuration | 1-2 hours |
| Tests | Update all test cases | 2-4 hours |

**Total Estimate**: 1-2 days

#### 3. Output Format for N Classes

```
Output0: [1, 4 + N + 32, 8400]

Example with 5 classes:
Output0: [1, 41, 8400]
         ↑ 4 bbox + 5 class scores + 32 masks

Layout per prediction:
[cx, cy, w, h, class_0_score, class_1_score, ..., class_4_score, mask_coeff_0, ..., mask_coeff_31]
```

## Code Locations

- **Model format constants**: `pkg/qrdet/types.go` (lines 136-154)
- **Output parsing**: `pkg/qrdet/postprocess.go` (function `parseOutput0()`)
- **Session management**: `pkg/qrdet/session.go` (tensor shapes)
- **Detection structure**: `pkg/qrdet/types.go` (struct `Detection`)

## References

- QRdet GitHub: https://github.com/Eric-Canas/qrdet
- QRdet releases: https://github.com/Eric-Canas/qrdet/releases/tag/v2.0_release
- YOLOv8 documentation: https://docs.ultralytics.com/
- ONNX model format: https://onnx.ai/

## Important Notes

1. **Do not assume standard YOLO format** when modifying inference code
2. **Document any model changes** in this file immediately
3. **Test with actual model output** before committing changes
4. **Keep model format versioned** if supporting multiple formats
5. **Consider API versioning** to handle format changes gracefully

---

**Last Updated**: 2025-12-20 (QRdet-s v2.0)
