# Golden reference files

Reference outputs of the Go implementation on Dataset-QR-2020 (99 images,
595 QR codes), used as an oracle for ports to other languages. They are
language-agnostic JSON and do not require this module or OpenCV to consume.

Regenerate with (requires OpenCV 4, see `Taskfile.yml`):

```bash
task golden
```

Output is deterministic: two runs differ only in `summary.json#generated_at`.

## Files

- `summary.json`: toolchain versions, model SHA-256, configuration,
  aggregate accuracy and failed images.
- `<image-stem>.json`: one file per annotated image.

## Per-image schema (`schema_version: 1`)

| Field | Description |
|-------|-------------|
| `image`, `width`, `height` | Source image file name and pixel size |
| `expected` | Annotated QR contents (multiset; duplicates are meaningful) |
| `threshold` | Matches required to pass (1–5 expected: 1; 6+: ceil(30%)) |
| `matched_count`, `passed` | Multiset matches of decoded results against `expected` |
| `detections[]` | Raw detector output (stage 1, before decoding) |
| `results[]` | End-to-end pipeline output, one entry per deduplicated detection |

`detections[]` / `results[].detection` (all coordinates in source-image pixels):

| Field | Description |
|-------|-------------|
| `confidence` | YOLO confidence after NMS |
| `bbox_xyxy` | `[x1, y1, x2, y2]` |
| `quad_xy` | Fitted quadrilateral, clockwise from top-left |
| `padded_quad_xy` | Expanded quadrilateral enclosing the polygon |
| `polygon_xy` | Segmentation polygon |

`results[]`:

| Field | Description |
|-------|-------------|
| `decoded` | Whether any decoder succeeded |
| `content` | Decoded payload (invalid UTF-8 replaced by U+FFFD) |
| `content_hex` | Raw payload bytes, hex-encoded (authoritative) |
| `content_valid_utf8` | Whether the raw payload is valid UTF-8 |
| `strategy` | Winning strategy, `crop/scale/flavor` (e.g. `bbox/scale_2.00/inversion`) |
| `matched_annotation` | Whether this result consumed an `expected` entry |

## Comparison guidance

- Detections depend on floating-point inference: compare boxes and quads with
  a tolerance (e.g. IoU ≥ 0.95), not exact equality.
- `strategy` documents which path succeeded in Go; a port may legitimately
  succeed through a different strategy.
- The primary acceptance criterion is `passed` per image and the aggregate
  98/99 images, 560/595 QR codes (only QR3q fails).
