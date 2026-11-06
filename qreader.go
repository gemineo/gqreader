package gqreader

import (
	"fmt"
	"sync"

	"gocv.io/x/gocv"

	"github.com/gemineo/gqreader/qrdecode"
	"github.com/gemineo/gqreader/qrdet"
)

// QReader performs end-to-end QR code detection and decoding.
// It combines YOLOv8-based detection with a brute-force decode pipeline.
//
// Create with New() or NewFromFile(), use DetectAndDecode() or DetectAndDecodeFile(),
// and always call Close() when done.
type QReader struct {
	detector *qrdet.Detector
	decoder  qrdecode.Decoder
	config   Config
}

// New creates a QReader using the embedded ONNX model.
// Requires the embed_model build tag.
func New(config Config) (*QReader, error) {
	detector, err := qrdet.NewDetector(config.DetectionConfig)
	if err != nil {
		return nil, fmt.Errorf("creating detector: %w", err)
	}

	return newQReader(detector, config)
}

// NewFromFile creates a QReader using an ONNX model loaded from the given path.
func NewFromFile(modelPath string, config Config) (*QReader, error) {
	detector, err := qrdet.NewDetectorFromFile(modelPath, config.DetectionConfig)
	if err != nil {
		return nil, fmt.Errorf("creating detector from %s: %w", modelPath, err)
	}

	return newQReader(detector, config)
}

func newQReader(detector *qrdet.Detector, config Config) (*QReader, error) {
	factory := config.DecoderFactory
	if factory == nil {
		factory = func() (qrdecode.Decoder, error) { return qrdecode.NewOpenCVDecoder() }
	}

	decoder, err := factory()
	if err != nil {
		detector.Close()
		return nil, fmt.Errorf("creating decoder: %w", err)
	}

	return &QReader{
		detector: detector,
		decoder:  decoder,
		config:   config,
	}, nil
}

// DetectAndDecode detects and decodes QR codes from an ImageSource.
// Returns results for all detected QR codes, whether decoded or not.
func (q *QReader) DetectAndDecode(img *qrdet.ImageSource) ([]QRResult, error) {
	detections, err := q.detector.Detect(img)
	if err != nil {
		return nil, fmt.Errorf("detection: %w", err)
	}

	if len(detections) == 0 {
		return nil, nil
	}

	mat := img.MatRef()
	return q.decodeAll(mat, detections), nil
}

// DetectAndDecodeFile detects and decodes QR codes from an image file.
func (q *QReader) DetectAndDecodeFile(path string) ([]QRResult, error) {
	img, err := qrdet.NewImageFromFile(path)
	if err != nil {
		return nil, fmt.Errorf("loading image: %w", err)
	}
	defer img.Close()

	return q.DetectAndDecode(img)
}

// DetectAndDecodeMat detects and decodes QR codes from a gocv.Mat.
// The Mat is not modified or closed.
func (q *QReader) DetectAndDecodeMat(mat gocv.Mat) ([]QRResult, error) {
	img := qrdet.NewImageFromMat(mat)
	return q.DetectAndDecode(img)
}

// Close releases all resources held by the QReader.
func (q *QReader) Close() error {
	var errs []error
	if q.decoder != nil {
		if err := q.decoder.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing decoder: %w", err))
		}
	}
	if q.detector != nil {
		if err := q.detector.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing detector: %w", err))
		}
	}
	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}

// decodeAll processes all detections, optionally in parallel.
func (q *QReader) decodeAll(img gocv.Mat, detections []qrdet.Detection) []QRResult {
	if len(detections) == 1 || q.config.MaxWorkers == 1 {
		// Sequential processing
		results := make([]QRResult, len(detections))
		for i, det := range detections {
			results[i] = decodeDetection(img, det, q.decoder, q.config)
			if q.config.TryHarder {
				results[i] = q.retryWithFallbacks(img, det, results[i])
			}
		}
		return deduplicateResults(results)
	}

	// Parallel processing: each detection gets its own decoder to avoid contention
	results := make([]QRResult, len(detections))
	var wg sync.WaitGroup

	limit := len(detections)
	if q.config.MaxWorkers > 0 && q.config.MaxWorkers < limit {
		limit = q.config.MaxWorkers
	}
	semaphore := make(chan struct{}, limit)

	for i, det := range detections {
		wg.Add(1)
		go func(idx int, d qrdet.Detection) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			// Each goroutine uses its own decoder for thread safety
			dec, err := q.config.DecoderFactory()
			if err != nil {
				results[idx] = QRResult{Detection: d}
				return
			}
			defer dec.Close()

			result := decodeDetection(img, d, dec, q.config)
			if q.config.TryHarder {
				result = q.retryWithFallbacks(img, d, result)
			}
			results[idx] = result
		}(i, det)
	}

	wg.Wait()
	return deduplicateResults(results)
}

// retryWithFallbacks runs each fallback decoder factory in order and returns the
// first fallback result that decodes something. If no fallback succeeds, the
// original primary result (decoded or not) is returned unchanged.
//
// Fallback decoders are tried even when the primary already decoded, because the
// primary may have returned a structurally valid but mis-encoded result (e.g. ZBar
// converting ISO-8859-1 byte sequences to Shift-JIS Unicode characters). The first
// fallback decoder that produces a non-empty result wins, since later decoders in
// the chain (such as OpenCV or WeChatQR) typically have more robust encoding support.
func (q *QReader) retryWithFallbacks(img gocv.Mat, det qrdet.Detection, primary QRResult) QRResult {
	for _, factory := range q.config.FallbackDecoderFactories {
		fbDec, err := factory()
		if err != nil {
			continue
		}
		r := decodeDetection(img, det, fbDec, q.config)
		fbDec.Close()
		if r.Decoded {
			return r
		}
	}
	return primary
}

// duplicateIoU is the bounding-box overlap above which two decoded results
// with the same content are considered the same physical symbol.
const duplicateIoU = 0.3

// deduplicateResults removes duplicate decoded QR codes: results with the
// same content whose detections overlap substantially are merged, keeping
// the highest confidence one. Distinct symbols that happen to encode the same
// payload (a sheet of identical labels) are all kept, as are undecoded results.
func deduplicateResults(results []QRResult) []QRResult {
	if len(results) <= 1 {
		return results
	}

	deduped := make([]QRResult, 0, len(results))
	for _, r := range results {
		if !r.Decoded {
			deduped = append(deduped, r)
			continue
		}
		merged := false
		for i := range deduped {
			d := &deduped[i]
			if !d.Decoded || d.Content != r.Content {
				continue
			}
			if bboxIoU(d.Detection.BBox(), r.Detection.BBox()) <= duplicateIoU {
				continue
			}
			if r.Detection.Confidence > d.Detection.Confidence {
				*d = r
			}
			merged = true
			break
		}
		if !merged {
			deduped = append(deduped, r)
		}
	}
	return deduped
}

// bboxIoU returns the intersection-over-union of two [x1, y1, x2, y2] boxes.
func bboxIoU(a, b [4]float32) float32 {
	ix := max(0, min(a[2], b[2])-max(a[0], b[0]))
	iy := max(0, min(a[3], b[3])-max(a[1], b[1]))
	inter := ix * iy
	areaA := max(0, a[2]-a[0]) * max(0, a[3]-a[1])
	areaB := max(0, b[2]-b[0]) * max(0, b[3]-b[1])
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}
