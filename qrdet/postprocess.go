package qrdet

import (
	"math"
	"sort"

	"github.com/gemineo/gqreader/internal/geometry"
	"github.com/gemineo/gqreader/internal/mathutil"
)

// rawDetection holds intermediate detection data before final processing.
type rawDetection struct {
	// Bounding box in center-width-height format (model space)
	cx, cy, w, h float32

	// Detection confidence
	confidence float32

	// Class index (always 0 for QR detection)
	classIndex int

	// Mask coefficients for prototype multiplication
	maskCoeffs [modelMaskCoefficients]float32
}

// processOutput converts raw ONNX output tensors into Detection structs.
//
// Output format (YOLOv8-seg):
//   - output0: (1, 116, 8400) where 116 = 4 (bbox) + 80 (classes) + 32 (mask coeffs)
//   - output1: (1, 32, 160, 160) prototype masks
//
// The output0 tensor is in transposed format compared to some implementations.
// Each of the 8400 predictions contains: [cx, cy, w, h, class_scores..., mask_coeffs...]
func processOutput(
	output0 []float32,
	output1 []float32,
	info *letterboxInfo,
	config Config,
) ([]Detection, error) {
	// Parse raw detections from output0
	rawDets := parseOutput0(output0, config.ConfidenceThreshold)

	if len(rawDets) == 0 {
		return nil, nil
	}

	// Apply Non-Maximum Suppression
	scoredBoxes := make([]mathutil.ScoredBox, len(rawDets))
	for i, det := range rawDets {
		bbox := geometry.FromCenterWH(det.cx, det.cy, det.w, det.h)
		scoredBoxes[i] = mathutil.ScoredBox{
			Box:   bbox,
			Score: det.confidence,
			Index: i,
		}
	}

	keptIndices := mathutil.NMS(scoredBoxes, config.NMSIoUThreshold, config.MaxDetections)

	// Build final detections
	detections := make([]Detection, 0, len(keptIndices))
	for _, idx := range keptIndices {
		raw := rawDets[idx]

		// Generate mask from coefficients and prototypes
		mask := generateMask(raw.maskCoeffs[:], output1, raw.cx, raw.cy, raw.w, raw.h, info.targetSize)

		// Extract polygon from mask
		polygon := maskToPolygon(mask, raw.cx, raw.cy, raw.w, raw.h, info.targetSize)

		// Transform coordinates to original image space
		x1, y1 := info.transformCoordinates(raw.cx-raw.w/2, raw.cy-raw.h/2)
		x2, y2 := info.transformCoordinates(raw.cx+raw.w/2, raw.cy+raw.h/2)

		// Transform polygon
		transformedPolygon := info.transformPolygon(polygon)

		// Fit quadrilateral
		quad := fitQuadrilateral(transformedPolygon)
		paddedQuad := expandQuadrilateral(quad, transformedPolygon)

		// Build detection
		det := buildDetection(
			raw.confidence,
			x1, y1, x2, y2,
			transformedPolygon,
			quad,
			paddedQuad,
			info.originalWidth,
			info.originalHeight,
		)

		detections = append(detections, det)
	}

	return detections, nil
}

// parseOutput0 extracts raw detections from the ONNX output tensor.
// The tensor has shape (1, 37, 8400) where (for QRdet single-class):
//   - Channels 0-3: bbox (cx, cy, w, h)
//   - Channel 4: confidence score
//   - Channels 5-36: 32 mask coefficients
//   - 8400 predictions
//
// Layout: [cx, cy, w, h, confidence, mask_coeff_0, ..., mask_coeff_31]
// Note: YOLOv8 output is in transposed format with shape (batch, channels, predictions)
func parseOutput0(output0 []float32, confThreshold float32) []rawDetection {
	numPredictions := modelOutputPredictions // 8400
	numChannels := modelOutputChannels       // 37

	detections := make([]rawDetection, 0, 100) // Pre-allocate reasonable capacity

	for i := 0; i < numPredictions; i++ {
		// Extract bounding box (channels 0-3)
		cx := output0[0*numPredictions+i]
		cy := output0[1*numPredictions+i]
		w := output0[2*numPredictions+i]
		h := output0[3*numPredictions+i]

		// For QRdet single-class detection, confidence is at channel 4
		confidence := output0[4*numPredictions+i]

		// Apply confidence threshold
		if confidence < confThreshold {
			continue
		}

		// Extract mask coefficients (last 32 channels)
		var maskCoeffs [modelMaskCoefficients]float32
		for j := 0; j < modelMaskCoefficients; j++ {
			coeffIdx := (numChannels - modelMaskCoefficients + j) * numPredictions
			maskCoeffs[j] = output0[coeffIdx+i]
		}

		detections = append(detections, rawDetection{
			cx:         cx,
			cy:         cy,
			w:          w,
			h:          h,
			confidence: confidence,
			classIndex: 0,
			maskCoeffs: maskCoeffs,
		})
	}

	return detections
}

// generateMask creates a binary mask from mask coefficients and prototypes.
// The mask is generated at prototype resolution (160x160) then cropped to the bbox.
//
// Formula: mask = sigmoid(coefficients @ prototypes) > 0.5
func generateMask(
	coeffs []float32,
	prototypes []float32,
	cx, cy, w, h float32,
	inputSize int,
) [][]bool {
	protoSize := modelPrototypeSize // 160

	// Scale factor from input size to prototype size
	scale := float32(protoSize) / float32(inputSize)

	// Calculate bbox in prototype space
	x1 := int(clamp32f((cx-w/2)*scale, 0, float32(protoSize-1)))
	y1 := int(clamp32f((cy-h/2)*scale, 0, float32(protoSize-1)))
	x2 := int(clamp32f((cx+w/2)*scale, 0, float32(protoSize-1)))
	y2 := int(clamp32f((cy+h/2)*scale, 0, float32(protoSize-1)))

	// Ensure valid dimensions
	if x2 <= x1 || y2 <= y1 {
		return nil
	}

	maskWidth := x2 - x1
	maskHeight := y2 - y1

	// Create mask
	mask := make([][]bool, maskHeight)
	for i := range mask {
		mask[i] = make([]bool, maskWidth)
	}

	// Compute mask values: sum(coeffs[i] * prototypes[i][y][x]) for each pixel
	for dy := 0; dy < maskHeight; dy++ {
		py := y1 + dy
		for dx := 0; dx < maskWidth; dx++ {
			px := x1 + dx

			// Sum over all prototype channels
			var value float32
			for c := 0; c < modelMaskCoefficients; c++ {
				// prototypes layout: (1, 32, 160, 160) -> [c][y][x]
				protoIdx := c*protoSize*protoSize + py*protoSize + px
				value += coeffs[c] * prototypes[protoIdx]
			}

			// Apply sigmoid and threshold
			mask[dy][dx] = sigmoid(value) > 0.5
		}
	}

	return mask
}

// maskToPolygon extracts contour points from a binary mask.
// Returns points in model input space (640x640).
func maskToPolygon(
	mask [][]bool,
	cx, cy, w, h float32,
	inputSize int,
) [][2]float32 {
	if mask == nil || len(mask) == 0 {
		// Return bbox corners as fallback polygon
		halfW := w / 2
		halfH := h / 2
		return [][2]float32{
			{cx - halfW, cy - halfH},
			{cx + halfW, cy - halfH},
			{cx + halfW, cy + halfH},
			{cx - halfW, cy + halfH},
		}
	}

	// Extract contour points using simple edge detection
	protoSize := modelPrototypeSize
	scale := float32(protoSize) / float32(inputSize)

	// Calculate bbox offset in prototype space
	offsetX := (cx - w/2) * scale
	offsetY := (cy - h/2) * scale

	// Scale back to input space
	invScale := float32(inputSize) / float32(protoSize)

	// Simple contour extraction: find edge pixels
	contourPoints := make([][2]float32, 0, 100)

	maskHeight := len(mask)
	maskWidth := len(mask[0])

	for y := 0; y < maskHeight; y++ {
		for x := 0; x < maskWidth; x++ {
			if !mask[y][x] {
				continue
			}

			// Check if this is an edge pixel (has at least one non-mask neighbor)
			isEdge := false
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					ny, nx := y+dy, x+dx
					if ny < 0 || ny >= maskHeight || nx < 0 || nx >= maskWidth {
						isEdge = true
						break
					}
					if !mask[ny][nx] {
						isEdge = true
						break
					}
				}
				if isEdge {
					break
				}
			}

			if isEdge {
				// Convert to input space coordinates
				px := (float32(x) + offsetX) * invScale
				py := (float32(y) + offsetY) * invScale
				contourPoints = append(contourPoints, [2]float32{px, py})
			}
		}
	}

	if len(contourPoints) < 3 {
		// Return bbox corners as fallback
		halfW := w / 2
		halfH := h / 2
		return [][2]float32{
			{cx - halfW, cy - halfH},
			{cx + halfW, cy - halfH},
			{cx + halfW, cy + halfH},
			{cx - halfW, cy + halfH},
		}
	}

	// Sort points by angle from centroid (for proper polygon ordering)
	centroidX, centroidY := computeCentroid(contourPoints)
	sort.Slice(contourPoints, func(i, j int) bool {
		angleI := math.Atan2(float64(contourPoints[i][1]-centroidY), float64(contourPoints[i][0]-centroidX))
		angleJ := math.Atan2(float64(contourPoints[j][1]-centroidY), float64(contourPoints[j][0]-centroidX))
		return angleI < angleJ
	})

	return contourPoints
}

// computeCentroid calculates the centroid of a set of points.
func computeCentroid(points [][2]float32) (float32, float32) {
	var sumX, sumY float32
	for _, p := range points {
		sumX += p[0]
		sumY += p[1]
	}
	n := float32(len(points))
	return sumX / n, sumY / n
}

// buildDetection creates a Detection struct with all coordinate variants.
func buildDetection(
	confidence float32,
	x1, y1, x2, y2 float32,
	polygon [][2]float32,
	quad [4][2]float32,
	paddedQuad [4][2]float32,
	imgWidth, imgHeight int,
) Detection {
	w := x2 - x1
	h := y2 - y1
	cx := (x1 + x2) / 2
	cy := (y1 + y2) / 2

	fWidth := float32(imgWidth)
	fHeight := float32(imgHeight)

	// Normalize polygon
	polygonN := make([][2]float32, len(polygon))
	for i, p := range polygon {
		polygonN[i] = [2]float32{p[0] / fWidth, p[1] / fHeight}
	}

	// Normalize quads
	var quadN, paddedQuadN [4][2]float32
	for i := 0; i < 4; i++ {
		quadN[i] = [2]float32{quad[i][0] / fWidth, quad[i][1] / fHeight}
		paddedQuadN[i] = [2]float32{paddedQuad[i][0] / fWidth, paddedQuad[i][1] / fHeight}
	}

	return Detection{
		Confidence:    confidence,
		ImageWidth:    imgWidth,
		ImageHeight:   imgHeight,
		BBoxXYXY:      [4]float32{x1, y1, x2, y2},
		BBoxXYXYN:     [4]float32{x1 / fWidth, y1 / fHeight, x2 / fWidth, y2 / fHeight},
		CenterXY:      [2]float32{cx, cy},
		CenterXYN:     [2]float32{cx / fWidth, cy / fHeight},
		WidthHeight:   [2]float32{w, h},
		WidthHeightN:  [2]float32{w / fWidth, h / fHeight},
		PolygonXY:     polygon,
		PolygonXYN:    polygonN,
		QuadXY:        quad,
		QuadXYN:       quadN,
		PaddedQuadXY:  paddedQuad,
		PaddedQuadXYN: paddedQuadN,
	}
}

// sigmoid applies the sigmoid function: 1 / (1 + exp(-x))
func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}
