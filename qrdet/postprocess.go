package qrdet

import (
	"math"

	"gocv.io/x/gocv"

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

		// Extract polygon from mask using FindContours
		polygon := maskToPolygon(mask, raw.cx, raw.cy, raw.w, raw.h, info.targetSize)
		mask.Close()

		// Transform coordinates to original image space
		x1, y1 := info.transformCoordinates(raw.cx-raw.w/2, raw.cy-raw.h/2)
		x2, y2 := info.transformCoordinates(raw.cx+raw.w/2, raw.cy+raw.h/2)

		// Transform polygon
		transformedPolygon := info.transformPolygon(polygon)

		// Fit quadrilateral (IoU-optimized) and compute expanded version
		quad, paddedQuad := fitQuadrilateral(transformedPolygon)

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
// The mask is returned as a CV_8UC1 Mat at prototype resolution, cropped to the bbox.
// Pixels where (coeff @ proto) > 0 are set to 255, others to 0.
//
// The caller must close the returned Mat.
// An empty Mat is returned if the bbox has no valid area.
//
// Formula: mask = (coefficients @ prototypes) > 0
// Note: sigmoid(x) > 0.5 ↔ x > 0, so the sigmoid is elided.
func generateMask(
	coeffs []float32,
	prototypes []float32,
	cx, cy, w, h float32,
	inputSize int,
) gocv.Mat {
	protoSize := modelPrototypeSize // 160

	scale := float32(protoSize) / float32(inputSize)

	x1 := int(clamp32f((cx-w/2)*scale, 0, float32(protoSize-1)))
	y1 := int(clamp32f((cy-h/2)*scale, 0, float32(protoSize-1)))
	x2 := int(clamp32f((cx+w/2)*scale, 0, float32(protoSize-1)))
	y2 := int(clamp32f((cy+h/2)*scale, 0, float32(protoSize-1)))

	if x2 <= x1 || y2 <= y1 {
		return gocv.NewMat()
	}

	maskWidth := x2 - x1
	maskHeight := y2 - y1

	// Accumulate weighted prototypes into a pixel buffer.
	//
	// Loop order: channel → row → col.
	// For a given channel c, the prototype row at (y1+dy) spans contiguous floats
	// proto[c*(protoSize²) + (y1+dy)*protoSize + x1 .. x1+maskWidth].
	// The innermost dx loop therefore accesses sequential memory, which is
	// cache-friendly and allows the compiler to auto-vectorize the multiply-add.
	//
	// The original dy→dx→c order caused a stride of protoSize² (25 600 floats =
	// 100 KB) between channel accesses — a cache miss on every coefficient step.
	pixelSums := make([]float32, maskHeight*maskWidth)
	for c := 0; c < modelMaskCoefficients; c++ {
		coeff := coeffs[c]
		if coeff == 0 {
			continue
		}
		protoBase := c*protoSize*protoSize + y1*protoSize + x1
		for dy := 0; dy < maskHeight; dy++ {
			protoRow := protoBase + dy*protoSize
			pixelRow := dy * maskWidth
			for dx := 0; dx < maskWidth; dx++ {
				pixelSums[pixelRow+dx] += coeff * prototypes[protoRow+dx]
			}
		}
	}

	// Threshold: sigmoid(x) > 0.5 ↔ x > 0, so we skip the sigmoid entirely.
	mask := gocv.NewMatWithSize(maskHeight, maskWidth, gocv.MatTypeCV8UC1)
	maskPtr, _ := mask.DataPtrUint8()
	for i, v := range pixelSums {
		if v > 0 {
			maskPtr[i] = 255
		}
	}

	return mask
}

// maskToPolygon extracts contour points from a binary mask using OpenCV's FindContours.
// This matches Python's cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE) as used by
// Ultralytics' mask.xy[0] accessor for segmentation polygon extraction.
//
// Returns points in model input space (640x640).
func maskToPolygon(
	mask gocv.Mat,
	cx, cy, w, h float32,
	inputSize int,
) [][2]float32 {
	bboxFallback := func() [][2]float32 {
		halfW := w / 2
		halfH := h / 2
		return [][2]float32{
			{cx - halfW, cy - halfH},
			{cx + halfW, cy - halfH},
			{cx + halfW, cy + halfH},
			{cx - halfW, cy + halfH},
		}
	}

	if mask.Empty() {
		return bboxFallback()
	}

	// Find contours — same algorithm as Python's cv2.findContours
	contours := gocv.FindContours(mask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	if contours.Size() == 0 {
		return bboxFallback()
	}

	// Select the largest contour by area (Python's mask.xy[0] takes the primary mask contour)
	largestIdx := 0
	if contours.Size() > 1 {
		largestArea := 0.0
		for i := 0; i < contours.Size(); i++ {
			area := gocv.ContourArea(contours.At(i))
			if area > largestArea {
				largestArea = area
				largestIdx = i
			}
		}
	}

	contour := contours.At(largestIdx)
	if contour.Size() < 3 {
		return bboxFallback()
	}

	// Transform contour points from mask space (cropped bbox in prototype space)
	// back to model input space (640x640)
	protoSize := modelPrototypeSize
	scale := float32(protoSize) / float32(inputSize)
	offsetX := (cx - w/2) * scale
	offsetY := (cy - h/2) * scale
	invScale := float32(inputSize) / float32(protoSize)

	points := make([][2]float32, contour.Size())
	for i := 0; i < contour.Size(); i++ {
		p := contour.At(i)
		px := (float32(p.X) + offsetX) * invScale
		py := (float32(p.Y) + offsetY) * invScale
		points[i] = [2]float32{px, py}
	}

	return points
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
