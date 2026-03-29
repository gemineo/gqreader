package qrdet

import (
	"fmt"
	"image"
	"image/color"

	"gocv.io/x/gocv"
)

// letterboxInfo stores information about the letterbox transformation
// applied to the input image. This is needed to transform coordinates
// back to the original image space.
type letterboxInfo struct {
	// Original image dimensions
	originalWidth  int
	originalHeight int

	// Scale factor applied
	scale float32

	// Padding applied (in pixels on the resized image)
	padLeft int
	padTop  int

	// Target size (model input size)
	targetSize int
}

// preprocessImage prepares an image for YOLOv8 inference.
// It performs:
// 1. Letterbox resize to maintain aspect ratio
// 2. BGR to RGB conversion
// 3. Normalization to [0, 1] range
// 4. HWC to CHW transposition
//
// Returns the preprocessed data in CHW format and letterbox info for coordinate transformation.
func preprocessImage(img gocv.Mat, targetSize int) ([]float32, *letterboxInfo, error) {
	if img.Empty() {
		return nil, nil, fmt.Errorf("input image is empty")
	}

	originalHeight := img.Rows()
	originalWidth := img.Cols()

	// Calculate scale factor to fit image in target size while maintaining aspect ratio
	scaleW := float32(targetSize) / float32(originalWidth)
	scaleH := float32(targetSize) / float32(originalHeight)
	scale := min32f(scaleW, scaleH)

	// Calculate new dimensions
	newWidth := int(float32(originalWidth) * scale)
	newHeight := int(float32(originalHeight) * scale)

	// Resize the image
	resized := gocv.NewMat()
	defer resized.Close()
	gocv.Resize(img, &resized, image.Point{X: newWidth, Y: newHeight}, 0, 0, gocv.InterpolationLinear)

	// Calculate padding to center the image
	padLeft := (targetSize - newWidth) / 2
	padTop := (targetSize - newHeight) / 2
	padRight := targetSize - newWidth - padLeft
	padBottom := targetSize - newHeight - padTop

	// Create letterboxed image with gray padding (114, 114, 114 is YOLO default)
	letterboxed := gocv.NewMat()
	defer letterboxed.Close()
	grayPadding := color.RGBA{R: 114, G: 114, B: 114, A: 0}
	gocv.CopyMakeBorder(resized, &letterboxed, padTop, padBottom, padLeft, padRight,
		gocv.BorderConstant, grayPadding)

	// Convert BGR to RGB
	rgb := gocv.NewMat()
	defer rgb.Close()
	gocv.CvtColor(letterboxed, &rgb, gocv.ColorBGRToRGB)

	// Convert to float32 in-place, then normalize to [0, 1].
	floatImg := gocv.NewMat()
	defer floatImg.Close()
	rgb.ConvertTo(&floatImg, gocv.MatTypeCV32FC3)
	floatImg.DivideFloat(255.0)

	// Rearrange from HWC (OpenCV) to CHW (ONNX).
	data, err := matToCHW(floatImg)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to convert image to CHW: %w", err)
	}

	info := &letterboxInfo{
		originalWidth:  originalWidth,
		originalHeight: originalHeight,
		scale:          scale,
		padLeft:        padLeft,
		padTop:         padTop,
		targetSize:     targetSize,
	}

	return data, info, nil
}

// matToCHW converts a gocv.Mat in HWC format to a CHW float32 slice.
// Input: Mat with shape (H, W, C) in float32
// Output: []float32 with layout [C][H][W]
func matToCHW(img gocv.Mat) ([]float32, error) {
	if img.Empty() {
		return nil, fmt.Errorf("input image is empty")
	}

	height := img.Rows()
	width := img.Cols()
	channels := img.Channels()

	if channels != 3 {
		return nil, fmt.Errorf("expected 3 channels, got %d", channels)
	}

	// Pre-allocate output: CHW layout
	data := make([]float32, channels*height*width)

	// Split channels
	channelMats := gocv.Split(img)
	defer func() {
		for _, ch := range channelMats {
			ch.Close()
		}
	}()

	// Copy each channel to the output
	for c := 0; c < channels; c++ {
		channelData, err := channelMats[c].DataPtrFloat32()
		if err != nil {
			return nil, fmt.Errorf("failed to get channel %d data: %w", c, err)
		}

		offset := c * height * width
		copy(data[offset:offset+height*width], channelData)
	}

	return data, nil
}

// transformCoordinates converts coordinates from model space back to original image space.
// This reverses the letterbox transformation applied during preprocessing.
func (info *letterboxInfo) transformCoordinates(x, y float32) (float32, float32) {
	// Remove padding offset
	x = x - float32(info.padLeft)
	y = y - float32(info.padTop)

	// Reverse scale
	x = x / info.scale
	y = y / info.scale

	// Clip to image bounds
	x = clamp32f(x, 0, float32(info.originalWidth))
	y = clamp32f(y, 0, float32(info.originalHeight))

	return x, y
}

// transformBBox transforms a bounding box from model space to original image space.
func (info *letterboxInfo) transformBBox(x1, y1, x2, y2 float32) (float32, float32, float32, float32) {
	x1, y1 = info.transformCoordinates(x1, y1)
	x2, y2 = info.transformCoordinates(x2, y2)
	return x1, y1, x2, y2
}

// transformPolygon transforms a polygon from model space to original image space.
func (info *letterboxInfo) transformPolygon(polygon [][2]float32) [][2]float32 {
	result := make([][2]float32, len(polygon))
	for i, pt := range polygon {
		result[i][0], result[i][1] = info.transformCoordinates(pt[0], pt[1])
	}
	return result
}

// Helper functions

func min32f(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func clamp32f(v, minVal, maxVal float32) float32 {
	if v < minVal {
		return minVal
	}
	if v > maxVal {
		return maxVal
	}
	return v
}
