// Package mathutil provides mathematical utilities for object detection.
package mathutil

import (
	"sort"

	"github.com/mhey01/qreader/internal/geometry"
)

// ScoredBox represents a bounding box with an associated score.
// Used as input for Non-Maximum Suppression.
type ScoredBox struct {
	Box   geometry.BBox
	Score float32
	Index int // Original index, preserved through NMS
}

// NMS performs Non-Maximum Suppression on a list of scored bounding boxes.
//
// The algorithm:
// 1. Sort boxes by score (descending)
// 2. Select the highest scoring box
// 3. Remove all boxes with IoU > threshold with the selected box
// 4. Repeat until no boxes remain or maxDetections is reached
//
// Parameters:
//   - boxes: Input boxes with scores
//   - iouThreshold: IoU threshold for suppression (0.3 is typical for QRdet)
//   - maxDetections: Maximum number of detections to keep (0 = no limit)
//
// Returns indices of the kept boxes in the original input order.
func NMS(boxes []ScoredBox, iouThreshold float32, maxDetections int) []int {
	if len(boxes) == 0 {
		return nil
	}

	// Create indices sorted by score (descending)
	indices := make([]int, len(boxes))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return boxes[indices[i]].Score > boxes[indices[j]].Score
	})

	// Track which boxes are suppressed
	suppressed := make([]bool, len(boxes))

	// Collect kept indices
	kept := make([]int, 0, min(len(boxes), maxDetections))

	for _, idx := range indices {
		if suppressed[idx] {
			continue
		}

		// Keep this box
		kept = append(kept, boxes[idx].Index)

		// Check max detections limit
		if maxDetections > 0 && len(kept) >= maxDetections {
			break
		}

		// Suppress overlapping boxes
		for _, otherIdx := range indices {
			if suppressed[otherIdx] || otherIdx == idx {
				continue
			}

			iou := boxes[idx].Box.IoU(boxes[otherIdx].Box)
			if iou > iouThreshold {
				suppressed[otherIdx] = true
			}
		}
	}

	return kept
}

// SoftNMS performs Soft Non-Maximum Suppression.
// Instead of completely removing overlapping boxes, it reduces their scores.
//
// Parameters:
//   - boxes: Input boxes with scores (modified in place)
//   - iouThreshold: IoU threshold for score reduction
//   - sigma: Gaussian sigma for score decay (default 0.5)
//   - scoreThreshold: Minimum score to keep a box
//   - maxDetections: Maximum number of detections to keep
//
// Returns indices of the kept boxes.
func SoftNMS(boxes []ScoredBox, iouThreshold, sigma, scoreThreshold float32, maxDetections int) []int {
	if len(boxes) == 0 {
		return nil
	}

	// Work with a copy of scores
	scores := make([]float32, len(boxes))
	for i, b := range boxes {
		scores[i] = b.Score
	}

	kept := make([]int, 0, min(len(boxes), maxDetections))

	for len(kept) < maxDetections || maxDetections == 0 {
		// Find max score
		maxIdx := -1
		maxScore := float32(0)
		for i, score := range scores {
			if score > maxScore {
				maxScore = score
				maxIdx = i
			}
		}

		if maxIdx == -1 || maxScore < scoreThreshold {
			break
		}

		// Keep this box
		kept = append(kept, boxes[maxIdx].Index)
		scores[maxIdx] = 0 // Mark as processed

		// Decay overlapping scores
		for i := range boxes {
			if scores[i] == 0 {
				continue
			}

			iou := boxes[maxIdx].Box.IoU(boxes[i].Box)
			if iou > iouThreshold {
				// Gaussian decay
				weight := gaussianDecay(iou, sigma)
				scores[i] *= weight
			}
		}
	}

	return kept
}

// gaussianDecay calculates the Gaussian weight for soft NMS.
func gaussianDecay(iou, sigma float32) float32 {
	return exp32(-iou * iou / sigma)
}

// exp32 computes exp(x) for float32.
func exp32(x float32) float32 {
	// Use a simple approximation for small values
	// For better accuracy, use math.Exp with conversion
	if x > 0 {
		return 1
	}
	if x < -10 {
		return 0
	}

	// Taylor series approximation: e^x ≈ 1 + x + x²/2 + x³/6 + x⁴/24
	x2 := x * x
	x3 := x2 * x
	x4 := x3 * x
	return 1 + x + x2/2 + x3/6 + x4/24
}

// Sigmoid applies the sigmoid function: 1 / (1 + exp(-x))
func Sigmoid(x float32) float32 {
	if x >= 0 {
		z := exp32(-x)
		return 1 / (1 + z)
	}
	z := exp32(x)
	return z / (1 + z)
}

// SigmoidSlice applies sigmoid to each element of a slice in place.
func SigmoidSlice(data []float32) {
	for i := range data {
		data[i] = Sigmoid(data[i])
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
