package qrdet

import (
	ort "github.com/yalue/onnxruntime_go"
)

// ExecutionProvider selects the hardware backend for ONNX Runtime inference.
type ExecutionProvider int

const (
	// ProviderCPU uses the default multi-threaded CPU execution provider.
	ProviderCPU ExecutionProvider = iota

	// ProviderCoreML uses Apple's CoreML framework (macOS only).
	// Offloads computation to the GPU and/or Neural Engine when available.
	// Falls back to CPU automatically if CoreML is unavailable.
	ProviderCoreML
)

// ProviderConfig configures the ONNX Runtime session options.
//
// The zero value is valid: it selects the CPU execution provider with
// full graph optimization and automatic thread count selection.
type ProviderConfig struct {
	// Provider selects the hardware execution provider.
	// Default: ProviderCPU.
	Provider ExecutionProvider

	// CoreMLOptions configures the CoreML execution provider (macOS only).
	// Common keys:
	//   "MLComputeUnits"  – "ALL" | "CPUAndNE" | "CPUAndGPU" | "CPUOnly" (default: "ALL")
	//   "ModelFormat"     – "MLProgram" | "NeuralNetwork" (default: "MLProgram")
	// See: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
	CoreMLOptions map[string]string

	// IntraOpThreads sets the number of threads for intra-operator parallelism.
	// 0 auto-selects: min(runtime.NumCPU(), 4).
	IntraOpThreads int

	// InterOpThreads sets the number of threads for inter-operator parallelism.
	// Only relevant when ORT runs in parallel execution mode.
	// 0 uses the ORT default.
	InterOpThreads int

	// GraphOptimization controls the ONNX graph optimization level.
	// The zero value maps to GraphOptimizationLevelEnableAll (best performance).
	// Set explicitly to ort.GraphOptimizationLevelDisableAll to opt out.
	GraphOptimization ort.GraphOptimizationLevel

	// EnableMemPattern enables memory pattern optimization.
	// When true ORT pre-allocates memory based on observed usage patterns.
	EnableMemPattern bool
}

// CPUProviderConfig returns a ProviderConfig optimized for CPU inference.
// Enables all graph optimizations and memory pattern for best throughput.
func CPUProviderConfig() ProviderConfig {
	return ProviderConfig{
		Provider:          ProviderCPU,
		GraphOptimization: ort.GraphOptimizationLevelEnableAll,
		EnableMemPattern:  true,
	}
}

// CoreMLProviderConfig returns a ProviderConfig using Apple's CoreML framework
// with all compute units (CPU + GPU + Neural Engine).
// Falls back to CPU automatically when CoreML is not available.
func CoreMLProviderConfig() ProviderConfig {
	return ProviderConfig{
		Provider: ProviderCoreML,
		CoreMLOptions: map[string]string{
			"MLComputeUnits": "ALL",
			"ModelFormat":    "MLProgram",
		},
		GraphOptimization: ort.GraphOptimizationLevelEnableAll,
		EnableMemPattern:  true,
	}
}

// CoreMLCPUOnlyConfig returns a ProviderConfig using CoreML with CPU-only compute units.
// Useful as an isolation baseline: same CoreML graph compilation overhead as ALL/GPU,
// but no hardware offload. Reveals how much of the speedup comes from the accelerators
// vs CoreML's own graph optimizations.
//
// Note: "CPUAndNE" (Neural Engine only) is not available in ONNX Runtime 1.24.x.
func CoreMLCPUOnlyConfig() ProviderConfig {
	return ProviderConfig{
		Provider: ProviderCoreML,
		CoreMLOptions: map[string]string{
			"MLComputeUnits": "CPUOnly",
			"ModelFormat":    "MLProgram",
		},
		GraphOptimization: ort.GraphOptimizationLevelEnableAll,
		EnableMemPattern:  true,
	}
}

// CoreMLNeuralEngineConfig is an alias kept for forward compatibility.
// It falls back to CPUAndGPU since "CPUAndNE" requires ORT 1.25+ with ANE support.
//
// Deprecated: Use CoreMLGPUConfig or CoreMLProviderConfig instead.
func CoreMLNeuralEngineConfig() ProviderConfig {
	return CoreMLGPUConfig()
}

// CoreMLGPUConfig returns a ProviderConfig using CoreML with CPU and GPU
// compute units. Useful for comparing ANE vs GPU dispatch on Apple Silicon.
func CoreMLGPUConfig() ProviderConfig {
	return ProviderConfig{
		Provider: ProviderCoreML,
		CoreMLOptions: map[string]string{
			"MLComputeUnits": "CPUAndGPU",
			"ModelFormat":    "MLProgram",
		},
		GraphOptimization: ort.GraphOptimizationLevelEnableAll,
		EnableMemPattern:  true,
	}
}
