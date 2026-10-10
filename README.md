## Piper1 Quantizer
A lightweight Python tool designed to quantize Piper1-gpl ONNX models to FP16 or INT8. FP16 reduces the model size by nearly 50% with minimal impact on synthesis quality; dynamic INT8 reduces it to roughly a third of the original, making both ideal for deployment on resource-constrained devices.

## Requirements

* Python 3.12
* onnx
* onnxconverter_common
* numpy
* onnxruntime (INT8 mode only)

## Features

* Easy CLI: Convert models directly from your terminal, FP16 by default or INT8 with `--mode int8`.
* Library Support: Import the quantization logic into your own Python scripts.
* Optimized for Piper: Automatically handles op_block_list to maintain stability for specific ONNX operations used in Piper models.

## Installation
Install the package in editable mode from your project root:

``` sh
python3.12 -m venv .venv
source .venv/bin/activate
python3 -m pip install .
``` 

## Usage

### 1. Command Line Interface

``` sh
# FP16 (default)
python3 quantize.py input_model.onnx output_model_fp16.onnx

# INT8 (dynamic quantization)
python3 quantize.py --mode int8 input_model.onnx output_model_int8.onnx
``` 

### 2. As a Python Library
You can also integrate the quantization logic into your own workflows:

from quantize import convert_to_fp16, convert_to_int8
# Convert your Piper model to float16
convert_to_fp16("path/to/model.onnx", "path/to/model_fp16.onnx")
# Convert your Piper model to dynamic int8
convert_to_int8("path/to/model.onnx", "path/to/model_int8.onnx")

## Why Quantize?
Piper models are often exported in float32. By converting to float16:

* File Size: Reduced by ~50% (e.g., 60MB → 30MB).
* Inference Speed: Can be faster on hardware with FP16 support (like modern GPUs and some mobile CPUs).
* Compatibility: Specifically keeps RandomNormalLike and Range operations in float32 to prevent audio artifacts.

By converting to INT8 (dynamic quantization, the same recipe sherpa-onnx ships for Piper voices):

* File Size: Reduced to roughly a third of the float32 original (e.g., en_US-amy-medium: 63MB → 19MB; uk_UA-tetiana-high: 114MB → 38MB).
* No Calibration: Activation scales are computed at runtime (DynamicQuantizeLinear), so no calibration dataset is needed.
* Hybrid Graph: Conv weights become uint8 (ConvInteger), while ops without an integer form (ConvTranspose, CumSum, Gather, ...) stay in float32. The result is not a fully integer model.
* Quality: INT8 changes the math much more than FP16 does. Listen to the output before shipping an INT8 voice; do not rely on the file being produced successfully.
