import argparse
import onnx
from onnxconverter_common import float16

def convert_to_fp16(input_path, output_path):
    """Function to be called from other Python projects."""
    model = onnx.load(input_path)

    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        # Common blocks that usually stay float32 for stability
        op_block_list=['RandomNormalLike', 'Range']
    )

    onnx.save(model_fp16, output_path)
    print(f"Successfully saved FP16 model to: {output_path}")

def convert_to_int8(input_path, output_path):
    """Function to be called from other Python projects.

    Dynamic INT8 quantization (the recipe sherpa-onnx ships for Piper
    voices): weights are stored as uint8 and Conv ops become ConvInteger,
    while activation scales are computed at runtime via
    DynamicQuantizeLinear. No calibration data is required.
    Ops without an integer form (ConvTranspose, CumSum, Gather, ...)
    stay in float32, so the result is a hybrid model, not a fully
    integer one.
    """
    # Imported lazily so FP16-only users do not need onnxruntime.
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )
    print(f"Successfully saved INT8 model to: {output_path}")

if __name__ == "__main__":
    # This part runs when you execute the script directly
    parser = argparse.ArgumentParser(description="Convert ONNX model to FP16 or INT8")
    parser.add_argument("input", help="Path to input .onnx file")
    parser.add_argument("output", help="Path to save output .onnx file")
    parser.add_argument(
        "--mode",
        choices=["fp16", "int8"],
        default="fp16",
        help="Quantization mode (default: fp16)",
    )

    args = parser.parse_args()
    if args.mode == "int8":
        convert_to_int8(args.input, args.output)
    else:
        convert_to_fp16(args.input, args.output)
