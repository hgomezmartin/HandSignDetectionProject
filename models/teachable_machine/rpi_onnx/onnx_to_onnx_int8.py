from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    # Entrada: tu ONNX original
    input_model  = "model.onnx"
    # Salida: el ONNX cuantizado a INT8
    output_model = "model_int8.onnx"

    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QUInt8
    )
    print(f"Modelo cuantizado guardado en: {output_model}")

if __name__ == "__main__":
    main()