import torch
import numpy as np
from pytorch_nndct.apis import torch_quantizer
from pqd_cnn import PQDCNN  # Your model class

def main():
    # Load trained model
    model = PQDCNN(num_classes=6)
    model.load_state_dict(torch.load('/workspace/pqd_cnn.pth'))
    model.eval()

    # Initialize quantizer
    quantizer = torch_quantizer(
        quant_mode='calib',
        module=model,
        input_args=torch.randn(1, 1, 256),
        output_dir='/workspace/quantized'
    )
    quant_model = quantizer.quant_model

    # Calibration process
    def calibrate():
        print("Running calibration...")
        for _ in range(100):  # 100 calibration iterations
            # Generate realistic input data (adjust as needed)
            dummy_input = torch.randn(1, 1, 256) * 230  # 230V nominal
            _ = quant_model(dummy_input)  # Forward pass

    calibrate()

    # Export quantized model
    print("Exporting quantized model...")
    quantizer.export_quant_config()
    quantizer.export_xmodel()

    print("Quantization complete!")
    print(f"Quantized model saved to: /workspace/quantized")

if __name__ == "__main__":
    main()