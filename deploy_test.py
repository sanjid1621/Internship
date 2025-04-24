import numpy as np
import torch

class PQDCNNSimulator:
    """PQD detector with PID correction values"""
    def __init__(self):
        print("Using fallback simulation mode")
        self.classes = ["Normal", "Sag", "Swell", "Harmonics", "Transient", "Fluctuation"]
        
        # PID controller parameters
        self.pid_gains = {
            "Normal": {"Kp": 0.1, "Ki": 0.01, "Kd": 0.05},
            "Sag": {"Kp": 0.8, "Ki": 0.2, "Kd": 0.1},
            "Swell": {"Kp": 0.6, "Ki": 0.15, "Kd": 0.08},
            "Harmonics": {"Kp": 0.4, "Ki": 0.1, "Kd": 0.05},
            "Transient": {"Kp": 1.0, "Ki": 0.05, "Kd": 0.2},
            "Fluctuation": {"Kp": 0.5, "Ki": 0.1, "Kd": 0.05}
        }
        
        # Simple model with proper dimensions
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 128, 6)  # Adjusted for 256 input samples
        )
        self.model.eval()

    def detect_pqd(self, voltage_window):
        """Detect PQD with proper input shaping"""
        with torch.no_grad():
            # Convert to proper 3D tensor: (batch=1, channels=1, length=256)
            tensor = torch.from_numpy(voltage_window.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            output = self.model(tensor).numpy()[0]
            return output

    def calculate_pid(self, voltage, disturbance_type):
        """Calculate PID correction values"""
        gains = self.pid_gains[disturbance_type]
        t = np.linspace(0, len(voltage)/5000, len(voltage))
        target = 230 * np.sin(2 * np.pi * 50 * t)
        
        error = target - voltage
        correction = (
            gains["Kp"] * error +
            gains["Ki"] * np.cumsum(error) * 0.001 +
            gains["Kd"] * np.gradient(error) * 100
        )
        return np.clip(correction, -30, 30)

    def test(self):
        """Run test with sample voltage data"""
        # Create test signal (256 samples)
        t = np.linspace(0, 0.05, 256)
        voltage = 230 * np.sin(2 * np.pi * 50 * t)
        
        # Add 30% sag between samples 100-200
        voltage[100:200] *= 0.7
        voltage += np.random.normal(0, 3, 256)  # Add noise
        
        # Detect PQD
        output = self.detect_pqd(voltage)
        predicted = np.argmax(output)
        disturbance_type = self.classes[predicted]
        
        # Calculate PID correction
        correction = self.calculate_pid(voltage, disturbance_type)
        
        # Print results
        print("\nTest successful!")
        print(f"Input shape: {voltage.shape}")
        print(f"Output shape: (1, {len(output)})")
        print(f"Sample outputs: {[f'{x:.2f}' for x in output]}")
        
        print("\n=== PQD Classification Result ===")
        print(f"Predicted Disturbance: {disturbance_type} (Confidence: {output[predicted]:.2f})")
        print("\nAll Class Scores:")
        for i, (cls, score) in enumerate(zip(self.classes, output)):
            print(f"{cls:10}: {score:8.2f} {'<-- Predicted' if i == predicted else ''}")
        
        print("\n=== PID Correction Values ===")
        print(f"Max correction: {np.max(np.abs(correction)):.2f} V")
        print(f"Avg correction: {np.mean(np.abs(correction)):.2f} V")
        print(f"Correction range: {np.min(correction):.2f} V to {np.max(correction):.2f} V")
        print(f"First 5 correction values: {[f'{x:.2f}' for x in correction[:5]]}")

if __name__ == "__main__":
    simulator = PQDCNNSimulator()
    simulator.test()