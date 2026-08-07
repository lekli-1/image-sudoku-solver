import os
import torch
from app.vision.model import SudokuCNN


def convert():
    # Load existing PyTorch model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    pth_path = os.path.join(project_root, "models", "sudoku_cnn.pth")
    onnx_path = os.path.join(project_root, "models", "sudoku_cnn.onnx")

    model = SudokuCNN()
    model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    model.eval()

    # Create a dummy input that matches the shape of one Sudoku cell
    # (Batch Size of 1, 1 Color Channel, 32x32 pixels)
    dummy_input = torch.randn(1, 1, 32, 32)

    # Export to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        input_names=['input'],
        output_names=['output']
    )

    print(f"Successfully converted model to {onnx_path}")


if __name__ == "__main__":
    convert()