import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.vision.model import SudokuCNN

def clean_cell(cell_img: np.ndarray, padding: int = 4) -> np.ndarray:
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()

    _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.ones_like(gray) * 255

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if w < 3 or h < 3:
        return np.ones_like(gray) * 255

    y_start = max(0, y - padding)
    y_end = min(gray.shape[0], y + h + padding)
    x_start = max(0, x - padding)
    x_end = min(gray.shape[1], x + w + padding)

    cropped_digit = gray[y_start:y_end, x_start:x_end]
    _, final_clean = cv2.threshold(cropped_digit, 150, 255, cv2.THRESH_BINARY)

    return final_clean


def predict_board(cells: list, model_path: str = "sudoku_cnn.pth") -> list[list[int]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SudokuCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    board = []
    current_row = []

    for i, cell_img in enumerate(cells):
        clean_img = clean_cell(cell_img)
        pil_img = Image.fromarray(clean_img)
        tensor_img = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor_img)
            prediction = torch.argmax(output, dim=1).item()

        current_row.append(prediction)

        if (i + 1) % 9 == 0:
            board.append(current_row)
            current_row = []

    return board