import numpy as np
import cv2
import onnxruntime as ort


def clean_cell(cell_img: np.ndarray, padding: int = 4) -> np.ndarray:
    """Removes grid lines and centers the digit."""
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


def predict_board(cells: list, model_path: str = "models/sudoku_cnn.onnx") -> list[list[int]]:
    # Load the ONNX model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    board = []
    current_row = []

    for img in cells:
        # CLEAN THE CELL FIRST
        clean_img = clean_cell(img)

        # Resize to 32x32
        resized = cv2.resize(clean_img, (32, 32))

        # Scale 0-255 to 0.0-1.0
        normalized_01 = resized.astype(np.float32) / 255.0

        # Scale to -1.0 to 1.0
        normalized_final = (normalized_01 - 0.5) / 0.5

        # Reshape to match the exact tensor shape ONNX expects: (Batch=1, Channel=1, H=32, W=32)
        tensor_img = np.expand_dims(normalized_final, axis=(0, 1)).astype(np.float32)

        # Run the ONNX prediction
        outputs = session.run(None, {input_name: tensor_img})

        # The output is an array of probabilities; get the index of the highest one
        prediction = np.argmax(outputs[0])
        current_row.append(int(prediction))

        if len(current_row) == 9:
            board.append(current_row)
            current_row = []

    return board