import cv2
import numpy as np


def process_image(image_path: str) -> tuple:
    """
    Main orchestrator function. Takes an image path and returns
    a list of 81 cropped cell images and the flattened 450x450 grid image.
    """
    # Read the image
    original_img = cv2.imread(image_path)

    # Preprocess
    processed_img = preprocess(original_img)

    # Find the corners of the biggest square
    corners = find_grid_corners(processed_img)

    if corners is None:
        raise ValueError("Could not find a Sudoku grid in the image.")

    # Warp the image flat
    flat_grid = warp_perspective(original_img, corners)

    # Slice into 81 cells
    cells = slice_grid(flat_grid)

    return cells, flat_grid


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Converts to grayscale, blurs, and applies adaptive thresholding.
    Returns a binary (black and white) image where the grid lines are white.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    # The (9, 9) is the kernel size. It must be an odd number.
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Apply Adaptive Thresholding
    # This turns the image strictly black and white.
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    return thresh


def find_grid_corners(thresh_image: np.ndarray) -> np.ndarray | None:
    """
    Finds contours, sorts by area, and returns the 4 corners of the largest square.
    """
    # Find all contours in the image
    # RETR_EXTERNAL gets only the outermost outlines, ignoring insides of numbers for now
    contours, _ = cv2.findContours(
        thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort the contours by their area, from largest to smallest
    # We only care about the top 10 largest shapes to save processing time
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Loop through the largest contours to find the grid
    for contour in contours:
        # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(contour, True)

        # Approximate the shape of the contour to smooth out jagged lines
        # 0.02 * perimeter is the mathematical precision of the smoothing
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # If the smoothed shape has exactly 4 corners, it's our Sudoku board!
        if len(approx) == 4:
            return approx

    # Return None if no 4-cornered shape was found
    return None


def order_corner_points(corners: np.ndarray) -> np.ndarray:
    """
    Orders the 4 corners in this specific order:
    Top-Left, Top-Right, Bottom-Right, Bottom-Left
    """
    # Create an empty array for the 4 ordered points
    ordered_corners = np.zeros((4, 2), dtype="float32")

    # Reshape the corners into a simpler list of 4 (x, y) coordinates
    corners = corners.reshape(4, 2)

    # Top-Left will have the smallest sum (x + y)
    # Bottom-Right will have the largest sum (x + y)
    s = corners.sum(axis=1)
    ordered_corners[0] = corners[np.argmin(s)]
    ordered_corners[2] = corners[np.argmax(s)]

    # Top-Right will have the smallest difference (y - x)
    # Bottom-Left will have the largest difference (y - x)
    diff = np.diff(corners, axis=1)
    ordered_corners[1] = corners[np.argmin(diff)]
    ordered_corners[3] = corners[np.argmax(diff)]

    return ordered_corners


def warp_perspective(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Warps the skewed grid into a perfect flat 2D square (450x450 pixels).
    """
    # Order the corners correctly
    ordered_corners = order_corner_points(corners)

    # Define the dimensions of our new perfect square
    # 450 is a great number because it's easily divisible by 9 (50 pixels per cell)
    side = 450

    # Create the coordinates for the perfect square we are projecting ONTO
    destination_corners = np.array([
        [0, 0],  # Top-Left
        [side - 1, 0],  # Top-Right
        [side - 1, side - 1],  # Bottom-Right
        [0, side - 1]  # Bottom-Left
    ], dtype="float32")

    # Calculate the mathematical transformation matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, destination_corners)

    # Apply the warp to the original image
    flat_image = cv2.warpPerspective(image, matrix, (side, side))

    return flat_image


def slice_grid(flat_image: np.ndarray) -> list[np.ndarray]:
    """
    Crops the 450x450 flat image into 81 individual 50x50 cell images.
    """
    cells = []

    # Because the image is 450x450, each of the 9 cells is 50x50 pixels
    cell_size = flat_image.shape[0] // 9

    for row in range(9):
        for col in range(9):
            # Calculate pixel coordinates for this cell
            y_start = row * cell_size
            y_end = y_start + cell_size
            x_start = col * cell_size
            x_end = x_start + cell_size

            # Crop the cell from the image using numpy slicing
            cell_img = flat_image[y_start:y_end, x_start:x_end]

            # --- CRITICAL OPTIONAL STEP ---
            # Crop an extra 4-5 pixels off all edges of the cell.
            # This removes the thick black grid lines so your CNN doesn't
            # mistake a vertical grid line for a '1' or a '7'.
            margin = 8
            cell_img = cell_img[margin:-margin, margin:-margin]

            cells.append(cell_img)

    return cells


def draw_solution(flat_grid: np.ndarray, original_board: list[list[int]], solved_board: list[list[int]]) -> np.ndarray:
    """
    Draws the solved numbers in bright green directly onto the perfectly flat Sudoku image.
    """
    # Ensure the image is in color (BGR) so we can draw green text!
    if len(flat_grid.shape) == 2:
        output_img = cv2.cvtColor(flat_grid, cv2.COLOR_GRAY2BGR)
    else:
        output_img = flat_grid.copy()

    side = output_img.shape[0]  # This is the 450px side length
    cell_size = side // 9
    font = cv2.FONT_HERSHEY_DUPLEX

    for row in range(9):
        for col in range(9):
            # Only draw if the cell was originally empty (0)
            if original_board[row][col] == 0 and solved_board[row][col] != 0:
                text = str(solved_board[row][col])

                # Math to perfectly center the text in the cell
                text_size = cv2.getTextSize(text, font, 1.2, 2)[0]
                text_x = (col * cell_size) + (cell_size - text_size[0]) // 2
                text_y = (row * cell_size) + (cell_size + text_size[1]) // 2

                # Draw text in Green (BGR format: 0, 255, 0)
                cv2.putText(output_img, text, (text_x, text_y), font, 1.2, (0, 255, 0), 2)

    return output_img