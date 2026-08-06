import cv2
import os
import numpy as np
from app.vision.extractor import preprocess, find_grid_corners, warp_perspective, slice_grid


def test_pipeline(image_path: str):
    # Load the original image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from '{image_path}'")
        return

    # Test Step 1: Preprocessing
    thresh = preprocess(img)

    # Test Step 2: Finding Corners
    corners = find_grid_corners(thresh)

    # Create a visual copy to draw on
    debug_img = img.copy()

    if corners is not None:
        print("Success! Grid corners found:")
        print(corners)

        # Draw a thick green polygon around the detected grid
        cv2.drawContours(debug_img, [corners], -1, (0, 255, 0), 4)

        # Draw red circles on each of the 4 corner points
        for point in corners:
            x, y = point[0]
            cv2.circle(debug_img, (x, y), 8, (0, 0, 255), -1)

        # Flatten the image
        flat_grid = warp_perspective(img, corners)

        # Slice
        cells = slice_grid(flat_grid)

        # --- VISUALIZE RESULTS (Moved inside the success block) ---
        cv2.imshow("1. Thresholded Image", thresh)
        cv2.imshow("2. Detected Grid", debug_img)
        cv2.imshow("3. Flattened Grid", flat_grid)
        cv2.imshow("4. Cell (0,0)", cells[56])
        print("Press any key on the image window to close it...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Failed: No grid found in the image.")
        # We can still show what the thresholding looked like to debug why it failed
        cv2.imshow("1. Thresholded Image", thresh)
        print("Press any key on the image window to close it...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_pipeline("../examples/screenshot_normal.png")