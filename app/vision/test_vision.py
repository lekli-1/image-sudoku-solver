import cv2
import numpy as np
from app.vision.extractor import preprocess, find_grid_corners, warp_perspective, slice_grid


def test_pipeline(image_path: str):
    # 1. Load the original image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from '{image_path}'")
        return

    # 2. Test Step 1: Preprocessing
    thresh = preprocess(img)

    # 3. Test Step 2: Finding Corners
    corners = find_grid_corners(thresh)

    # 4. Create a visual copy to draw on
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


    else:
        print("Failed: No grid found in the image.")

    # --- VISUALIZE RESULTS ---
    # Option A: If running locally on your computer (pops up windows)
    cv2.imshow("1. Thresholded Image", thresh)
    cv2.imshow("2. Detected Grid", debug_img)
    cv2.imshow("3. Flattened Grid", flat_grid)
    cv2.imshow("4. Cell (0,0)", cells[1])
    print("Press any key on the image window to close it...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Option B: Save images to disk to view them in your editor
    # cv2.imwrite("debug_thresh.png", thresh)
    # cv2.imwrite("debug_grid.png", debug_img)


if __name__ == "__main__":
    # Replace with the actual path to a sample image!
    test_pipeline("screenshot_skewed.png")