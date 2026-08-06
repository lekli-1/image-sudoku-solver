import sys
from app.vision.extractor import process_image
from app.vision.predict import predict_board
from app.core.board import SudokuBoard
from app.core.solver import solve


def main(image_path: str):
    print("\n--- Starting Sudoku Solver Pipeline ---")
    print(f"Processing: {image_path}")

    # STEP 1: Computer Vision (Extract the cells)
    print("\n1. Finding and extracting the grid...")
    try:
        cells = process_image(image_path)
    except Exception as e:
        print(f"Error extracting grid: {e}")
        print("Ensure the image is clear and contains a visible Sudoku grid.")
        return

    # STEP 2: Artificial Intelligence (Read the numbers)
    print("2. AI is reading the numbers...")
    try:
        # Pass the 81 cell images to our trained PyTorch model
        raw_grid = predict_board(cells, model_path="sudoku_cnn.pth")
    except Exception as e:
        print(f"Error reading digits: {e}")
        return

    # STEP 3: Logic Initialization
    # Wrap the 2D list into our SudokuBoard class
    board = SudokuBoard(raw_grid)

    print("\n--- AI Detected Board ---")
    print(board)

    # STEP 4: Solving the Puzzle
    print("3. Solving the puzzle...")
    success = solve(board)

    if success:
        print("\n--- Solved Board ---")
        print(board)
        print("Success! The puzzle is complete.")
    else:
        print("\n Failed to solve the board.")
        print("This usually means the OCR misread a number (e.g., read a 1 as a 7), ")
        print("making the puzzle mathematically impossible. Check the 'AI Detected Board' above.")


if __name__ == "__main__":
    # You can run this from the terminal using: python main.py my_photo.jpg
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    else:
        # Fallback to a default image if none is provided in the terminal
        target_image = "sample_sudoku.jpg"

    main(target_image)