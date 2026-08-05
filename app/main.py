from app.core.board import SudokuBoard
from app.core.solver import solve

if __name__ == "__main__":
    raw_grid = [
        [7, 8, 0, 4, 0, 0, 1, 2, 0],
        [6, 0, 0, 0, 7, 5, 0, 0, 9],
        [0, 0, 0, 6, 0, 1, 0, 7, 8],
        [0, 0, 7, 0, 4, 0, 2, 6, 0],
        [0, 0, 1, 0, 5, 0, 9, 3, 0],
        [9, 0, 4, 0, 6, 0, 0, 0, 5],
        [0, 7, 0, 3, 0, 0, 0, 1, 2],
        [1, 2, 0, 0, 0, 7, 4, 0, 0],
        [0, 4, 9, 2, 0, 6, 0, 0, 7]
    ]

    # Board object
    game_board = SudokuBoard(raw_grid)

    print("Original Board:")
    print(game_board)

    print("\nSolving...\n")
    success = solve(game_board)

    if success:
        print("Solved Board:")
        print(game_board)
    else:
        print("This board cannot be solved.")