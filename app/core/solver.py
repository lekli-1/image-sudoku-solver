from app.core.board import SudokuBoard

def solve(board: SudokuBoard) -> bool:
    """
    Solves the SudokuBoard in-place using backtracking.
    Returns True if solved, False if unsolvable.
    """
    empty_spot = board.find_empty()

    if not empty_spot:
        return True  # Board is full, puzzle is solved

    row, col = empty_spot

    for num in range(1, 10):
        if board.is_valid_to_add(num, row, col):
            board.set_cell(row, col, num)

            if solve(board):
                return True

            # Backtrack
            board.set_cell(row, col, 0)

    return False