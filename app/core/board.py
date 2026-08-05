class SudokuBoard:
    def __init__(self, grid: list[list[int]]):
        self.grid = grid

    def get_grid(self) -> list[list[int]]:
        return self.grid

    def find_empty(self) -> tuple[int, int] | None:
        """
        Finds the next empty cell (represented by 0).
        Returns a tuple (row, col) or None if full.
        """
        for i in range(9):
            for j in range(9):
                if self.grid[i][j] == 0:
                    return (i, j)
        return None

    def is_valid_to_add(self, num: int, row: int, col: int) -> bool:
        """
        Checks if placing 'num' at (row, col) follows Sudoku rules.
        """
        # Check the row
        for i in range(9):
            if self.grid[row][i] == num and col != i:
                return False

        # Check the column
        for i in range(9):
            if self.grid[i][col] == num and row != i:
                return False

        # Check the 3x3 box
        box_x = col // 3
        box_y = row // 3

        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if self.grid[i][j] == num and (i, j) != (row, col):
                    return False

        return True

    def set_cell(self, row: int, col: int, num: int) -> None:
        """Updates a specific cell on the board."""
        self.grid[row][col] = num

    def __str__(self) -> str:
        """
        Returns a string representation of the Sudoku board.
        """
        output = ""
        for i in range(9):
            if i % 3 == 0 and i != 0:
                output += "- - - - - - - - - - - - - \n"

            for j in range(9):
                if j % 3 == 0 and j != 0:
                    output += " | "

                if j == 8:
                    output += str(self.grid[i][j]) + "\n"
                else:
                    output += str(self.grid[i][j]) + " "
        return output