class sudoku:
    def __init__(self, board: list[list[int]]) -> None:
        self.board = board

    def isvalid(self):
        # Check rows
        for row in self.board:
            if not self.isvalidgroup(row):
                return False

        # Check columns
        for col in range(9):
            column = [self.board[row][col] for row in range(9)]
            if not self.isvalidgroup(column):
                return False

        # Check 3x3 subgrids
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                subgrid = [self.board[x][y] for x in range(i, i+3) for y in range(j, j+3)]
                if not self.isvalidgroup(subgrid):
                    return False

        return True

    def isvalidgroup(self, group):
        # Remove zeros (empty cells) and check for duplicates
        numbers = [n for n in group if n != 0]
        return len(numbers) == len(set(numbers))