import unittest
from board import sudoku


class TestSudokuValidator(unittest.TestCase):

    def test_valid_complete_sudoku(self):
        """Test a completely solved valid sudoku"""
        valid_board = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ]
        s = sudoku(valid_board)
        self.assertTrue(s.isvalid())

    def test_valid_partial_sudoku(self):
        """Test a valid partially filled sudoku (with zeros)"""
        partial_board = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
        s = sudoku(partial_board)
        self.assertTrue(s.isvalid())

    def test_empty_board(self):
        """Test an empty sudoku (all zeros should be valid)"""
        empty_board = [[0] * 9 for _ in range(9)]
        s = sudoku(empty_board)
        self.assertTrue(s.isvalid())

    def test_invalid_duplicate_in_row(self):
        """Test sudoku with duplicate in a row"""
        invalid_board = [
            [5, 5, 4, 6, 7, 8, 9, 1, 2],  # Two 5s in row
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ]
        s = sudoku(invalid_board)
        self.assertFalse(s.isvalid())

    def test_invalid_duplicate_in_column(self):
        """Test sudoku with duplicate in a column"""
        invalid_board = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [5, 7, 2, 1, 9, 5, 3, 4, 8],  # Another 5 in column 0
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ]
        s = sudoku(invalid_board)
        self.assertFalse(s.isvalid())

    def test_invalid_duplicate_in_subgrid(self):
        """Test sudoku with duplicate in a 3x3 subgrid"""
        invalid_board = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 5, 3, 4, 2, 5, 6, 7],  # Two 5s in top-left 3x3
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ]
        s = sudoku(invalid_board)
        self.assertFalse(s.isvalid())

    def test_single_row_with_duplicates(self):
        """Test just a single row with duplicates"""
        row = [1, 2, 3, 4, 5, 6, 7, 8, 8]
        s = sudoku([[0] * 9 for _ in range(9)])
        self.assertFalse(s.isvalidgroup(row))

    def test_single_row_valid(self):
        """Test a single valid row"""
        row = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        s = sudoku([[0] * 9 for _ in range(9)])
        self.assertTrue(s.isvalidgroup(row))

    def test_row_with_zeros(self):
        """Test row with zeros (empty cells) - should be valid"""
        row = [1, 2, 0, 4, 5, 0, 7, 8, 9]
        s = sudoku([[0] * 9 for _ in range(9)])
        self.assertTrue(s.isvalidgroup(row))


if __name__ == '__main__':
    unittest.main()