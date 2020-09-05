import sys

from SudokuSolve.extractor import Extractor
from SudokuSolve import sudoku_solver

def output(a):
    sys.stdout.write(str(a))


def display_sudoku(sudoku):
    for i in range(9):
        for j in range(9):
            cell = sudoku[i][j]
            if cell == 0 or isinstance(cell, set):
                output('.')
            else:
                output(cell)
            if (j + 1) % 3 == 0 and j < 8:
                output(' |')

            if j != 8:
                output('  ')
        output('\n')
        if (i + 1) % 3 == 0 and i < 8:
            output("--------+----------+---------\n")


def main(image):
    extractor = Extractor(image)
    grid = extractor.get_sudoku()
    print('Sudoku:')
    display_sudoku(grid.tolist())
    solution = sudoku_solver(grid)
    print('Solution:')
    display_sudoku(solution.tolist())


if __name__ == '__main__':

    main(sys.argv[1])

