import numpy as np
import cv2

rows, cols = 7, 10  # squares (inner corners will be rows-1, cols-1)
square_px = 80

board = np.kron(
    [[(i + j) % 2 for j in range(cols)] for i in range(rows)],
    np.ones((square_px, square_px), dtype=np.uint8)
) * 255

cv2.imwrite("checkerboard.png", board)
print(f"Saved checkerboard.png ({board.shape[1]}x{board.shape[0]} px)")
print(f"Inner corners: {cols-1} cols x {rows-1} rows")
print(f"Use: --cols {cols-1} --rows {rows-1}")
