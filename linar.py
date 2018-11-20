def reduce_row_down(A, i):
    if A[i][i] == 0:
        return A
    for o in range(i + 1, len(A)):
        f = A[o][i] / A[i][i]
        for j in range(i, len(A[0])):
            A[o][j] = A[o][j] - A[i][j] * f
    return A

def reduce_row_up(A, i):
    if A[i][i] == 0:
        return A
    for o in range(i - 1, -1, -1):
        f = A[o][i] / A[i][i]
        for j in range(i, len(A[0])):
            A[o][j] = A[o][j] - A[i][j] * f
    return A

def normalize_diagonal(A):
    for i in range(len(A)):
        f = A[i][i]
        if f != 0:
            for j in range(i, len(A[0])):
                A[i][j] = A[i][j] / f

def reduced_row_eshelon_form(A):
    for i in range(0, len(A)):
        reduce_row_down(A, i)
        reduce_row_up(A, i)
    normalize_diagonal(A)
    return A

def gaussian_elimination(A):
    for i in range(len(A)):
        reduce_row_down(A, i)
    for i in range(len(A) - 1, 0, -1):
        reduce_row_up(A, i)
    normalize_diagonal(A)
    return A

A = [[3, -7, -2, 1, 0, 0],
     [-3, 5, 1, 0, 1, 0],
     [6, -4, 0, 0, 0, 1]]

R = gaussian_elimination(A)
print(A)

[[0.6666666666666666, 1.3333333333333333, 0.5],
 [1.0, 2.0, 0.5],
 [-3.0, -5.0, -1.0]]

B
reduced_row_eshelon_form