def reduce_row_down(A, i, L = None):
    if A[i][i] == 0:
        return A
    for o in range(i + 1, len(A)):
        f = A[o][i] / A[i][i]
        if L is not None:
            L[o][i] = f
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

def LU_decomposition(A):
    m = len(A)
    L = [[0 for i in range(m)] for j in range(m)]
    for i in range(m):
        L[i][i] = 1
    for i in range(0, len(A)):
        reduce_row_down(A, i, L)
    U = A
    return L, U


A = [[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 3, 1, 0, 0, 1]]
A_ = gaussian_elimination(A)

[[1.0, 0.0, 0.0],
 [0.0, 1.0, 0.0],
 [0.0,-3.0, 1.0]]

print(A_)
