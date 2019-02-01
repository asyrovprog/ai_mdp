import math

EPSILON = 1e-8

def szero(s):
    return abs(s) < EPSILON

def identity(d):
    I = [[0 for i in range(d)] for j in range(d)]
    for i in range(d):
        I[i][i] = 1
    return I

def vadd(v, u):
    r = [0 for i in range(len(v))]
    for i in range(len(v)):
        r[i] = v[i] + u[i]
    return r

def vsub(v, u):
    r = [0 for i in range(len(v))]
    for i in range(len(v)):
        r[i] = v[i] - u[i]
    return r

def smul(v, s):
    u = [0 for i in range(len(v))]
    for i in range(len(v)):
        u[i] = v[i] * s
    return u

def sdiv(v, s):
    return smul(v, 1.0 / s)

def dot(u, v):
    r = 0
    for i in range(len(u)):
        r = r + u[i] * v[i]
    return r

def norm2(u):
    r = 0
    for i in range(len(u)):
        r = r + u[i] ** 2
    return math.sqrt(r)

def gram_schmidt(U):
    V = []
    for i in range(len(U)):
        u_i = U[i]
        for j in range(len(V)):
            v_j = V[j]
            f = dot(u_i, v_j)
            u_i = vsub(u_i, smul(v_j, f))
        s = norm2(u_i)
        if szero(s):
            return None
        V.append(sdiv(u_i, norm2(u_i)))
    return V

def mult(A, B):
    R = [[0 for j in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A[0])):
                R[i][j] = R[i][j] + A[i][k] * B[k][j]
    return R

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

def sort_rows(A):
    s = [[i, 0] for i in range(len(A))]
    for i in range(len(s)):
        s[i][1] = len(A[0])
        for j in range(len(A[0])):
            if not szero(A[i][j]):
                s[i][1] = j
                break
    s = sorted(s, key = lambda x: x[1])
    R = [None for i in range(len(A))]
    for i in range(len(s)):
        R[i] = A[s[i][0]]
    return R

def reduced_row_eshelon_form(A):
    for i in range(0, len(A)):
        reduce_row_down(A, i)
        reduce_row_up(A, i)
    normalize_diagonal(A)
    return sort_rows(A)

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

def del_row_col(A, r, c):
    R = [[0 for i in range(len(A[0]) - 1)] for j in range(len(A) - 1)]
    for i in range(len(A)):
        if i != r:
            for j in range(len(A[0])):
                if j != c:
                    ri = i if i < r else i - 1
                    rj = j if i < c else j - 1
                    R[ri][rj] = A[i][j]
    return R


def det(A):
    if len(A) == 1:
        return A[0][0]
    if len(A) == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    d = 0
    for i in range(len(A)):
        s = 1 if i % 2 == 0 else -1
        B = del_row_col(A, i, 0)
        d = d + s * A[i][0] * det(B)
    return d

def dets(A):
    def mul(a, b):
        s = 1
        if a[0] == '-':
            s = s * -1
            a = a[1:]
        if b[0] == '-':
            s = s * -1
            b = b[1:]
        if a == 0 or b == 0:
            return 0
        return ('-' if s == -1 else '') + str(a) + str(b)
    def add(a, b):
        if a == 0:
            return b
        if b == 0:
            return a
        return a + "+" + b
    def fixsign(x):
        if x[0] == '-' and x[1] == '-':
            return x[2:]
        return x
    def sub(a, b):
        if a == 0:
            return fixsign("-" + b)
        if b == 0:
            return a
        return a + "-" + b
    if len(A) == 1:
        return A[0][0]
    if len(A) == 2:
        return sub(mul(A[0][0],A[1][1]), mul(A[0][1], A[1][0]))
    d = 0
    for i in range(len(A)):
        s = 1 if i % 2 == 0 else -1
        B = del_row_col(A, i, 0)
        d = add(d, mul(str(s), mul(A[i][0], dets(B))))
    return d


def trace(A):
    r = 0
    for i in range(min(len(A), len(A[0]))):
        r = r + A[i][i]
    return r

def transpose(A):
    R = [[0 for i in range(len(A))] for j in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            R[j][i] = A[i][j]
    return R

def vectorize_elements(v):
    u = [None for i in range(len(v))]
    for i in range(len(v)):
        u[i] = [v[i]]
    return u

def least_square(X, Y):
    A = [[0 for i in range(len(X[0]) + 1)] for j in range(len(X))]
    for i in range(len(X)):
        for j in range(len(X[0]) + 1):
            A[i][j] = 1 if j == 0 else X[i][j - 1]
    b = Y
    At = transpose(A)
    At_x_b = mult(At, b)
    At_x_A = mult(At, A)

    E = At_x_A
    for i in range(len(E)):
        E[i].append(At_x_b[i][0])

    R = reduced_row_eshelon_form(E)
    return R

def orthogonal_projection(v, spanW):
    W = gram_schmidt(spanW)
    r = [0 for i in range(len(v))]
    for i in range(len(W)):
        r = vadd(r, smul(W[i], dot(W[i], v)))
    return r



# 1 - The set of three-by-one matrices with the first row one larger than the third row.
# 2 - 4 (1, 0, -1)...
# 3 - 4
# 4 - 4 1/sqrt(5) (2, 0, 1)...
# 5 - 1 1/sqrt(2) (1, 1)

from numpy import linalg


A = [[2,1,0],
     [1,2,1],
     [0,1,2]]

v = [[1],
     [math.sqrt(2)],
     [1]]

print(mult(A, v))
print(math.sqrt(2))
print(4.82842712474619 / 3.414213562373095)