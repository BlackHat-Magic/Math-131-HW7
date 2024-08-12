def gauss_seidel(A, b, x0, epsilon, max_):
    n = len(A)
    x = x0.copy()
    iterations = 0
    for _ in range(max_):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]
        error = np.linalg.norm(x - x_old, ord=2)
        iterations += 1
        if(error < epsilon):
            break
    return(x, error, iterations)