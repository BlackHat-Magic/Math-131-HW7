def jacobi(A, b, x0, epsilon, max_):
    n = len(A)
    x = np.copy(x0)
    D = np.diag(A)
    R = A - np.diagflat(D)
    iterations = 0
    for _ in range(max_):
        x_new = (b - np.dot(R, x)) / D 
        residual = np.linalg.norm(x_new - x, ord=np.inf)
        if(residual < epsilon):
            break
        x = x_new
        iterations += 1
    final_residual = np.linalg.norm(np.dot(A, x) - b, ord=np.inf)
    return(x, final_residual, iterations)
#