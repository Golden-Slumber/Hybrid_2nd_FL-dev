import numpy


def NAG(grad, x_0, L, sigma, n_iter=100):
    """
    Nestrov's accelerated gradient descent for strongly convex functions
    """
    x = x_0
    y = x_0
    root_kappa = numpy.sqrt(L / sigma)
    r = (root_kappa - 1) / (root_kappa + 1)
    r_1 = 1 + r
    r_2 = r

    for t in range(1, n_iter + 1):
        prev_y = y
        y = x - grad(x) / L
        x = r_1 * y - r_2 * prev_y
        if numpy.linalg.norm(y) < 1e-10:
            break
    return y, t


def GD(grad, x_0, eta, n_iter=100):
    x = x_0
    for t in range(1, n_iter + 1):
        x -= eta * grad(x)
        if numpy.linalg.norm(x) < 1e-10:
            break
    return x, t
