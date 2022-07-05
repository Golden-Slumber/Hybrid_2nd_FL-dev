import numpy
import networkx
import cvxpy


def asymmetric_fdla_matrix(G, m):
    n = G.number_of_nodes()

    ind = networkx.adjacency_matrix(G).toarray() + numpy.eye(n)
    ind = ~ind.astype(bool)

    average_vec = m / m.sum()
    average_matrix = numpy.ones((n, 1)).dot(average_vec[numpy.newaxis, :]).T
    one_vec = numpy.ones(n)

    W = cvxpy.Variable((n, n))

    if ind.sum() == 0:
        prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.norm(W - average_matrix)),
                             [cvxpy.sum(W, axis=1) == one_vec, cvxpy.sum(W, axis=0) == one_vec])
    else:
        prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.norm(W - average_matrix)),
                             [W[ind] == 0, cvxpy.sum(W, axis=1) == one_vec, cvxpy.sum(W, axis=0) == one_vec])
    prob.solve()

    W = W.value
    W[ind] = 0
    W -= numpy.diag(W.sum(axis=1) - 1)
    alpha = numpy.linalg.norm(W - average_matrix, 2)
    return W, alpha


def symmetric_fdla_matrix(G):
    n = G.number_of_nodes()

    ind = networkx.adjacency_matrix(G).toarray() + numpy.eye(n)
    ind = ~ind.astype(bool)

    average_matrix = numpy.ones((n, n)) / n
    one_vec = numpy.ones(n)

    W = cvxpy.Variable((n, n))

    if ind.sum() == 0:
        prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.norm(W - average_matrix)),
                             [W == W.T, cvxpy.sum(W, axis=1) == one_vec])
    else:
        prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.norm(W - average_matrix)),
                             [W[ind] == 0, W == W.T, cvxpy.sum(W, axis=1) == one_vec])
    prob.solve()

    W = W.value
    W = (W + W.T) / 2
    W[ind] = 0
    W -= numpy.diag(W.sum(axis=1) - 1)
    alpha = numpy.linalg.norm(W - average_matrix, 2)

    return W, alpha


def generate_mixing_matrix(model):
    return symmetric_fdla_matrix(model.G)
