import numpy


def _logit(X, w):
    return 1 / (1 + numpy.exp(-X.dot(w)))


def generate_data(m_total, dim, LAMBDA, noise_ratio, seed=1024):
    Y_0_total = []
    Y_total = []

    total_number = int(m_total + m_total / 4)
    X = numpy.random.randn(total_number, dim, seed=seed)
    norm = numpy.sqrt(numpy.linalg.norm(X.T.dot(X), 2) / total_number)
    X /= norm + LAMBDA
    X_total = X

    # Generate labels
    w_0 = numpy.random.rand(dim, seed=seed)
    Y_0_total = _logit(X_total, w_0)
    Y_0_total[Y_0_total > 0.5] = 1
    Y_0_total[Y_0_total <= 0.5] = 0

    if noise_ratio is not None:
        noise = numpy.random.binomial(1, noise_ratio, total_number, seed=seed)
        Y_total = numpy.multiply(noise - Y_0_total, noise) + numpy.multiply(Y_0_total, 1 - noise)
    else:
        Y_total = Y_0_total

    return X_total, Y_total