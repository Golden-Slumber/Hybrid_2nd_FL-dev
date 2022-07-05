import numpy
import networkx
import matplotlib.pyplot as plt


class Model(object):

    def __init__(self, n_devices_by_cluster, m_mean, balanced=True):
        self.n_devices_by_cluster = n_devices_by_cluster
        self.n_cluster = len(n_devices_by_cluster)
        self.n_devices = sum(n_devices_by_cluster)
        self.m_mean = m_mean  # average number of data samples held by each cluster
        self.dim = None  # dimension of the variable
        self.X = None  # data
        self.Y = None  # noisy label
        self.X_by_cluster = []
        self.Y_by_cluster = []
        self.X_tot = None
        self.Y_tot = None
        self.X_test = None
        self.Y_test = None
        # self.Y_0 = None  # original label
        # self.x_0 = None  # true variable values
        self.sigma = 0  # strong convexity constant
        self.L = None  # smoothness constant
        self.balanced = balanced  # iid or non-iid
        self.m_tot = m_mean * self.n_cluster

        # generate the number of samples held by each cluster
        if self.balanced:
            self.m = numpy.ones(self.n_cluster, dtype=int) * self.m_mean
        else:
            tmp = numpy.random.random(self.n_cluster)
            tmp *= self.m_tot * 0.3 / tmp.sum()
            tmp = tmp.astype(int) + int(self.m_mean * 0.7)

            extra = self.m_tot - tmp.sum()
            i = 0
            while extra > 0:
                tmp[i] += 1
                extra -= 1
                i += 1
                i %= self.n_cluster

            self.m = tmp

    def split_data(self, m, X):
        """
        helper function to split data according to the number of training samples per device
        """
        cum_sum = m.cumsum().astype(int).tolist()
        inds = zip([0] + cum_sum[:-1], cum_sum)
        return [X[start:end] for (start, end) in inds]

    def grad(self, w, i=None, j=None):
        """
        gradient at w.
        if i is none, returns the full gradient;
        if i is not none but j is, returns the gradient at i-th device;
        otherwise, returns the gradient of j-th sample at i-th device.
        """
        pass

    def grad_full(self, w, i=None):
        """
        full gradient at w
        """
        pass

    def grad_vec(self, w):
        """
        gradient at w, returns the matrix of gradients of all subproblems.
        """
        pass

    def hessian(self, w, i=None, j=None):
        """
        Hessian matrix at w.
        """
        pass

    def obj_fun(self, w, i=None, j=None):
        """
        objective function value at w.
        """
        pass


# if __name__ == '__main__':
