import sys
from sklearn.model_selection import train_test_split
import numpy
from scipy import optimize
from Models.model import Model
from sklearn.metrics import accuracy_score

home_dir = '../'
sys.path.append(home_dir)
numpy.random.seed(1024)


class LogisticRegression(Model):
    """
    f(w) = - (\sum y_i log(1/(1 + exp(w^T x_i))) + (1 - y_i) log(1 - 1/(1 + exp(w^T x_i)))) + \frac{lambda}{2} \Vert w \Vert^2
    """

    def _logit(self, X, w):
        return 1 / (1 + numpy.exp(-X.dot(w)))

    def __init__(self, n_devices, m_mean, kappa=10, **kwargs):
        super().__init__(n_devices, m_mean, **kwargs)

        self.kappa = kappa
        # self.noise_ratio = noise_ratio
        # self.x_min = None
        if kappa == 1:
            self.lam = 100
        else:
            self.lam = 1 / (self.kappa - 1)
        # these parameters can not be determined directly
        self.L = None
        self.sigma = None
        self.w_min = None
        self.f_min = None

    def load_dataset(self, X_tot, X_test, Y_tot, Y_test):
        self.X_tot = X_tot
        self.X_test = X_test
        self.Y_tot = Y_tot
        self.Y_test = Y_test
        _, self.dim = self.X_tot.shape

        # split data among clusters
        self.X = self.split_data(self.m, self.X_tot)
        self.Y = self.split_data(self.m, self.Y_tot)

        # split data among devices
        for i in range(self.n_cluster):
            m_mean_device = int(self.m_mean / self.n_devices_by_cluster[i])
            if self.balanced:
                m_device = numpy.ones(self.n_devices_by_cluster[i], dtype=int) * m_mean_device
            else:
                m_device = numpy.random.random(self.n_devices_by_cluster[i])
                m_device *= self.m[i] * 0.3 / m_device.sum()
                m_device = m_device.astype(int) + int(self.m_mean * 0.7)

                extra = self.m[i] - m_device.sum()
                i = 0
                while extra > 0:
                    m_device[i] += 1
                    extra -= 1
                    i += 1
                    i %= self.n_devices_by_cluster[i]
            X_device = self.split_data(m_device, self.X[i])
            Y_device = self.split_data(m_device, self.Y[i])
            self.X_by_cluster.append(X_device)
            self.Y_by_cluster.append(Y_device)

        # calculate optimal solution
        # self.w_min = optimize.minimize(self.obj_fun, numpy.random.rand(self.dim), jac=self.grad, method='BFGS',
        #                                options={'gtol': 1e-8}).x
        self.w_min = self.Newton_optimize()
        self.f_min = self.obj_fun(self.w_min)
        hessian = self.hessian(self.w_min)
        sig = numpy.linalg.svd(hessian, compute_uv=False)
        self.L = sig[0]
        self.sigma = sig[-1]
        self.kappa = self.L / self.sigma
        print('L: ' + str(self.L) + ' u: ' + str(self.sigma))

    def Newton_optimize(self, max_iter=50, tol=1e-10):
        alpha_list = 1 / (4 ** numpy.arange(0, 10))
        x = numpy.random.rand(self.dim)

        for i in range(max_iter):
            grad = self.grad_full(x)
            if numpy.linalg.norm(grad) <= tol:
                break
            hessian = self.hessian(x)
            p = numpy.dot(numpy.linalg.pinv(hessian), grad)

            alpha = 1
            pg = - 0.1 * numpy.sum(numpy.multiply(p, grad))
            obj_old = self.obj_fun(x)
            print('iter: ' + str(i) + ' loss: ' + str(obj_old))
            for j in range(len(alpha_list)):
                alpha = alpha_list[j]
                obj_new = self.obj_fun(x - alpha * p)

                if obj_new < obj_old + pg * alpha:
                    break
            x = x - alpha * p
        return x

    def _get_accuracy(self, w):
        y_res = self._logit(self.X_test, w)
        y_res[y_res > 0.5] = 1
        y_res[y_res <= 0.5] = 0
        acc = accuracy_score(self.Y_test, y_res)
        return acc

    def hessian(self, w, i=None, j=None, k=None):
        """
        :param w: model parameter
        :param i: cluster
        :param j: device
        :param k: batch
        """
        if i is None:
            z = self._logit(self.X_tot, w)
            z = numpy.multiply(z, 1 - z).reshape(1, self.m_tot)
            return numpy.dot(numpy.multiply(z, self.X_tot.T), self.X_tot) / self.m_tot + self.lam * numpy.eye(self.dim)
        elif j is None:
            batch_size = self.X[i].shape[0]
            z = self._logit(self.X[i], w)
            D = numpy.diag(numpy.multiply(z, 1 - z))
            return numpy.dot(numpy.dot(self.X[i].T, D), self.X[i]) / batch_size + self.lam * numpy.eye(self.dim)
        elif k is None:
            X = self.X_by_cluster[i]
            batch_size = X[j].shape[0]
            z = self._logit(X[j], w)
            D = numpy.diag(numpy.multiply(z, 1 - z))
            return numpy.dot(numpy.dot(X[j].T, D), X[j]) / batch_size + self.lam * numpy.eye(self.dim)
        else:
            X = self.X_by_cluster[i]
            if type(k) is numpy.ndarray:
                z = self._logit(X[j][k], w)
                D = numpy.diag(numpy.multiply(z, 1 - z))
                return numpy.dot(numpy.dot(X[j][k].T, D), X[j][k]) / len(k) + self.lam * numpy.eye(self.dim)
            else:
                z = self._logit(X[j][k], w)
                return z * (1 - z) * numpy.dot(X[j][k].T, X[j][k]) + self.lam

    def grad(self, w, i=None, j=None, k=None):
        """
        :param w: model parameter
        :param i: cluster
        :param j: device
        :param k: batch
        """
        if i is None:
            return self.X_tot.T.dot(self._logit(self.X_tot, w) - self.Y_tot) / self.m_tot + w * self.lam
        elif j is None:
            batch_size = self.X[i].shape[0]
            # print(self.X[i].shape)
            return self.X[i].T.dot(self._logit(self.X[i], w) - self.Y[i]) / batch_size + w * self.lam
        elif k is None:
            X = self.X_by_cluster[i]
            Y = self.Y_by_cluster[i]
            batch_size = X[j].shape[0]
            return X[j].T.dot(self._logit(X[j], w) - Y[j]) / batch_size + w * self.lam
        else:
            X = self.X_by_cluster[i]
            Y = self.Y_by_cluster[i]
            if type(k) is numpy.ndarray:
                return (self._logit(X[j][k], w) - Y[j][k]).dot(X[j][k]) / len(j) + w * self.lam
            else:
                return (self._logit(X[j][k], w) - Y[j][k]) * X[j][k] + w * self.lam

    def grad_full(self, w, i=None):
        if i is None:
            # self.Y_tot = self.Y_tot.reshape((self.m_tot, 1))
            return self.X_tot.T.dot(self._logit(self.X_tot, w) - self.Y_tot) / self.m_tot + w * self.lam
        else:
            if type(i) is numpy.ndarray:
                return (self._logit(self.X_tot[i], w) - self.Y_tot[i]).dot(self.X_tot[i]) / len(i) + w * self.lam
            else:
                return (self._logit(self.X_tot[i], w) - self.Y_tot[i]) * self.X_tot[i] + w * self.lam

    def obj_fun(self, w, i=None, j=None, k=None):
        """
        :param w: model parameter
        :param i: cluster
        :param j: device
        :param k: batch
        """
        if i is None:
            tmp = self.X_tot.dot(w)
            return - numpy.sum((self.Y_tot - 1) * tmp - numpy.log(1 + numpy.exp(-tmp))) / self.m_tot + numpy.sum(
                w ** 2) * self.lam / 2
        elif j is None:
            tmp = self.X[i].dot(w)
            batch_size = self.X[i].shape[0]
            return - numpy.sum((self.Y[i] - 1) * tmp - numpy.log(1 + numpy.exp(-tmp))) / batch_size + numpy.sum(
                w ** 2) * self.lam / 2
        elif k is None:
            X = self.X_by_cluster[i]
            Y = self.Y_by_cluster[i]
            batch_size = X[j].shape[0]
            tmp = X[j].dot(w)
            return - numpy.sum((Y[j] - 1) * tmp - numpy.log(1 + numpy.exp(-tmp))) / batch_size + numpy.sum(
                w ** 2) * self.lam / 2
        else:
            X = self.X_by_cluster[i]
            Y = self.Y_by_cluster[i]
            tmp = X[j][k].dot(w)
            return -((Y[j][k] - 1) * tmp - numpy.log(1 + numpy.exp(-tmp))) + numpy.sum(
                w ** 2) * self.lam / 2
