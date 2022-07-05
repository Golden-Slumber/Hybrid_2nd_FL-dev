import sys
import numpy
from sklearn.model_selection import train_test_split
from scipy import optimize
from Models.logistic_regression import LogisticRegression
from System.cluster import Cluster
from constants import *

home_dir = '../'
sys.path.append(home_dir)


class Server(object):

    def __init__(self, n_devices_by_cluster, m_mean, data_name, global_iter=50, cluster_iter=10, kappa=10,
                 balanced=True, full=False, mode=HYBRID):
        # basic information
        self.n_cluster = len(n_devices_by_cluster)
        self.n_devices_by_cluster = n_devices_by_cluster
        self.n_devices = sum(n_devices_by_cluster)
        self.model = LogisticRegression(n_devices_by_cluster, m_mean, kappa=kappa, balanced=balanced)
        self.clusters = []
        self.full = full
        self.mode = mode

        # learning information
        self.global_iter = global_iter
        self.cluster_iter = cluster_iter
        self.t = 0
        self.w = None
        self.func_error = numpy.zeros(self.global_iter + 1)
        self.var_error = numpy.zeros(self.global_iter + 1)
        self.acc = numpy.zeros(self.global_iter + 1)

        # dataset information
        self.data_name = data_name
        self.m_mean = m_mean
        self.m_tot = m_mean * self.n_cluster
        self.balanced = balanced
        self.kappa = kappa

    def load_dataset(self):
        file_name = home_dir + 'Resources/' + self.data_name + '.npz'
        npz_file = numpy.load(file_name)
        num_tot = int(self.m_tot + self.m_tot / 4)
        X_tot = npz_file['x_mat'][:num_tot, :]
        Y_tot = npz_file['y_vec'][:num_tot]
        X_tot, X_test, Y_tot, Y_test = train_test_split(X_tot, Y_tot, test_size=0.2,
                                                        random_state=0)
        self.model.load_dataset(X_tot, X_test, Y_tot, Y_test)
        self.w = numpy.random.randn(self.model.dim)

    def initialization(self):
        x_0 = numpy.random.randn(self.model.dim)
        for i in range(self.n_cluster):
            cluster = Cluster(i, self.model, self.cluster_iter, x_0, full=self.full, mode=self.mode)
            cluster.init_graph()
            cluster.update_connection2server()
            self.clusters.append(cluster)

    def save_metric(self):
        self.func_error[self.t] = abs((self.model.obj_fun(self.w) - self.model.f_min) / self.model.f_min)
        self.var_error[self.t] = numpy.linalg.norm(self.w - self.model.w_min) / numpy.linalg.norm(self.model.w_min)
        self.acc[self.t] = self.model._get_accuracy(self.w)

    def get_results(self):
        res = {
            'w': self.w,
            'var_error': self.var_error[:self.global_iter + 1],
            'func_error': self.func_error[:self.global_iter + 1],
            'acc': self.acc[:self.global_iter + 1]
        }
        return res

    def convergence_check(self):
        if numpy.linalg.norm(self.model.grad(self.w)) < 1e-10:
            return True
        if numpy.linalg.norm(self.w - self.model.w_min) > 1e5:
            print("Something went wrong, the algorithm diverges!")
            exit(-1)

    def converge_metric(self):
        i = self.t + 1
        while i <= self.global_iter:
            self.func_error[i] = self.func_error[i - 1]
            self.var_error[i] = self.var_error[i - 1]
            self.acc[i] = self.acc[i - 1]
            i += 1

    def training(self):
        self.initialization()
        self.save_metric()

        for self.t in range(1, self.global_iter + 1):
            new_w = numpy.zeros(self.model.dim)
            new_s = numpy.zeros(self.model.dim)
            for i in range(self.n_cluster):
                self.clusters[i].local_training()
                new_w = numpy.add(new_w, self.clusters[i].get_model())
                new_s = numpy.add(new_s, self.clusters[i].get_gradient())
            new_w = new_w / self.n_cluster
            new_s = new_s / self.n_cluster
            for i in range(self.n_cluster):
                self.clusters[i].set_model(new_w)
                self.clusters[i].set_gradient(new_s)
                self.clusters[i].update_connection2server()
            self.w = new_w

            self.save_metric()
            print(
                "iter " + str(self.t) + " var error: " + str(self.var_error[self.t]) + " func error: " + str(
                    self.func_error[self.t]) + " acc: " + str(self.acc[self.t]))
            if self.convergence_check():
                print("Converge !")
                self.converge_metric()
                break
        return self.get_results()


if __name__ == '__main__':
    n_devices_by_cluster = [1, 1, 1, 1, 1]
    m_mean = 88000
    data_name = 'covtype'
    kappa = 10000

    demo = 'hybrid_2nd_demo'
    legends = ['hybrid_2nd']

    server = Server(n_devices_by_cluster, m_mean, data_name, kappa=kappa)
    server.load_dataset()
    res = server.training()

