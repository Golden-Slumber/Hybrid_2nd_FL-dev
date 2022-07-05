import sys
import numpy
import networkx
from sklearn.model_selection import train_test_split

from Utils.local_optimization_algorithms import NAG, GD
from Utils.mixing_matrix_construction import symmetric_fdla_matrix
from tqdm import tqdm
from constants import *

home_dir = '../'
sys.path.append(home_dir)


class Cluster(object):

    def __init__(self, index, model, cluster_iter, x, full=False, mode=HYBRID):
        self.index = index
        self.model = model
        self.n_devices = self.model.n_devices_by_cluster[self.index]
        self.cluster_iter = cluster_iter
        self.full = full
        self.mode = mode

        self.n_edges = None
        self.G = None
        self.W = None
        self.W_s = None
        self.connection2server = None

        self.x = numpy.random.rand(self.model.dim, self.n_devices)
        for i in range(self.n_devices):
            self.x[:, i] = x
        self.y = self.x.copy()
        self.s = numpy.zeros((self.model.dim, self.n_devices))
        for i in range(self.n_devices):
            self.s[:, i] = self.grad(self.x[:, i], i)
        self.prev_grad = self.s.copy()

        # local optimizer
        self.mu = 0.005
        self.local_optimizer = 'NAG'
        self.local_iter = 500
        self.delta = None

    def init_graph(self):
        graph = networkx.cycle_graph(self.n_devices)
        self.n_edges = graph.number_of_edges()
        self.G = graph

        if self.mode == HYBRID:
            self.W, alpha = symmetric_fdla_matrix(self.G)

            # stable
            W_min_diag = min(numpy.diag(self.W))
            tmp = (1 - 1e-1) / (1 - W_min_diag)
            self.W_s = self.W * tmp + numpy.eye(self.n_devices) * (1 - tmp)
        elif self.mode == LOCAL:
            self.W = numpy.eye(self.n_devices)
            self.W_s = numpy.eye(self.n_devices)

    def update_connection2server(self, prob=0.5):
        self.connection2server = numpy.zeros(self.n_devices)
        if self.full:
            for i in range(self.n_devices):
                self.connection2server[i] = 1
        else:
            while sum(self.connection2server) == 0:
                for i in range(self.n_devices):
                    if numpy.random.rand() < prob:
                        self.connection2server[i] = 1

    def get_model(self):
        local_w = numpy.zeros(self.model.dim)
        n_active_devices = 0
        for i in range(self.n_devices):
            if self.connection2server[i] == 1:
                local_w = numpy.add(local_w, self.x[:, i])
                n_active_devices += 1
        local_w /= n_active_devices
        return local_w

    def get_gradient(self):
        local_s = numpy.zeros(self.model.dim)
        n_active_devices = 0
        for i in range(self.n_devices):
            if self.connection2server[i] == 1:
                local_s = numpy.add(local_s, self.s[:, i])
                n_active_devices += 1
        local_s /= n_active_devices
        return local_s

    def set_model(self, w):
        for i in range(self.n_devices):
            self.x[:, i] = w

    def set_gradient(self, s):
        for i in range(self.n_devices):
            self.s[:, i] = s
            self.prev_grad = self.s.copy()

    def local_training(self):
        print("local training cluster " + str(self.index))
        for i in tqdm(range(self.cluster_iter)):
            self.information_mixing()
            self.local_update()
            # print("cluster " + str(self.index) + ": local iter " + str(i))

    def information_mixing(self):
        self.y = self.x.dot(self.W)
        self.s = self.s.dot(self.W_s)
        self.s = numpy.subtract(self.s, self.prev_grad)
        for i in range(self.n_devices):
            self.prev_grad[:, i] = self.grad(self.y[:, i], i)
        # self.prev_grad = self.grad(self.y)
        self.s = numpy.add(self.s, self.prev_grad)

    def local_update(self):
        for i in range(self.n_devices):
            grad_y = self.grad(self.y[:, i], i)

            def _grad(tmp):
                return self.grad(tmp, i) - grad_y + self.s[:, i] + self.mu * (tmp - self.y[:, i])

            if self.local_optimizer == 'NAG':
                self.x[:, i], count = NAG(_grad, self.y[:, i].copy(), self.model.L + self.mu,
                                          self.model.sigma + self.mu, self.local_iter)
            else:
                if self.delta is not None:
                    self.x[:, i], count = GD(_grad, self.y[:, i].copy(), self.delta, self.local_iter)
                else:
                    self.x[:, i], count = GD(_grad, self.y[:, i].copy(),
                                             2 / (self.model.L + self.mu + self.model.sigma + self.mu), self.local_iter)

    def obj_fun(self, w, i=None, j=None):
        return self.model.obj_fun(w, self.index, i, j)

    def grad(self, w, i=None, j=None):
        return self.model.grad(w, self.index, i, j)

    def grad_full(self, w, i=None):
        return self.model.grad_full(w, i)

    def hessian(self, w, i=None, j=None):
        return self.model.hessian(w, self.index, i, j)
