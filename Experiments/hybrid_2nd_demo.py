import sys
import matplotlib
import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from System.server import Server
from constants import *

home_dir = '../'
sys.path.append(home_dir)


def plot_result(res, legends, demo):
    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line_list = []
    for i in range(len(legends)):
        line, = plt.semilogy(res[i][0], color=color_list[i], linestyle='-',
                             marker=marker_list[i],
                             markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=20)
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('Training Loss', fontsize=20)
    # plt.ylim(bottom=1e-4)
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/hybrid_2nd_demo/' + demo + '_err.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    line_list = []
    for i in range(len(legends)):
        line, = plt.plot(res[i][1], color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=1)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=20)
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('Test Accuracy', fontsize=20)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/hybrid_2nd_demo/' + demo + '_acc.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()


if __name__ == '__main__':
    n_devices_by_cluster = [8, 8, 8, 8]
    # m_mean = 88000
    m_mean = 70400
    data_name = 'covtype'
    kappa = 10000
    global_iter = 50

    demo = 'hybrid_2nd_demo'
    legends = ['hybrid_2nd', 'local_2nd']


    # results = numpy.zeros((2, 2, global_iter+1))
    # server = Server(n_devices_by_cluster, m_mean, data_name, kappa=kappa, full=True, mode=HYBRID)
    # server.load_dataset()
    # res = server.training()
    # results[0][0] = res['func_error']
    # results[0][1] = res['acc']
    # server = Server(n_devices_by_cluster, m_mean, data_name, kappa=kappa, full=True, mode=LOCAL)
    # server.load_dataset()
    # res = server.training()
    # results[1][0] = res['func_error']
    # results[1][1] = res['acc']
    out_file_name = home_dir + 'Outputs/hybrid_2nd_demo/' + demo + '.npz'
    # numpy.savez(out_file_name, res=results)
    npz_file = numpy.load(out_file_name, allow_pickle=True)
    results = npz_file['res']
    # print(type(res))
    # print(res)
    plot_result(results, legends, demo)
