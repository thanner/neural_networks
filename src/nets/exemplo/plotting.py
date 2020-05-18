import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def plot_error(vetor_error1, test):
    plt.figure(1)
    plt.plot(vetor_error1, label="Total Squared Error", linewidth=1.0, color="red")
    # plt.plot(vetor_error2, label="Median Squared Error", linewidth=1.0, color="blue")
    plt.ylabel("Total Squared Error")
    plt.xlabel("Epoch")
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, ncol=2, mode="expand", borderaxespad=0., prop={'size': 10})
    # plt.ylim([0, 1.02])
    plt.draw()
    name = test + '.png'
    plt.savefig(name)
    plt.clf()
    return
