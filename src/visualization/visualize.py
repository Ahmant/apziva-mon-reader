import matplotlib.pyplot as plt


def plot_model_metrics(data, title, xlabel, ylabel, legends = [], save_path = None):
    # fig = plt.gcf()

    for row in data:
        plt.plot(row)

    plt.axis(ymin=0.4,ymax=1)
    plt.grid()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.legend(legends)
    
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
