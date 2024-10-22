import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def violinplot(values,metric_name,labels, filename="plot.png",value_lim=(0,1)):
    """
    Compute and save the violinplots for each array of values
    Arguments:
        values (2D np.ndarray): np.array which contains array of values, one for each labels
        metric_name (str): Name of the values metric
        labels (list of str):  label associated to each array of value 
        filename (str): filename of the saved figure
        value_lim (tuple): ymin and ymax on the figure's scale
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))


    axs.violinplot(values,
                  showmeans=False,
                  showmedians=True)
    axs.yaxis.grid(True)
    axs.set_ylim(value_lim[0],value_lim[1])
    axs.set_xticks([y + 1 for y in range(len(values))],
                labels=labels)
    axs.set_xlabel('label')
    axs.set_ylabel(metric_name)
    
    plt.savefig(filename)

def distribution_plot(values,metric_name,filename="plot.png"):
    """
    Compute and save the distribution_plot for an array of values
    Arguments:
        values (np.array): values for which the distribution is computed
        metric_name (str): Name of the values metric
        filename (str): filename of the saved figure
    """

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

    sns.kdeplot(np.array(values),ax=axs)
    mean_value=np.mean(values)
    axs.axvline(mean_value, c='red', ls='--', lw=1.5,label=f"mean={mean_value: .3g}")
    axs.set_xlabel(metric_name)
    axs.set_ylabel("")
    axs.legend()
    
    plt.savefig(filename)