import matplotlib.pyplot as plt
import numpy as np
import ast

epochs = 20


def average_dicts(dicts):
    """
    averages a list of dictionaries. Each dictionary is assumed to have the same number of keys. Returns a
    single dictionary that is the average of every dictionary in dicts.
    """

    avg_dict = {}

    num = len(dicts)
    for key in dicts[0]:
        curr = 0
        for i in range(num):
            curr += dicts[i][key]
        avg_dict[key] = curr / num

    return [avg_dict]


def collapse(dict):
    list = []
    for i in dict:
        for j in range(dict[i]):
            list.append(i)
    return list


def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals


def _plot_dict_line(d, color, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        plt.plot(xvals, yvals, color=color, label=label)
    else:
        plt.plot(xvals, yvals)


def plot_lines(data, title, xlabel, ylabel, labels=None, colors = None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments:
    data     -- a list of dictionaries, each of which will be plotted
                as a line with the keys on the x-axis and the values on
                the y-axis.
    title    -- title label for the plot
    xlabel   -- x-axis label for the plot
    ylabel   -- y-axis label for the plot
    labels   -- optional dictionary of string-int pairs that represent the labels in the plot.
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    # Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    # Create a new figure
    fig = plt.figure()

    # Plot the data
    if labels:

        if sum(labels.values()) != len(data):
            msg = "labels must have the same number of values as data"
            raise ValueError(msg)
        else:
            for d, l in zip(data, collapse(labels)):
                _plot_dict_line(d, color=colors[l], label=l)
        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)
        plt.legend(newHandles, newLabels)
        plt.setp(plt.gca().get_legend().get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    # Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(mins)
    plt.ylim(ymin=ymin)

    plt.xticks(np.arange(1, epochs, 1))

    # Label the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Draw grid lines
    plt.grid(True)

    # Show the plot
    fig.show()

    # Save to file
    if filename:
        plt.savefig(filename)


f = open("normal_data.txt", "r")
gold_data = ast.literal_eval(f.read())

f = open("shuffle_data.txt", "r")
shuffle_data = ast.literal_eval(f.read())

f = open("scatter_data.txt", "r")
scatter_data = ast.literal_eval(f.read())


golden = average_dicts(gold_data[0: 10])
print(golden)
normal = average_dicts(gold_data[10: 20])
shuffle = average_dicts(shuffle_data[10: 20])
scatter = average_dicts(scatter_data[10: 20])

data = golden + normal + shuffle + scatter
print(data)
labels = {"Standard": 1, "Sparse Normal": 1, "Sparse Shuffle": 1, "Sparse Scatter": 1}
colors = {"Standard": "red", "Sparse Normal": "blue", "Sparse Shuffle": "green", "Sparse Scatter": "purple"}

plot_lines(data, "Combined Average Plots", "Epoch", "Accuracy", labels,
           colors, "combined_avg_plots.png")