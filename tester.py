import torch
import pickle
import statistics
import matplotlib.pyplot as plt


import numpy as np
from mouvis.train import NeuralTrainJob
print(torch.cuda.is_available())

EPOCHS = 120
TRIALS = 10


def plot_oracle_data(epochs, trials, type):
    pred = []
    loo = []
    o_fracs = []

    connect_types = ['normal', 'golden', 'shuffle', 'scatter']

    assert type in connect_types, 'Invalid Connection-Type'
    idx = connect_types.index(type)

    for i in range(trials):
        job = NeuralTrainJob('store')
        config = {'data_config': {'seed': i + 100 * idx}}
        config = job.get_config(config)

        job.process(config, num_epochs=epochs)

        torch.cuda.empty_cache()

        model, pred_corrs, loo_corrs, o_frac = job.fetch_model(config)
        pred.append(pred_corrs)
        loo.append(loo_corrs)
        o_fracs.append(o_frac)
        print(o_fracs)

    print(o_fracs)
    x = list(np.arange(1, trials+1))

    _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, o_fracs, color="red")

    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Oracle Fraction')
    ax.set_title('Shuffle Performance on Oracle Trials')

    plt.show()
    return o_fracs


print(plot_oracle_data(EPOCHS, TRIALS, "scatter"))


performances = {}
performances['golden'] = [0.6898909275395828, 0.6755399381783108, 0.6855399381783108, 0.6928341258973828, 0.6865556698479608, 0.6920673312022132, 0.6920506202122719, 0.6835489839587763, 0.6757137076294014, 0.6782621114222429]
performances['normal'] = [0.6738887059547112, 0.6849047694595186, 0.6829935432111026, 0.6866169647262521, 0.6781422024848085, 0.6837444239005093, 0.6833226851534242, 0.6946403180976516, 0.6811048104794704, 0.6850212191887654]
performances['shuffle'] = [0.4986117944832542, 0.5351649641983518, 0.5115923981651825, 0.5088758593866048, 0.5079523303414862, 0.4912778592221663, 0.5042603857503126, 0.5112597462480785, 0.5193853754111828, 0.5056084920026916]
performances['scatter'] = [0.5309831162042373, 0.5432335048019404, 0.5234467815255955, 0.5087783308445244, 0.5160405582188833, 0.5131143663975528, 0.5179374827617276, 0.5305938609639955, 0.5461498072022611, 0.5581226080232572]

with open('perf_data.pickle', 'wb') as handle:
    pickle.dump(performances, handle, protocol=pickle.HIGHEST_PROTOCOL)

colors = ["gold", "red", "blue", "green"]
x = list(np.arange(1, 11))

gol = plt.scatter(x, performances["golden"], color=colors[0])
nor = plt.scatter(x, performances["normal"], color=colors[1])
shu = plt.scatter(x, performances["shuffle"], color=colors[2])
sca = plt.scatter(x, performances["scatter"], color=colors[3])

plt.legend((gol, nor, shu, sca), ('golden', 'normal', 'shuffle', 'scatter'), fontsize=8)
plt.title("Connect-type Performances on Oracle Trials")
plt.xlabel("Trial Number")
plt.ylabel("Oracle Fraction")
plt.show()


print(statistics.mean(performances['scatter']))
print(statistics.stdev(performances['scatter']))


