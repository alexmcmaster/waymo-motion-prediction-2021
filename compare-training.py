import sys

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import tensorflow as tf


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    #return ret[n - 1:] / n
    #return np.hstack((a[:n], ret[n:] / n))
    return ret / np.hstack((np.arange(n)+1, np.full(len(ret)-n, n)))


def reject_outliers(data, m = 2.):
    if isinstance(data, list):
        data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


def wrap_tf_iter(tf_iter):
    while True:
        try:
            yield tf_iter.next()
        except StopIteration:
            return
        except Exception as err:
            return


fig, ax = plt.subplots(1, 1)
ma_n = 20
xlim = int(sys.argv[1])
legend = list()
for i, logfile in enumerate(sys.argv[2:]):
    model_name = logfile.split("/")[1]
    lr_steps = list()
    lr = list()
    train_loss_steps = list()
    train_loss = list()
    dev_loss_steps = list()
    dev_loss = list()
    for summary in wrap_tf_iter(tf.compat.v1.train.summary_iterator(logfile)):
        s = summary.summary.value
        if len(s) < 1:
            print("Empty event:")
            print(summary)
            continue
        tag = s[0].tag
        val = s[0].simple_value
        if tag == "train/loss":
            train_loss_steps.append(summary.step)
            train_loss.append(val)
        elif tag == "lr":
            lr_steps.append(summary.step)
            lr.append(val)
        elif tag == "dev/loss":
            dev_loss_steps.append(summary.step)
            dev_loss.append(val)
        else:
            print(f"Unrecognized tag: {tag}")
    train_loss_ma = moving_average(train_loss, ma_n)
    ax.plot(train_loss_steps, train_loss_ma)
    legend.append(f"{model_name} -- Training Loss (MA{10*ma_n})")
    dev_loss_best = [min(dev_loss[:i+1]) for i in range(len(dev_loss))]
    # ^^ Painfully inefficient, but still fast at the scale we're dealing with.
    ax.plot(dev_loss_steps, dev_loss_best)
    legend.append(f"{model_name} -- Best Validation Loss")

ax.set_xlim(0, xlim)
ax.legend(legend)
plt.show()

