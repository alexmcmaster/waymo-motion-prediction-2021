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


lr_steps = list()
lr = list()
train_loss_steps = list()
train_loss = list()
dev_loss_steps = list()
dev_loss = list()

for i, logfile in enumerate(sys.argv[1:]):
    lr_step_offset = 0
    train_loss_step_offset = 0
    dev_loss_step_offset = 0
    for summary in tf.compat.v1.train.summary_iterator(logfile):
        s = summary.summary.value
        if len(s) < 1:
            print("Empty event:")
            print(summary)
            continue
        tag = s[0].tag
        val = s[0].simple_value
        if tag == "train/loss":
            if i > 0 and train_loss_step_offset == 0:
                train_loss_step_offset = train_loss_steps[-2] - summary.step
            train_loss_steps.append(summary.step + train_loss_step_offset)
            train_loss.append(val)
        elif tag == "lr":
            if i > 0 and lr_step_offset == 0:
                lr_step_offset = lr_steps[-2] - summary.step
            lr_steps.append(summary.step + lr_step_offset)
            lr.append(val)
        elif tag == "dev/loss":
            if i > 0 and dev_loss_step_offset == 0:
                dev_loss_step_offset = dev_loss_steps[-2] - summary.step
            dev_loss_steps.append(summary.step + dev_loss_step_offset)
            dev_loss.append(val)
        else:
            print(f"Unrecognized tag: {tag}")


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(lr_steps, lr)

ax1.set_title("MotionCNN Training Progress")
ax1.set_ylabel("Learning Rate")

legend = list()
ax2.plot(train_loss_steps, train_loss)
legend.append("Training Loss")
try:
    ma_n = 200
    train_loss_ma = moving_average(train_loss, ma_n)
    ax2.plot(train_loss_steps, train_loss_ma)
    legend.append(f"Training Loss (MA{10*ma_n})")
except Exception as err:
    print(f"Failed to plot train_loss_ma: {err}")
ax2.plot(dev_loss_steps, dev_loss)
legend.append("Validation Loss")
try:
    dev_loss_best = [min(dev_loss[:i+1]) for i in range(len(dev_loss))]
    # ^^ Painfully inefficient, but still fast at the scale we're dealing with.
    ax2.plot(dev_loss_steps, dev_loss_best)
    legend.append("Best Validation Loss")
except Exception as err:
    print(f"Failed to plot dev_loss_best: {err}")

ax2.set_xlabel("Training Step")
ax2.set_ylabel("Loss")
try:
    ax2.set_ylim(0, 1.5 * max(reject_outliers(dev_loss)))
except Exception as err:
    print(f"Failed to set ylim: {err}")

ax2.legend(legend)

plt.show()
