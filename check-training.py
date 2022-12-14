import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    #return ret[n - 1:] / n
    return ret / n


lr_steps = list()
lr = list()
train_loss_steps = list()
train_loss = list()
dev_loss_steps = list()
dev_loss = list()

for summary in tf.compat.v1.train.summary_iterator(sys.argv[1]):
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

ma_n = 200
train_loss_ma = moving_average(train_loss, ma_n)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(lr_steps, lr)
ax1.set_ylabel("Learning Rate")
ax2.plot(train_loss_steps, moving_average(train_loss, ma_n))
ax2.plot(dev_loss_steps, dev_loss)
ax2.set_xlabel("Training Step")
ax2.set_ylabel("Loss")
ax2.set_ylim(0, 1.1 * train_loss_ma[ma_n-1])
ax2.legend([f"Training Loss (MA{10*ma_n})", "Validation Loss"])
plt.show()
