import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
np.random.seed(42)


target_bias = 2.0
target_slope = -1.0

y0_mu = [1., 1.]
y1_mu = [3., 3.]
yt_sig = [.3, .3]
yv_sig = [.6, .6]

train_size = 4000
test_size = 2000
alice_size = 2000

outpath = "./data/"
visualize = True


def gen_class_data(mu, sig, sz, label):
    feats = [np.random.normal(loc=m, scale=s, size=sz)
             for m, s in zip(mu, sig)]
    x = np.array(feats).reshape(sz, -1)
    if label == 0:
        return x, np.zeros((sz,))
    elif label == 1:
        return x, np.ones((sz,))
    msg = "Inappropriate label value {label} for binary classification."
    raise ValueError(msg.format(label=label))


def gen_dataset(y0_params, y1_params, sz):
    if isinstance(sz, (tuple, list)):
        sz0, sz1 = sz
    else:
        sz0 = sz1 = sz // 2

    x0, y0 = gen_class_data(*y0_params, sz=sz0, label=0)
    x1, y1 = gen_class_data(*y1_params, sz=sz1, label=1)

    x = np.concatenate((x0, x1))
    y = np.concatenate((y0, y1))

    return x, y


def softmax(x, axis=None):
    if not np.all(x < 0):
        x = x - np.max(x)
    num = np.exp(x)
    den = np.sum(num, axis=axis)
    return num / den


if __name__ == "__main__":
    tparams = [(y0_mu, yt_sig), (y1_mu, yt_sig)]
    vparams = [(y0_mu, yv_sig), (y1_mu, yv_sig)]

    x_bob, y_bob = gen_dataset(*tparams, sz=train_size)
    x_alice, y_alice = gen_dataset(*vparams, sz=alice_size)
    x_test, y_test = gen_dataset(*vparams, sz=test_size)

    y_train = np.expand_dims(y_train, 1)
    y_test = np.expand_dims(y_test, 1)
    y_alice = np.expand_dims(y_alice, 1)
    y_bob = np.expand_dims(y_bob, 1)

    cols = ["x", "y", "label"]
    test = pd.DataFrame(np.concatenate((x_test, y_test), axis=-1),
                        columns=cols).to_csv(outpath + "test")
    alice = pd.DataFrame(np.concatenate((x_alice, y_alice), axis=-1),
                         columns=cols).to_csv(outpath + "alice")
    bob = pd.DataFrame(np.concatenate((x_bob, y_bob), axis=-1),
                       columns=cols).to_csv(outpath + "bob")

    if visualize:
        plt.scatter(x=x_bob[:, 0], y=x_bob[:, 1], c=y_bob)
        plt.show()
        plt.scatter(x=x_alice[:, 0], y=x_alice[:, 1], c=y_alice)
        plt.show()
        plt.scatter(x=x_test[:, 0], y=x_test[:, 1], c=y_test)
        plt.show()
