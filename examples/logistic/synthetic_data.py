import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(42)


target_bias = 2.0
target_slope = -1.0

y0_mu = [1., 1.]
y1_mu = [3., 3.]
yt_sig = [.5, .5]
yv_sig = [.8, .8]

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


def sample_alice_data(x_train, y_train, sz):
    if isinstance(sz, (tuple, list)):
        sz0, sz1 = sz
    else:
        sz0 = sz1 = sz // 2

    x0 = x_train[y_train == 0]
    x1 = x_train[y_train == 1]
    ix = np.arange(len(x_train))
    ix0 = ix[y_train == 0]
    ix1 = ix[y_train == 1]

    l2_10 = np.exp(np.exp(np.linalg.norm(y1_mu - x0, axis=-1)))
    l2_10 /= np.max(l2_10)
    l2_01 = np.exp(np.exp(np.linalg.norm(y0_mu - x1, axis=-1)))
    l2_01 /= np.max(l2_01)

    p0 = softmax(1 - l2_10)
    p1 = softmax(1 - l2_01)

    ix0_alice = np.random.choice(ix0, sz0, replace=False, p=p0)
    ix1_alice = np.random.choice(ix1, sz1, replace=False, p=p1)
    ix_alice = np.concatenate((ix0_alice, ix1_alice))

    msk = np.zeros(y_train.shape, dtype=bool)
    msk[ix_alice] = True
    x_alice, y_alice = x_train[msk], y_train[msk]
    x_bob, y_bob = x_train[~msk], y_train[~msk]

    return (x_alice, y_alice), (x_bob, y_bob)


if __name__ == "__main__":
    tparams = [(y0_mu, yt_sig), (y1_mu, yt_sig)]
    vparams = [(y0_mu, yv_sig), (y1_mu, yv_sig)]

    x_train, y_train = gen_dataset(*tparams, sz=train_size)
    x_test, y_test = gen_dataset(*vparams, sz=test_size)
    (x_alice, y_alice), (x_bob, y_bob) = sample_alice_data(x_train,
                                                           y_train,
                                                           alice_size)

    y_train = np.expand_dims(y_train, 1)
    y_test = np.expand_dims(y_test, 1)
    y_alice = np.expand_dims(y_alice, 1)
    y_bob = np.expand_dims(y_bob, 1)

    cols = ["x", "y", "label"]
    train = pd.DataFrame(np.concatenate((x_train, y_train), axis=-1),
                         columns=cols).to_csv(outpath + "train")
    test = pd.DataFrame(np.concatenate((x_test, y_test), axis=-1),
                        columns=cols).to_csv(outpath + "test")
    alice = pd.DataFrame(np.concatenate((x_alice, y_alice), axis=-1),
                         columns=cols).to_csv(outpath + "alice")
    bob = pd.DataFrame(np.concatenate((x_bob, y_bob), axis=-1),
                       columns=cols).to_csv(outpath + "bob")

    if visualize:
        plt.scatter(x=x_bob[:, 0], y=x_bob[:, 1], c=y_bob)
        plt.show()
        plt.scatter(x=x_train[:, 0], y=x_train[:, 1], c=y_train)
        plt.show()
        plt.scatter(x=x_test[:, 0], y=x_test[:, 1], c=y_test)
        plt.show()
