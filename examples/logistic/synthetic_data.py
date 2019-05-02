import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
np.random.seed(42)

y0_mu = [1., 1.]
y1_mu = [3., 3.]
yt_sig = [.25, .25]
yv_sig = [.65, .65]

train_size = 4000
test_size = 2000
alice_size = 2000

outpath = "./data/"
visualize = False
train = True


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


def sigmoid(x, grad=False):
    if grad:
        sig = sigmoid(x)
        return sig * (1 - sig)
    print(x[:4])
    y = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    print(y[:4])
    return y


def bce(y, y_hat):
    pos = y * np.log(y_hat)
    neg = (1 - y) * np.log(1 - y_hat)
    return -1 * (pos + neg)

class DataGen:
    def __init__(self, x_train, y_train, batch_size, shuffle=True):
        self.x, self.y = self.shuffle(x_train, y_train, shuffle)
        self.batch_size = batch_size
        self._i = 0
        self._stopping = len(x_train) // batch_size + 1

    def __len__(self):
        return self._stopping

    def __iter__(self):
        return self

    def next(self):
        if self._i == self._stopping:
            self._i = 0
            raise StopIteration

        try:
            b = self._i * self.batch_size
            e = (self._i + 1) * self.batch_size
            x, y = self.x[b:e], self.y[b:e]
        except:
            b = self._i * self.batch_size
            x, y = self.x[b:], self.y[b:]

        self._i += 1
        return x, y

    def shuffle(self, x, y, shuffle=True):
        if shuffle:
            data = np.concatenate((x, y), axis=-1)
            np.random.shuffle(data)
            x, y = data[:, :-1], data[:, -1]
        return x, np.expand_dims(y, -1)

class LogisticRegression:
    def __init__(self):
        self.w = np.random.uniform(-.01, .01, size=(2, 1))
        self.b = np.zeros((1,1))

    def __call__(self, x):
        logits = np.dot(x, self.w) + self.b
        # print("logit", logits[:10])
        y_hat = sigmoid(logits)
        # print("y_hat", y_hat[:10])
        return y_hat

    def compute_grad(self, dL, x):
        n = x.shape[0]
        dydw = np.dot(x.T, dL) / n
        dydb = np.mean(dL)
        return dydw, dydb

    def update(self, dydw, dydb, lr):
        self.w -= lr * dydw
        self.b -= lr * dydb


if __name__ == "__main__":
    tparams = [(y0_mu, yt_sig), (y1_mu, yt_sig)]
    vparams = [(y0_mu, yv_sig), (y1_mu, yv_sig)]

    x_bob, y_bob = gen_dataset(*tparams, sz=train_size)
    x_alice, y_alice = gen_dataset(*vparams, sz=alice_size)
    x_train = np.concatenate((x_bob, x_alice), axis=0)
    y_train = np.concatenate((y_bob, y_alice), axis=0)
    x_test, y_test = gen_dataset(*vparams, sz=test_size)

    y_test = np.expand_dims(y_test, 1)
    y_alice = np.expand_dims(y_alice, 1)
    y_bob = np.expand_dims(y_bob, 1)
    y_train = np.expand_dims(y_train, 1)

    # cols = ["x", "y", "label"]
    # test = pd.DataFrame(np.concatenate((x_test, y_test), axis=-1),
    #                     columns=cols).to_csv(outpath + "test")
    # alice = pd.DataFrame(np.concatenate((x_alice, y_alice), axis=-1),
    #                      columns=cols).to_csv(outpath + "alice")
    # bob = pd.DataFrame(np.concatenate((x_bob, y_bob), axis=-1),
    #                    columns=cols).to_csv(outpath + "bob")

    if visualize:
        plt.scatter(x=x_bob[:, 0], y=x_bob[:, 1], c=y_bob)
        plt.show()
        plt.scatter(x=x_alice[:, 0], y=x_alice[:, 1], c=y_alice)
        plt.show()
        plt.scatter(x=x_test[:, 0], y=x_test[:, 1], c=y_test)
        plt.show()
    if train:
        bob = True
        alice = True
        epochs = 20
        batch_size = 128
        learning_rate = .05
        model = LogisticRegression()

        if bob and not alice:
            datagen = DataGen(x_bob, y_bob, batch_size=batch_size)
        elif not bob and alice:
            datagen = DataGen(x_alice, y_alice, batch_size=batch_size)
        else:
            datagen = DataGen(x_train, y_train, batch_size=batch_size)


        for e in range(epochs):
            for i, (x, y) in enumerate(datagen):
                y_hat = model(x)
                # print(y_hat)
                loss = bce(y, y_hat)
                dL = y_hat - y
                grads = model.compute_grad(dL, x)
                model.update(*grads, lr=learning_rate)
                if i % 5 == 4:
                    msg = "Epoch: {e}. Batch: {batch}. Loss: {loss}."
                    print(msg.format(e=e, batch=i, loss=np.mean(dL)))

        y_hat = model(x_test)
        corr = np.sum(np.round(y_hat) == y_test)
        print("Accuracy: {acc}".format(acc = float(corr) / len(y_test)))
