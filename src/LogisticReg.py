import numpy as np

class LogisticRegression(object):
    def __init__(self, l1_coef, l2_coef):
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.w = None

    def fit(self, X, y, epochs=10, lr=0.1, batch_size=100):
        n, k = X.shape
        if self.w is None:
            self.w = np.random.randn(k + 1)

        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)

        losses = []

        for i in range(epochs):
            for X_batch, y_batch in self.generate_batches(X_train, y, batch_size):
                p = self._predict_proba(X_batch)
                losses.append(self._get_loss(y_batch, p))
                grad = self.get_grad(X_batch, y_batch, p)
                self.w -= X_batch.T @ (p - y_batch) / len(X_batch)
        return losses

    def get_grad(self, X_batch, y_batch, predictions):

        grad_basic = X_batch.T @ (predictions - y_batch) / len(X_batch)
        return grad_basic

    def predict_proba(self, X):
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return self.sigmoid(np.dot(X_, self.w))

    def _predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def sigmoid(h):
        return 1. / (1 + np.exp(-h))

    def _get_loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    def accuracy(y_pred, y):
        return sum(y_pred == y) / len(y)

    def generate_batches(X, y, batch_size):
        assert len(X) == len(y)
        X = np.array(X)
        y = np.array(y)

        perm =  np.random.permutation(len(X))

        for i in range(len(perm) // batch_size):
            idx = perm[i*batch_size:(i+1)*batch_size]
            curr_x = X[idx]
            curr_y = y[idx]
            yield curr_x, curr_y
