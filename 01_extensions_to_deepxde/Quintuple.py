from .data import Data
from .sampler import BatchSampler


class Quintuple(Data):
    """Dataset with each data point as a quintuple.

    The tuple of the first four elements are the input, and the fifth element is the
    output. This dataset can be used with the network ``MIONetCatesianProd_3Branches`` for operator
    learning.

    Args:
        X_train: A tuple of four NumPy arrays.
        y_train: A NumPy array.
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test

        self.train_sampler = BatchSampler(len(self.train_y), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (
            (self.train_x[0][indices], self.train_x[1][indices], self.train_x[2][indices]),
            self.train_x[3],
            self.train_y[indices],
        )

    def test(self):
        return self.test_x, self.test_y
