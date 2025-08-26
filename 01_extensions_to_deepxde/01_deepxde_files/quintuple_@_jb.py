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
    

class QuintupleCartesianProd(Data):
    """Cartesian Product input data format for extended MIONet architecture.

    This dataset can be used with networks that expect three inputs of shape (N1, ·),
    one of shape (N2, ·), and an output of shape (N1, N2).

    Args:
        X_train: A tuple of four NumPy arrays.
            - First three elements: shape (`N1`, `dim_i`)
            - Fourth element: shape (`N2`, `dim4`)
        y_train: A NumPy array of shape (`N1`, `N2`)
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        if (
            len(X_train[0]) != y_train.shape[0]
            or len(X_train[1]) != y_train.shape[0]
            or len(X_train[2]) != y_train.shape[0]
            or len(X_train[3]) != y_train.shape[1]
        ):
            raise ValueError("Training dataset shape does not match quintuple cartesian format.")
        if (
            len(X_test[0]) != y_test.shape[0]
            or len(X_test[1]) != y_test.shape[0]
            or len(X_test[2]) != y_test.shape[0]
            or len(X_test[3]) != y_test.shape[1]
        ):
            raise ValueError("Testing dataset shape does not match quintuple cartesian format.")

        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = BatchSampler(len(X_train[3]), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            return (
                self.train_x[0][indices],
                self.train_x[1][indices],
                self.train_x[2][indices],
                self.train_x[3],
            ), self.train_y[indices]
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        return (
            self.train_x[0][indices_branch],
            self.train_x[1][indices_branch],
            self.train_x[2][indices_branch],
            self.train_x[3][indices_trunk],
        ), self.train_y[indices_branch][:, indices_trunk]

    def test(self):
        return self.test_x, self.test_y
