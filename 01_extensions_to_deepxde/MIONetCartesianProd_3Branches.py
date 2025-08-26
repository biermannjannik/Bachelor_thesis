import torch
from .fnn import FNN
from .nn import NN
from .. import activations
import torch.nn as nn
from typing import List, Callable, Literal

class MIONetCartesianProd_3Branches(NN):
    """MIONet with 3 input functions and independant branch networks for with a single Cartesian product (B1,B2,B3)xT"""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_branch3,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
        trunk_last_activation=False,
        merge_operation="mul",
        layer_sizes_merger=None,
        output_merge_operation="mul",
        layer_sizes_output_merger=None,
    ):
        super().__init__()

        # Aktivierungen für jeden Branch und Trunk
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_branch3 = activations.get(activation["branch3"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = activations.get(activation)
            self.activation_branch2 = activations.get(activation)
            self.activation_branch3 = activations.get(activation)
            self.activation_trunk = activations.get(activation)


        # Branch 1
        if callable(layer_sizes_branch1[1]):
            self.branch1 = layer_sizes_branch1[1]
        else:
            self.branch1 = FNN(layer_sizes_branch1, self.activation_branch1, kernel_initializer)

        # Branch 2
        if callable(layer_sizes_branch2[1]):
            self.branch2 = layer_sizes_branch2[1]
        else:
            self.branch2 = FNN(layer_sizes_branch2, self.activation_branch2, kernel_initializer)

        # Branch 3
        if callable(layer_sizes_branch3[1]):
            self.branch3 = layer_sizes_branch3[1]
        else:
            self.branch3 = FNN(layer_sizes_branch3, self.activation_branch3, kernel_initializer)

        # Merger-Netz (optional)
        if layer_sizes_merger is not None:
            self.activation_merger = activations.get(activation["merger"])
            if callable(layer_sizes_merger[1]):
                self.merger = layer_sizes_merger[1]
            else:
                self.merger = FNN(layer_sizes_merger, self.activation_merger, kernel_initializer)
        else:
            self.merger = None

        # Output Merger (optional)
        if layer_sizes_output_merger is not None:
            self.activation_output_merger = activations.get(activation["output merger"])
            if callable(layer_sizes_output_merger[1]):
                self.output_merger = layer_sizes_output_merger[1]
            else:
                self.output_merger = FNN(
                    layer_sizes_output_merger,
                    self.activation_output_merger,
                    kernel_initializer,
                )
        else:
            self.output_merger = None

        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.trunk_last_activation = trunk_last_activation
        self.merge_operation = merge_operation
        self.output_merge_operation = output_merge_operation

    def forward(self, inputs):
        x_func1 = inputs[0]
        x_func2 = inputs[1]
        x_func3 = inputs[2]
        x_loc = inputs[3]

        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        y_func3 = self.branch3(x_func3)

        # Branch Outputs zusammenführen
        if self.merge_operation == "cat":
            x_merger = torch.cat((y_func1, y_func2, y_func3), dim=1)
        else:
            if not (y_func1.shape[-1] == y_func2.shape[-1] == y_func3.shape[-1]):
                raise AssertionError("Output sizes der drei Branches stimmen nicht überein.")
            if self.merge_operation == "add":
                x_merger = y_func1 + y_func2 + y_func3
            elif self.merge_operation == "mul":
                x_merger = y_func1 * y_func2 * y_func3
            else:
                raise NotImplementedError(f"{self.merge_operation} Operation nicht implementiert.")

        if self.merger is not None:
            y_func = self.merger(x_merger)
        else:
            y_func = x_merger

        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)

        y_loc = self.trunk(x_loc)
        if self.trunk_last_activation:
            y_loc = self.activation_trunk(y_loc)

        if y_func.shape[-1] != y_loc.shape[-1]:
            raise AssertionError("Output-Größen von Merger-Netz und Trunk-Netz stimmen nicht überein.")

        # Output Merger
        if self.output_merger is None:
            y = torch.einsum("ip,jp->ij", y_func, y_loc)
        else:
            y_func = y_func[:, None, :]
            y_loc = y_loc[None, :]
            if self.output_merge_operation == "mul":
                y = torch.mul(y_func, y_loc)
            elif self.output_merge_operation == "add":
                y = y_func + y_loc
            elif self.output_merge_operation == "cat":
                y_func = y_func.repeat(1, y_loc.shape[1], 1)
                y_loc = y_loc.repeat(y_func.shape[0], 1, 1)
                y = torch.cat((y_func, y_loc), dim=2)
            shape0 = y.shape[0]
            shape1 = y.shape[1]
            y = y.reshape(shape0 * shape1, -1)
            y = self.output_merger(y)
            y = y.reshape(shape0, shape1)

        y += self.b
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y

