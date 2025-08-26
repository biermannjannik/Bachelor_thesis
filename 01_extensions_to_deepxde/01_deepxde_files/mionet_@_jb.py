import torch

from .fnn import FNN
from .nn import NN
from .. import activations


import torch.nn as nn
from typing import List, Callable, Literal


class MIONetCartesianProd(NN):
    """MIONet with two input functions for Cartesian product format."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
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

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = activations.get(activation)
        if callable(layer_sizes_branch1[1]):
            # User-defined network
            self.branch1 = layer_sizes_branch1[1]
        else:
            # Fully connected network
            self.branch1 = FNN(
                layer_sizes_branch1, self.activation_branch1, kernel_initializer
            )
        if callable(layer_sizes_branch2[1]):
            # User-defined network
            self.branch2 = layer_sizes_branch2[1]
        else:
            # Fully connected network
            self.branch2 = FNN(
                layer_sizes_branch2, self.activation_branch2, kernel_initializer
            )
        if layer_sizes_merger is not None:
            self.activation_merger = activations.get(activation["merger"])
            if callable(layer_sizes_merger[1]):
                # User-defined network
                self.merger = layer_sizes_merger[1]
            else:
                # Fully connected network
                self.merger = FNN(
                    layer_sizes_merger, self.activation_merger, kernel_initializer
                )
        else:
            self.merger = None
        if layer_sizes_output_merger is not None:
            self.activation_output_merger = activations.get(activation["output merger"])
            if callable(layer_sizes_output_merger[1]):
                # User-defined network
                self.output_merger = layer_sizes_output_merger[1]
            else:
                # Fully connected network
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
        x_loc = inputs[2]
        # Branch net to encode the input function
        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        if self.merge_operation == "cat":
            x_merger = torch.cat((y_func1, y_func2), 1)
        else:
            if y_func1.shape[-1] != y_func2.shape[-1]:
                raise AssertionError(
                    "Output sizes of branch1 net and branch2 net do not match."
                )
            if self.merge_operation == "add":
                x_merger = y_func1 + y_func2
            elif self.merge_operation == "mul":
                x_merger = torch.mul(y_func1, y_func2)
            else:
                raise NotImplementedError(
                    f"{self.merge_operation} operation to be implimented"
                )
        # Optional merger net
        if self.merger is not None:
            y_func = self.merger(x_merger)
        else:
            y_func = x_merger
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        y_loc = self.trunk(x_loc)
        if self.trunk_last_activation:
            y_loc = self.activation_trunk(y_loc)
        # Dot product
        if y_func.shape[-1] != y_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of merger net and trunk net do not match."
            )
        # output merger net
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
        # Add bias
        y += self.b
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y
        

class MIONetCartesianProd_V2(NN):
    """
    Theory-based MIONet with Cartesian product input format.

    Supports:
    - Multiple input functions (via separate branch networks)
    - Evaluation at arbitrary locations via trunk network
    - Low-rank (default) and high-rank mode
    """

    def __init__(
        self,
        branch_nets: List[nn.Module],
        trunk_net: nn.Module,
        mode: Literal["low-rank", "high-rank"] = "low-rank",
        output_merger: Callable[[torch.Tensor], torch.Tensor] = None,
        use_bias: bool = True,
    ):
        super().__init__()
        self.branch_nets = nn.ModuleList(branch_nets)
        self.trunk_net = trunk_net
        self.mode = mode
        self.output_merger = output_merger
        self.bias = nn.Parameter(torch.zeros(1)) if use_bias else None

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: [x_func1, x_func2, x_loc]
                - x_func1: [B, d1] input function 1
                - x_func2: [B, d2] input function 2
                - x_loc:   [M, d_y] evaluation locations

        Returns:
            y: [B, M] matrix of evaluations
        """
        x_func1, x_func2, x_loc = inputs
        B = x_func1.shape[0]
        M = x_loc.shape[0]

        # Pass through branch networks
        y_func1 = self.branch_nets[0](x_func1)  # [B, d_t]
        y_func2 = self.branch_nets[1](x_func2)  # [B, d_t]

        # Combine function encodings (here: element-wise multiplication)
        merged_func = y_func1 * y_func2  # [B, d_t]

        # Encode locations via trunk
        y_loc = self.trunk_net(x_loc)  # [M, d_t]

        if self.mode == "high-rank":
            # Dot product between merged_func and y_loc
            y = torch.matmul(merged_func, y_loc.T)  # [B, M]

        elif self.mode == "low-rank":
            # Prepare tensors for element-wise interaction
            merged_func = merged_func[:, None, :]  # [B, 1, d_t]
            y_loc = y_loc[None, :, :]              # [1, M, d_t]

            product = merged_func * y_loc          # [B, M, d_t]

            if self.output_merger is not None:
                y = self.output_merger(product.view(B * M, -1)).view(B, M)  # [B, M]
            else:
                y = product.sum(dim=-1)  # [B, M]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # Optional bias
        if self.bias is not None:
            y = y + self.bias

        # Optional output transform hook
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)

        return y


    def forward2(self, inputs: List[torch.Tensor], x_loc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: List of tensors [B, d_i] for each input function v_i
            x_loc: [M, d_y] evaluation locations

        Returns:
            [B, M] tensor of function evaluations
        """
        B = inputs[0].shape[0]
        M = x_loc.shape[0]

        # Encode location
        y_loc = self.trunk_net(x_loc)  # [M, d_t]

        # Encode each input function with its branch net
        branch_outputs = [net(v_i) for net, v_i in zip(self.branch_nets, inputs)]  # each [B, d_t]

        if self.mode == "high-rank":
            # Elementwise multiply all branch outputs
            merged_func = branch_outputs[0]
            for out in branch_outputs[1:]:
                merged_func = merged_func * out  # [B, d_t]

            # Output via dot product
            y = torch.matmul(merged_func, y_loc.T)  # [B, M]

        elif self.mode == "low-rank":
            # Compute elementwise product across inputs and location features
            merged_func = branch_outputs[0][:, None, :]  # [B, 1, d_t]
            for out in branch_outputs[1:]:
                merged_func = merged_func * out[:, None, :]  # [B, 1, d_t]
            y_loc = y_loc[None, :, :]  # [1, M, d_t]
            product = merged_func * y_loc  # [B, M, d_t]

            # Optionally apply projection
            if self.output_merger is not None:
                y = self.output_merger(product.view(B * M, -1)).view(B, M)
            else:
                y = product.sum(dim=-1)  # sum over d_t → [B, M]

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if self.bias is not None:
            y = y + self.bias

        return y


class MIONetCartesianProd_3Branches(NN):
    """MIONet mit drei Eingabefunktionen (Branches) im Cartesian-Produkt-Format."""

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





class PODMIONet(NN):
    """MIONet with two input functions and proper orthogonal decomposition (POD)
    for Cartesian product format."""

    def __init__(
        self,
        pod_basis,
        layer_sizes_branch1,
        layer_sizes_branch2,
        activation,
        kernel_initializer,
        layer_sizes_trunk=None,
        regularization=None,
        trunk_last_activation=False,
        merge_operation="mul",
        layer_sizes_merger=None,
    ):
        super().__init__()

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
            self.activation_merger = activations.get(activation["merger"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = activations.get(activation)
        self.pod_basis = torch.as_tensor(pod_basis, dtype=torch.float32)
        if callable(layer_sizes_branch1[1]):
            # User-defined network
            self.branch1 = layer_sizes_branch1[1]
        else:
            # Fully connected network
            self.branch1 = FNN(
                layer_sizes_branch1, self.activation_branch1, kernel_initializer
            )
        if callable(layer_sizes_branch2[1]):
            # User-defined network
            self.branch2 = layer_sizes_branch2[1]
        else:
            # Fully connected network
            self.branch2 = FNN(
                layer_sizes_branch2, self.activation_branch2, kernel_initializer
            )
        if layer_sizes_merger is not None:
            if callable(layer_sizes_merger[1]):
                # User-defined network
                self.merger = layer_sizes_merger[1]
            else:
                # Fully connected network
                self.merger = FNN(
                    layer_sizes_merger, self.activation_merger, kernel_initializer
                )
        else:
            self.merger = None
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = FNN(
                layer_sizes_trunk, self.activation_trunk, kernel_initializer
            )
            self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.trunk_last_activation = trunk_last_activation
        self.merge_operation = merge_operation

    def forward(self, inputs):
        x_func1 = inputs[0]
        x_func2 = inputs[1]
        x_loc = inputs[2]
        # Branch net to encode the input function
        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        # connect two branch outputs
        if self.merge_operation == "cat":
            x_merger = torch.cat((y_func1, y_func2), 1)
        else:
            if y_func1.shape[-1] != y_func2.shape[-1]:
                raise AssertionError(
                    "Output sizes of branch1 net and branch2 net do not match."
                )
            if self.merge_operation == "add":
                x_merger = y_func1 + y_func2
            elif self.merge_operation == "mul":
                x_merger = torch.mul(y_func1, y_func2)
            else:
                raise NotImplementedError(
                    f"{self.merge_operation} operation to be implimented"
                )
        # Optional merger net
        if self.merger is not None:
            y_func = self.merger(x_merger)
        else:
            y_func = x_merger
        # Dot product
        if self.trunk is None:
            # POD only
            y = torch.einsum("bi,ni->bn", y_func, self.pod_basis)
        else:
            y_loc = self.trunk(x_loc)
            if self.trunk_last_activation:
                y_loc = self.activation_trunk(y_loc)
            y = torch.einsum("bi,ni->bn", y_func, torch.cat((self.pod_basis, y_loc), 1))
            y += self.b
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y
