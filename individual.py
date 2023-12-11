from copy import deepcopy

import numpy as np

from fc import FC


class Individual:
    def __init__(self, dim_list: list[int] = [], **ga_operator_config):
        """
            dim_list: [ in_dim, hidden_dim1, hidden_dim2, ..., out_dim ]
        """

        self.layers: list[FC] = []
        self.loss: float = None
        self.acc: float = None
        self.age: int = 0

        for i in range(len(dim_list) - 1):
            in_dim = dim_list[i]
            out_dim = dim_list[i + 1]
            assert in_dim > 0

            if i < len(dim_list) - 2:
                # hidden layer
                activation = FC.ACT_RELU
            elif out_dim == 1:
                # output layer
                activation = FC.ACT_SIGMOID
            else:
                # output layer
                activation = FC.ACT_SOFTMAX

            fc = FC.build_from_dim(
                in_dim=in_dim,
                out_dim=out_dim,
                activation=activation,
                **ga_operator_config,
            )
            self.layers.append(fc)

    @property
    def fitness(self) -> float:
        return self.acc

    @property
    def in_dim(self) -> int:
        return self.layers[0].in_dim

    @property
    def out_dim(self) -> int:
        return self.layers[-1].out_dim

    def copy(self):
        ind = deepcopy(self)
        ind.age = 0
        return ind

    def predict(self, x: np.ndarray) -> float:
        res = np.copy(x)
        for layer in self.layers:
            res = layer.forward(res)

        return res[0]

    def increment_age(self):
        self.age += 1

    def set_acc(self, acc: float):
        assert self.acc is None  # accuracy should be calculated only once
        self.acc = acc

    def set_loss(self, loss: float):
        assert self.loss is None  # loss should be calculated only once
        self.loss = loss

    def add_layer(self, layer: FC):
        assert (len(self.layers) == 0) or (self.out_dim == layer.in_dim)
        self.layers.append(layer)

    def mutate(self):
        """update in place"""
        for fc in self.layers:
            fc.mutate()

    def recombine(self, other) -> tuple:
        assert isinstance(other, Individual)
        assert len(self.layers) == len(other.layers)

        child_a, child_b = Individual(), Individual()

        for self_layer, other_layer in zip(self.layers, other.layers):
            fc_a, fc_b = self_layer.recombine(other_layer)
            child_a.add_layer(fc_a)
            child_b.add_layer(fc_b)

        return child_a, child_b
