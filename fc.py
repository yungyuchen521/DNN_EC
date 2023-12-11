from typing import Callable

import numpy as np
from tensorflow.keras.initializers import GlorotUniform


class FC:
    ACT_RELU = "relu"
    ACT_SIGMOID = "sigmoid"
    ACT_SOFTMAX = "softmax"

    BIAS_MU = 0
    BIAS_SIGMA = 0.1

    def __init__(
        self,
        wgts: np.ndarray,
        bias: np.ndarray,
        activation: str,
        **config, # callbacks of GA operators
    ):
        assert len(wgts.shape) == 2
        assert len(bias.shape) == 1
        assert wgts.shape[1] == bias.shape[0]
        assert activation in (self.ACT_RELU, self.ACT_SIGMOID, self.ACT_SOFTMAX)

        self.wgts: np.ndarray = wgts # in_dim X out_dim
        self.bias: np.ndarray = bias # out_dim
        self.activation: str = activation

        self.wgts_crossover_callback: Callable = config["wgts_crossover_callback"]
        self.wgts_mutate_callback: Callable = config["wgts_mutate_callback"]
        self.bias_crossover_callback: Callable = config["bias_crossover_callback"]
        self.bias_mutate_callback: Callable = config["bias_mutate_callback"]

    @classmethod
    def build_from_dim(cls, in_dim: int, out_dim: int, activation: str, **config):
        initializer = GlorotUniform()
        wgts = np.array(initializer(shape=[in_dim, out_dim]))
        bias = np.random.normal(cls.BIAS_MU, cls.BIAS_SIGMA, size=out_dim)

        return FC(wgts, bias, activation, **config)

    @property
    def config(self) -> dict:
        return {
            "wgts_crossover_callback": self.wgts_crossover_callback,
            "wgts_mutate_callback": self.wgts_mutate_callback,
            "bias_crossover_callback": self.bias_crossover_callback,
            "bias_mutate_callback": self.bias_mutate_callback,
        }

    @property
    def in_dim(self) -> int:
        return self.wgts.shape[0]

    @property
    def out_dim(self) -> int:
        return self.wgts.shape[1]

    def forward(self, x: np.ndarray) -> np.ndarray:
        res = np.matmul(x, self.wgts)
        res += self.bias

        if self.activation == self.ACT_RELU:
            res = self.relu(res)
        elif self.activation == self.ACT_SIGMOID:
            res = self.sigmoid(res)
        else:
            res = self.softmax(res)

        return res

    def mutate(self):
        """ update in place """
        self.wgts_mutate_callback(self.wgts)
        self.bias_mutate_callback(self.bias)

    def recombine(self, other) -> tuple:
        assert isinstance(other, FC)

        wgts1, wgts2 = self.wgts_crossover_callback(self.wgts, other.wgts)
        bias1, bias2 = self.bias_crossover_callback(self.bias, other.bias)

        child1 = FC(wgts1, bias1, self.activation, **self.config)
        child2 = FC(wgts2, bias2, self.activation, **self.config)

        return child1, child2
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 1
        return x * (x > 0)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 1
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 1
        return np.exp(x) / sum(np.exp(x))
