import numpy as np


class FullyConnectLayer:

    def __init__(self):
        self.weight = 0
        self.bias = 0
        pass

    def forward(self, input):
        self.input = input
        self.output = np.matmul(input, self.weight) + self.bias

        return self.output
