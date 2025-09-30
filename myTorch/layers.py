import numpy as np
import time


class FullyConnectLayer:
    def __init__(self, num_input: int, num_output: int):
        self.num_input = num_input
        self.num_output = num_output
        print(
            "\tFully connected layer with input %d, output %d."
            % (self.num_input, self.num_output)
        )

    def init_param(self, std=0.01):
        self.weight = np.random.normal(
            loc=0.0, scale=std, size=(self.num_input, self.num_output)
        )

    def forward(self, input):
        start_time = time.time()
        self.input = input
        print("start_time was %d" % (start_time))

        output = np.matmul(input, self.weight) + self.bias
        return output

    def backward(self, top_diff):
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0)
        self.bottom_diff = np.dot(top_diff, self.weight.T)

        return self.bottom_diff

    def update_param(self, lr):
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def save_param(self):
        return self.weight


class ReLULayer:
    def __init__(self, num_input: int, num_output: int):
        self.num_input = num_input
        self.num_output = num_output
        print(
            "\tReLU Function layer with input %d, output %d."
            % (self.num_input, self.num_output)
        )

    def forward(self, input):
        self.input = input

        output = np.maximun(0, input)
        return output

    def backward(self, top_diff):
        bottom_diff = top_diff
        bottom_diff[self.input < 0] = 0
        return bottom_diff
