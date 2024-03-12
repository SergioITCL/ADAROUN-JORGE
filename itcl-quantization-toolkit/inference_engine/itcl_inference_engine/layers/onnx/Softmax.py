import unittest
import numpy as np

from itcl_inference_engine.layers.common.IOperator import IOperator


class Softmax(IOperator):
    """The operator computes normalized input as a softmax normalization
    """
    def __init__(self) -> None:
        super().__init__()

    def infer(self, input: np.ndarray) -> np.ndarray:
        """Normalize the input

        Softmax(input) = Exp(input) / ReduceSum(Exp(input))
        Args:
            input (np.ndarray): Input Tensor as float32 (Dequantized)

        Returns:
            np.ndarray: Float32 output tensor (Each value should be between 0 and 1)
        """


        return np.exp(input) / np.sum(np.exp(input))


