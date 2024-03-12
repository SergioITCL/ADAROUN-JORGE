
import numpy as np
from itcl_inference_engine.layers.common.IOperator import IOperator

class Tanh(IOperator):
    """Tanh Operator
        https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh
    """
    def __init__(self) -> None:
        super().__init__()
    
    def infer(self, input: np.ndarray) -> np.ndarray:
        """Calculates the tanh for each element of the input array.

        Args:
            input (np.ndarray): an np.float32 tensor (Dequantized)

        Returns:
            np.ndarray fp32: Dequantized tensor Ouput
        """
        return np.array(np.tanh(input))

    


