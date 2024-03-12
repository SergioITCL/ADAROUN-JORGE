from typing import List, Union
from itcl_quantization.json.specification import Operator
import numpy as np
from itcl_quantization.quantization.lut import ReducedLUT
from itcl_inference_engine.layers.common.IOperator import IOperator


class Lut_activation(IOperator):
    """Tanh Operator
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh
    """

    def __init__(self, LUT: Union[List[int], ReducedLUT], offset=0) -> None:
        super().__init__()
        self.__LUT = LUT
        self.__offset = offset
        lut = lambda x: self.__LUT[x]

        self.__apply_lut_vectorize = np.vectorize(lut)

    @classmethod
    def from_model(cls, operator: Operator):
        """Buils the operator from a given node

        Args:
            operator (Operator): Json Operator Dict

        Returns:
            TanhLUT Instance
        """

        i = operator["inputs"]
        node = i[0]
        if node is None:
            raise ValueError("Missing input")
        lut = node["LUT"]

        if lut is None:
            raise ValueError("Missing LUT")

        # Use the Reduced LUT if it is available
        rl = lut.get("reduced_LUT")
        if rl is not None:
            LUT = ReducedLUT.deserialize(rl)
        else:
            LUT = lut["LUT"] or []

        offset = lut.get("offset") or 0

        return cls(LUT, offset)

    def infer(self, input: np.ndarray) -> np.ndarray:
        """Calculates the tanh for each element of the input array.

        Args:
            input (np.ndarray): an np.float32 tensor (Dequantized)

        Returns:
            np.ndarray fp32: Dequantized tensor Ouput
        """
        return self.__apply_lut_vectorize(input + self.__offset)
