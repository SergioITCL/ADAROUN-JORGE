import numpy as np
from itcl_inference_engine.layers.common.ILayer import ILayer
from itcl_quantization.json.specification import Operator


class SoftMax(ILayer):
    """
    SOFTMAX
    Input 0:
        data_type  : int8
        range      : [-128, 127]
        granularity: per-tensor
    Output 0:
        data_type  : int8
        range      : [-128, 127]
        granularity: per-tensor
        restriction: (scale, zero_point) = (1.0 / 256.0, -128)
    """

    def __init__(self, input_scale, input_zerop) -> None:
        super().__init__()

    @classmethod
    def from_model(cls, layer: Operator):
        return cls(layer["inputs"][0]["scale"], layer["inputs"][0]["zero_point"])

    def infer(self, input: np.ndarray):
        return input
