import numpy as np
from itcl_inference_engine.layers.common.ILayer import ILayer
from itcl_quantization.json.specification import Operator

class RELU(ILayer):


    def __init__(self, input_scale: float, input_zp: int, output_scale: float, output_zp: int) -> None:
        super().__init__()
        self.input_scale = input_scale
        self.input_zp = input_zp
        self.output_scale = output_scale
        self.output_zp = output_zp
        self.k = input_scale / output_scale
    @classmethod
    def from_model(cls, layer: Operator):

        input = layer["inputs"][0]
        output = layer["outputs"][0]

        

        return cls(input["scale"], input["zero_point"], output["scale"], output["zero_point"]) # type: ignore

    def infer(self, input: np.ndarray):
        """
        Applies the relu function to a quantized input tensor
        
        :param input: the input tensor
        :type input: np.ndarray
        :return: The output of the relu function
        """
        
        y = (input.astype(np.int16) ).clip(min=self.input_zp)

        relu = (self.output_zp + self.k  * (y - self.output_zp)).astype(np.int8)

        return relu 