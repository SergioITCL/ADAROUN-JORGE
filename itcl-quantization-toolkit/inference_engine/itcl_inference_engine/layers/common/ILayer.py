
import numpy as np
from itcl_quantization.json.specification import Operator
import abc


class ILayer(metaclass=abc.ABCMeta):
    """ Layer Interface that wraps all of the TFLITE layer implementations under one interface type.
    """

    def __init__(self) -> None:
        """ Constructor for the ILayer interface.
        """
        pass

    @classmethod
    def from_model(cls, operator: Operator):
        """ Alternative Constructor of the layer.

        Builds the layer given a JSON Operator

        Args:
            operator (Operator): Json Operator with all the input and output tensors.

        Raises:
            NotImplemented: If the method is not implemented
        """
        raise NotImplemented

    def infer(self, input: np.ndarray) -> np.ndarray:
        """ Infer Method
        This method infers the input of the layer with the corresponding operation. 
        The shape of the np.ndarray must match the input shape of the layer. 

        Args:
            input (np.ndarray): Input tensor to be inferred 

        Raises:
            NotImplemented: 

        Returns:
            np.ndarray: The inferred output tensor
        """
        raise NotImplemented
