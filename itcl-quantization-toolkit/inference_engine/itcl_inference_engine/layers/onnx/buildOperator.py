

from itcl_quantization.json.specification import Operator

from itcl_inference_engine.layers.common.IOperator import IOperator
from itcl_inference_engine.layers.common.SigmoidLUT import SigmoidLUT
from itcl_inference_engine.layers.common.TanhLUT import TanhLUT
from itcl_inference_engine.layers.onnx.Softmax import Softmax
from itcl_inference_engine.layers.onnx.Tanh import Tanh
from itcl_inference_engine.layers.onnx.QLinearAdd import QLinearAdd
from itcl_inference_engine.layers.onnx.QLinearMatMul import QLinearMatMul
from itcl_inference_engine.layers.onnx.QuantizeLinear import QuantizeLinear
from itcl_inference_engine.layers.onnx.QLinearSigmoid import QLinearSigmoid
from itcl_inference_engine.layers.onnx.DequantizeLinear import DequantizeLinear


def build_operator(operator: Operator) -> IOperator:
    """Given an operator and a model with the the operators and initializers (Tensors)
    Builds the python version of the operator

    Args:
        operator (_type_): _description_
        model (_type_): _description_

    Raises:
        ValueError: If the operator is not compatible with the Inference Engine

    Returns:
        IOperator: An operator with all its quantization parameters and tensors loaded.
    """

    opt = operator["op_type"]

    if (opt == "QLinearAdd"):
        return QLinearAdd.from_model(operator,)

    elif (opt == "QLinearMatMul"):
        return QLinearMatMul.from_model(operator)
    elif (opt == "QuantizeLinear"):
        return QuantizeLinear.from_model(operator)
    elif (opt == "QLinearSigmoid"):
        return QLinearSigmoid.from_model(operator)
    elif (opt == "SigmoidLUT"):
        return SigmoidLUT.from_model(operator)
    elif (opt == "DequantizeLinear"):
        return DequantizeLinear.from_model(operator)
    elif (opt == "Softmax"):
        return Softmax()
    elif (opt == "Tanh"):
        return Tanh()
    elif (opt == "TanhLUT"):
        return TanhLUT.from_model(operator)
    else:
        raise ValueError(f"{opt} is not a valid operator")
