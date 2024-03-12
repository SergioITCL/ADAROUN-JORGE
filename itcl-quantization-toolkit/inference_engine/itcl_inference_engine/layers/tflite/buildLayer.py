from itcl_inference_engine.layers.common.ILayer import ILayer
from itcl_quantization.json.specification import Operator
from itcl_inference_engine.layers.common.SigmoidLUT import SigmoidLUT
from itcl_inference_engine.layers.tflite.Dequantize import Dequantize
from itcl_inference_engine.layers.tflite.Quantize import Quantize
from itcl_inference_engine.layers.tflite.FullyConnected import FullyConnected
from itcl_inference_engine.layers.tflite.Logistic import Sigmoid
from itcl_inference_engine.layers.tflite.SoftMax import SoftMax
from itcl_inference_engine.layers.tflite.Tanh import Tanh
from itcl_inference_engine.layers.common.TanhLUT import TanhLUT
from itcl_inference_engine.layers.common.Relu import RELU
from itcl_inference_engine.util.checkLayerType import LayerType 

def build_layer(layer: Operator) -> ILayer:
        layer_type = layer["op_type"]

        layer_builder = None
        if LayerType.isTanH(layer_type):
            layer_builder = Tanh
        elif LayerType.isTanHLUT(layer_type):
            
            layer_builder = TanhLUT
        elif LayerType.isFullyConnected(layer_type):
            layer_builder = FullyConnected
        elif LayerType.isLogistic(layer_type):
            layer_builder = Sigmoid
        elif LayerType.isSigmoidLUT(layer_type):
            layer_builder = SigmoidLUT
        elif LayerType.isSoftmax(layer_type):
            layer_builder = SoftMax
        elif LayerType.isQuantize(layer_type):
            layer_builder = Quantize
        elif LayerType.isDequantize(layer_type):
            layer_builder = Dequantize
        elif LayerType.isRelu(layer_type):
            layer_builder = RELU
        else:
            raise ValueError("Invalid Layer " + layer_type)

        return layer_builder.from_model(layer)
