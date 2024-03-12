class LayerType():
    
    @staticmethod
    def isFullyConnected(layerCode: str):
        return "FULLY_CONN" in layerCode.upper() 

    @staticmethod
    def isTanH(layerCode: str):
        return "TANH" == layerCode.upper()

    @staticmethod
    def isTanHLUT(layerCode: str):
        return "TANHLUT" == layerCode.upper()

    @staticmethod
    def isLogistic(layerCode: str):
        return "LOGISTIC" in layerCode.upper()

    @staticmethod
    def isSoftmax(layerCode: str):
        return "SOFTMAX" in layerCode.upper()
    
    @staticmethod
    def isQuantize(layerCode: str):
        return "QUANTIZE" == layerCode.upper()

    @staticmethod
    def isDequantize(layerCode: str):
        return "DEQUANTIZE" in layerCode.upper()
    
    @staticmethod
    def isSigmoidLUT(layerCode: str):
        return "SIGMOIDLUT" in layerCode.upper()

    @staticmethod
    def isRelu(layerCode: str):
        return "RELU" == layerCode.upper()