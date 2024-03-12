from tensorflow import keras


class CheckType:
    @staticmethod
    def isDense(layer):
        return isinstance(layer, keras.layers.Dense)

    @staticmethod
    def isInput(layer):
        return isinstance(layer, keras.layers.InputLayer) or isinstance(
            layer, keras.layers.Input
        )
