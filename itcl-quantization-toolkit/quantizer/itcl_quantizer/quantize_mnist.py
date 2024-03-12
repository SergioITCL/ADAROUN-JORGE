from time import time

from itcl_inference_engine.network.Network import Network
from tensorflow import keras
from itcl_quantizer.config.models.keras import KerasDenseCfg, QuantizerCfg
from itcl_quantizer.tensor_extractor.keras.keras_builder import build

if __name__ == "__main__":
    # load mnist dataset:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # scale mnist
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # flatten mnist
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    data = x_train  # [0:100]

    def loss(net: Network) -> float:

        res = net.infer(x_test)
        res = res.T.argmax(axis=0)

        hits = 0
        for pred, exp in zip(res, y_test):

            # print(pred, exp)
            if pred == exp:
                hits += 1
        return -hits
    
    cfg = QuantizerCfg()
    cfg.dense.kernel_dtype = "int2"

    t = time()
    net = build(
        "./models/keras/mnist_big_low_complexity_float32_tanh.h5",
        "mnist_big_low_int4.json",
        data,  # loss
        cfg=cfg
    )
    print(f"Quantization completed in {time() - t}s")
    seq_net = net.as_sequential_network()
    print("Network Loss:", loss(seq_net))
