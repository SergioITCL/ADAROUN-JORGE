import time
import numpy as np
from network.Network import SequentialNetwork
from tensorflow.keras.datasets import mnist


def main():
    model_path = (
        "models/tflite/mnist_high_complexity_float32_tanh_quant_deq_in_out_layers.json"
    )
    # model_path = "models/onnx/mnist_tanh_uint8_lut_reduced.json"
    model_path = "./mnist.json"
    model_path = "models/itclq/mnist/mnist.json"
    model_path = "models/tflite/mnist_high_complexity_float32_tanh.json"
    model_path = "models/tflite/mnist_low_complexity_float32_relu_quant_deq_in_out_layers.json"
    model_path = "mnist.json"
    print(model_path)
    
    net = SequentialNetwork.from_json_file(model_path)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_test_dense = (
        x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2])).astype(
            "float32"
        )
        / 255
    )
    #batch = np.expand_dims(x_test_dense, axis=2)
    batch = x_test_dense
    res = net.infer(batch)
    res = res.T.argmax(axis=0)
    start = time.time()
    hits = 0
    
    for pred, exp in zip(res, y_test):
        
        #print(pred, exp)
        if pred == exp:
            hits +=1
    print(f"ACC: {hits / len(x_test_dense)}")
    print(f"Hits: {hits}")
    print(f"Time: {time.time() - start}")


if __name__ == "__main__":
    main()
