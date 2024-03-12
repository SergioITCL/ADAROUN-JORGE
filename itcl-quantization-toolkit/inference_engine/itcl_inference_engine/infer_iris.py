import time
import numpy as np
from network.Network import SequentialNetwork
from tensorflow.keras.datasets import mnist


def main():

    model_path = "./models/itclq/iris/iris.json"
    print(model_path)
    
    net = SequentialNetwork.from_json_file(model_path)

    X, Y = np.load("./models/itclq/iris/data/x_train.npy"), np.load("./models/itclq/iris/data/y_train.npy")


    res = net.infer(X, isBatch=True)
    
    res = res.T.argmax(axis=1)
    
    start = time.time()
    hits = 0
    
    for pred, exp in zip(res, Y):
        if pred == exp:
            hits +=1
    print(f"ACC: {hits / len(Y)}")
    print(f"Hits: {hits}")
    print(f"Time: {time.time() - start}")


if __name__ == "__main__":
    main()
