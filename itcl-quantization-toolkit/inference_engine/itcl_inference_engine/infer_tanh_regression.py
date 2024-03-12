import time
import numpy as np
from network.Network import SequentialNetwork
from tensorflow.keras.datasets import mnist
from itertools import groupby


def interpolate_noise(arr: np.ndarray):
    

    res = []

    for k, g in groupby(arr):
        segment = [None] * len(list(g))
        segment[len(segment)  // 2] = k
        res.extend(segment)

    #res = np.array(res, dtype=object)

    x = np.array(res, dtype="float32")

    xp = [i for i, yi in enumerate(x) if np.isfinite(yi)]
    fp = [yi for i, yi in enumerate(x) if np.isfinite(yi)]
    return np.interp(x=list(range(len(x))), xp=xp, fp=fp)



def main():
    model_path = "./models/itclq/regression/abs_tanh_linear_last_int16.json"
    model_path = "./abs_tanh_linear.json"
    net = SequentialNetwork.from_json_file(model_path)
    
    ext_input = np.expand_dims(np.arange(-15, 15, 0.01), axis=1)
    train_input = np.expand_dims(np.arange(-10, 10, 0.01), axis=1) 


    #expected_ext = np.array(np.load("./linespace.npy"))
    expected_ext = np.abs(ext_input)
    expected_train = np.abs(train_input)

    results_ext = np.array([net.infer(x) for x in ext_input])
    results_train = np.array([net.infer(x) for x in train_input])

    
    # print mse between expected and results
    #print(expected_ext[500:-500] - results_train)

    print("Train MAE: ", np.mean(np.abs(expected_train - results_train)))
    print(f"Extended DataSet MSE: {np.mean((expected_ext - results_ext) ** 2)}")
    print(f"Train MSE: {np.mean((expected_train - results_train) ** 2)}")


    results_train_inter = interpolate_noise(results_train)
    results_train_inter = np.expand_dims(results_train_inter, axis=1)
    print(results_train_inter.shape)
    print(expected_train.shape)
    print(f"Inter Train MSE: {np.mean((expected_train - results_train_inter) ** 2)}")


    results_ext_inter = interpolate_noise(results_ext)

    # plot the expected and results
    import matplotlib.pyplot as plt
    plt.plot(expected_ext, label="expected")
    #plt.plot(results_ext, label="results")
    plt.plot(results_ext, label="results")
    plt.plot(results_ext_inter, label="Interpolated")
    # Draw a vertical line that crosses the results[500] point
    plt.axvline(x=500, color="red", linestyle="--", alpha=0.3,)
    plt.axvline(x=2500, color="red", linestyle="--", alpha=0.3)

    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()
