import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from network.Network import SequentialNetwork
from tensorflow.keras.datasets import mnist


def main():

    fn = lambda x: x ** 3

    train_x = linespace = np.expand_dims(np.arange(-10, 10, 0.005), axis=1) 

    scaler = MinMaxScaler()
    scaler.fit(train_x)

    train_y = fn(train_x)
    train_x = scaler.transform(train_x)
    train_y = scaler.transform(train_y)

    input = train_x

    #input = np.expand_dims(train_x, axis=1)
   
    print(input.shape)
    
    #model_path = "./models/itclq/regression/abs_tanh_linear_int16_in_int8.json"
    model_path = "./models/itclq/regression/x3_optimized.json"
    model_path = "./models/itclq/regression/x3.json"

    model_path = "./models/tflite/x3tflite.json"
    model_path = "./models/itclq/regression/x3_symmetric.json"
    net = SequentialNetwork.from_json_file(model_path)

    net.infer(np.array([0.5]))
    

    results = net.infer(input)

    results_train = results 
    print(results.shape)

    #results = np.array([net.infer(x) for x in input])
    #results_train = np.array([net.infer(x) for x in input])

    expected = train_y
    print(expected.shape)
    mse = np.mean((expected - results_train) ** 2)
    print(f"Train MSE: {mse}")


    # plot the expected and results
    import matplotlib.pyplot as plt
    plt.plot(expected, label="expected")
    plt.plot(results )
    plt.title(f"MSE: {mse}")
    plt.plot()

    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()

# %%
