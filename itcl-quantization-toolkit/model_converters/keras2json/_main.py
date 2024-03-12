import json
from tensorflow import keras
from keras2json import convert



def _main():

    model_path = "./models/lstm/crypto.h5"
    out_path = "lstm.json"

    model = keras.models.load_model(model_path, compile=False)

    network = convert(model)

    with open(out_path, "w") as f:
        json.dump(
            network,
            f,
        )

    print(f"Generated model at {out_path}")


if __name__ == "__main__":
    _main()
