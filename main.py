import numpy as np
import util

def main():
    X, X_no_stamp, y0, y1, y2 = util.get_data_set("dataSets/two_notes__no_octave.npz")

    print("We have a total of ", X.shape[0], " examples.")
    print(X.shape)
    print(X_no_stamp.shape)
    print(y0.shape)
    print(y1.shape)
    print(y2.shape)

if __name__ == "__main__":
    # util.extract_dataset_to_file(saveName="dataSets/two_notes__no_octave.npz", num_notes=2, use_octave=False, songPath="./songs")
    main()
