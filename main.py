import numpy as np
import util
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error


def main():
    dataSet = "dataSets/four_notes__no_octave.npz"


    # Get dataset, split into training set and test set
    X, X_no_stamp, y0, y1, y2 = util.get_data_set(dataSet)

    X_train, X_test, X_no_stamp_train, X_no_stamp_test, y0_train, y0_test, y1_train, y1_test, y2_train, y2_test \
        = sk.model_selection.train_test_split(X, X_no_stamp, y0, y1, y2, test_size=.15)

    X0_train = X_no_stamp_train
    X0_test  = X_no_stamp_test

    X1_train = X_train
    X1_test  = X_test

    X2_train = X_train
    X2_test  = X_test


    # Linear regression for delay
    print("\nTraining linear regression for delay.")
    linReg0 = LinearRegression(fit_intercept=True)
    linReg0.fit(X0_train, y0_train)
    print("Done training.")

    print("Training set rms error: %f" % np.sqrt(mean_squared_error(y0_train, linReg0.predict(X0_train))))
    print("Test set rms error:     %f" % np.sqrt(mean_squared_error(y0_test,  linReg0.predict(X0_test ))))


    # Linear regression for duration
    print("\nTraining linear regression for duration.")
    linReg1 = LinearRegression(fit_intercept=True)
    linReg1.fit(X1_train, y1_train)
    print("Done training.")

    print("Training set rms error: %f" % np.sqrt(mean_squared_error(y1_train, linReg1.predict(X1_train))))
    print("Test set rms error:     %f" % np.sqrt(mean_squared_error(y1_test,  linReg1.predict(X1_test ))))


    # Multiclass logistic regression for pitch
    print("\nTraining logistic regression for pitch.")
    logReg2 = LogisticRegression(fit_intercept=True, C=1, class_weight=None, multi_class='ovr')
    logReg2.fit(X2_train, y2_train)
    print("Done training.")

    print("Training set mean accuracy: %f" % logReg2.score(X2_train, y2_train))
    print("Test set mean accuracy:     %f" % logReg2.score(X2_test, y2_test))





if __name__ == "__main__":
    # util.extract_dataset_to_file(saveName="dataSets/two_notes__no_octave.npz",    num_notes=2, use_octave=False, songPath="./songs")
    # util.extract_dataset_to_file(saveName="dataSets/three_notes__no_octave.npz",  num_notes=3, use_octave=False, songPath="./songs")
    # util.extract_dataset_to_file(saveName="dataSets/four_notes__no_octave.npz",   num_notes=3, use_octave=False, songPath="./songs")
    # util.extract_dataset_to_file(saveName="dataSets/two_notes__use_octave.npz",   num_notes=2, use_octave=True,  songPath="./songs")
    # util.extract_dataset_to_file(saveName="dataSets/three_notes__use_octave.npz", num_notes=3, use_octave=True,  songPath="./songs")
    # util.extract_dataset_to_file(saveName="dataSets/four_notes__use_octave.npz",  num_notes=3, use_octave=True,  songPath="./songs")
    main()