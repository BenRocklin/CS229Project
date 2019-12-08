import util

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras.activations import linear, relu, softmax
#from tensorflow.keras.regularizers import l2

def collectDatasets():
    util.extract_dataset_to_file(saveName="dataSets/two_notes__no_octave.npz",    num_notes=2, use_octave=False, songPath="./songs")
    util.extract_dataset_to_file(saveName="dataSets/three_notes__no_octave.npz",  num_notes=3, use_octave=False, songPath="./songs")
    util.extract_dataset_to_file(saveName="dataSets/four_notes__no_octave.npz",   num_notes=3, use_octave=False, songPath="./songs")
    util.extract_dataset_to_file(saveName="dataSets/two_notes__use_octave.npz",   num_notes=2, use_octave=True,  songPath="./songs")
    util.extract_dataset_to_file(saveName="dataSets/three_notes__use_octave.npz", num_notes=3, use_octave=True,  songPath="./songs")
    util.extract_dataset_to_file(saveName="dataSets/four_notes__use_octave.npz",  num_notes=3, use_octave=True,  songPath="./songs")

def trainModels(dataSet):
    # Get dataset, split into training set and test set
    X, y0, y1, y2 = util.get_data_set(dataSet)

    X_train, X_test, y0_train, y0_test, y1_train, y1_test, y2_train, y2_test \
        = train_test_split(X, y0, y1, y2, test_size=.15)

    X0_train = X_train
    X0_test  = X_test

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
    print("Test set rms error:     %f" % np.sqrt(mean_squared_error(y1_test,  linReg1.predict(X1_test))))


    # Multiclass logistic regression for pitch
    print("\nTraining logistic regression for pitch.")
    logReg2 = LogisticRegression(fit_intercept=True, C=1, class_weight=None, multi_class='ovr')
    logReg2.fit(X2_train, y2_train)
    print("Done training.")

    print("Training set accuracy: %f" % logReg2.score(X2_train, y2_train))
    print("Test set accuracy:     %f" % logReg2.score(X2_test, y2_test))



    # Neural net for delay
    print("\nTraining neural net for delay.")
    nn0 = Sequential()
    nn0.add(Dense(10, activation='relu'))
    nn0.add(Dense(10, activation='relu'))
    nn0.add(Dense(1,  activation='relu'))
    nn0.compile(optimizer='adam', loss='mse', metrics=['mse'])
    hnn0 = nn0.fit(X0_train, y0_train, epochs=500, batch_size=500)
    print("Done training.")

    print("Training set rms error: %f" % np.sqrt(hnn0.history['mse'][-1]))
    print("Test set rms error:     %f" % np.sqrt(nn0.evaluate(x=X0_test,  y=y0_test, verbose=0)[1]))


    # Neural net for duration
    print("\nTraining neural net for duration.")
    nn1 = Sequential()
    nn1.add(Dense(10, activation='relu'))
    nn1.add(Dense(10, activation='relu'))
    nn1.add(Dense(1,  activation='relu'))
    nn1.compile(optimizer='adam', loss='mse', metrics=['mse'])
    hnn1 = nn1.fit(X1_train, y1_train, epochs=500, batch_size=500)
    print("Done training.")

    print("Training set rms error: %f" % np.sqrt(hnn1.history['mse'][-1]))
    print("Test set rms error:     %f" % np.sqrt(nn1.evaluate(x=X1_test,  y=y1_test, verbose=0)[1]))


    # Neural net for pitch
    numClasses = y2.shape[1] # y2, y2_train, y2_test need to be one-hot vectors
    print("\nTraining neural net for pitch.")
    nn2 = Sequential()
    nn2.add(Dense(10, activation='relu'))
    nn2.add(Dense(10, activation='relu'))
    nn2.add(Dense(numClasses,  activation='softmax'))
    nn2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hnn2 = nn2.fit(X2_train, y2_train, epochs=2, batch_size=500)
    print("Done training.")

    print("Training set accuracy: %f" % np.sqrt(hnn2.history['accuracy'][-1]))
    print("Test set accuracy:     %f" % np.sqrt(nn2.evaluate(x=X2_test,  y=y2_test, verbose=0)[1]))



def main():
    collectDatasets()
    trainModels("dataSets/four_notes__no_octave.npz")


if __name__ == "__main__":
    main()