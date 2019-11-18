import util

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
#from tensorflow.keras.activations import linear, relu, softmax

def collectDatasets():
    util.extract_dataset_to_file(saveName="dataSets/two_notes__no_octave.npz",    num_notes=2, use_octave=False, songPath="./songs")
    util.extract_dataset_to_file(saveName="dataSets/three_notes__no_octave.npz",  num_notes=3, use_octave=False, songPath="./songs")
    util.extract_dataset_to_file(saveName="dataSets/four_notes__no_octave.npz",   num_notes=3, use_octave=False, songPath="./songs")
    # util.extract_dataset_to_file(saveName="dataSets/two_notes__use_octave.npz",   num_notes=2, use_octave=True,  songPath="./songs")
    # util.extract_dataset_to_file(saveName="dataSets/three_notes__use_octave.npz", num_notes=3, use_octave=True,  songPath="./songs")
    # util.extract_dataset_to_file(saveName="dataSets/four_notes__use_octave.npz",  num_notes=3, use_octave=True,  songPath="./songs")

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

    print("Training set rms error: %f" % np.sqrt(mean_squared_error(y0_train, util.relu(linReg0.predict(X0_train)))))
    print("Test set rms error:     %f" % np.sqrt(mean_squared_error(y0_test,  util.relu(linReg0.predict(X0_test )))))
    print("Label mean:  %f" % np.mean(y0_train))
    print("Label stdev: %f" % np.std( y0_train))


    # Linear regression for duration
    print("\nTraining linear regression for duration.")
    linReg1 = LinearRegression(fit_intercept=True)
    linReg1.fit(X1_train, y1_train)
    print("Done training.")

    print("Training set rms error: %f" % np.sqrt(mean_squared_error(y1_train, util.relu(linReg1.predict(X1_train)))))
    print("Test set rms error:     %f" % np.sqrt(mean_squared_error(y1_test,  util.relu(linReg1.predict(X1_test )))))
    print("Label mean:  %f" % np.mean(y1_train))
    print("Label stdev: %f" % np.std( y1_train))
    '''
    # Multiclass logistic regression for pitch
    print("\nTraining logistic regression for pitch.")
    y2_train_i = util.one_hot_to_integer(y2_train)
    y2_test_i  = util.one_hot_to_integer(y2_test)
    logReg2 = LogisticRegression(fit_intercept=True, C=1, class_weight=None, multi_class='ovr')
    logReg2.fit(X2_train, y2_train_i)
    print("Done training.")

    print("Training set accuracy: %f" % logReg2.score(X2_train, y2_train_i))
    print("Test set accuracy:     %f" % logReg2.score(X2_test, y2_test_i))
    util.plot_confusion_matrix(save_name="confusion_logistic.png", y_true=y2_test_i, y_pred=logReg2.predict(X2_test), normalize=False)
    
    
    # Neural net for delay
    for nneurons in [20, 50, 100]:
        for nhiddenlayers in [2, 3, 4]:
            print("\n\n\n", nneurons, "Neurons per Hidden Layer:")
            print(nhiddenlayers, "Hidden Layers:")
            print("\nTraining neural net for delay.")
            nn0 = Sequential()
            for _ in range(nhiddenlayers):
                nn0.add(Dense(nneurons, activation='relu'))
            nn0.add(Dense(1,  activation='relu'))
            nn0.compile(optimizer='adam', loss='mse', metrics=['mse'])
            hnn0 = nn0.fit(X0_train, y0_train, epochs=300, batch_size=5000, verbose=2)
            nn0.save('nnModels/nn0/%d_neurons__%d_layers.h5' % (nneurons, nhiddenlayers))
            
            print("Done training.")

            _, ax1 = plt.subplots()
            ax1.plot(np.sqrt(hnn0.history['loss']), color='tab:red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Root Mean Square Error', color='tab:red')

            #print("Training set rms error: %f" % np.sqrt(nn0.evaluate(x=X0_train, y=y0_train, verbose=0)[1])) # this is slow
            print("Test set rms error:     %f" % np.sqrt(nn0.evaluate(x=X0_test,  y=y0_test,  verbose=0)[1]))
            plt.savefig('figs/nn0/%d_neurons__%d_layers.png' % (nneurons, nhiddenlayers))
    
    # Neural net for duration
    for nneurons in [20, 50, 100]:
        for nhiddenlayers in [2, 3, 4]:
            print("\nTraining neural net for duration.")
            nn1 = Sequential()
            for _ in range(nhiddenlayers):
                nn1.add(Dense(nneurons, activation='relu'))
            nn1.compile(optimizer='adam', loss='mse', metrics=['mse'])
            hnn1 = nn1.fit(X1_train, y1_train, epochs=300, batch_size=5000, verbose=2)
            print("Done training.")
            nn1.save('nnModels/nn1/%d_neurons__%d_layers.h5' % (nneurons, nhiddenlayers))

            _, ax1 = plt.subplots()
            ax1.plot(np.sqrt(hnn1.history['loss']), color='tab:red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Root Mean Square Error', color='tab:red')

            #print("Training set rms error: %f" % np.sqrt(nn1.evaluate(x=X1_train, y=y1_train, verbose=0)[1])) # this is slow
            print("Test set rms error:     %f" % np.sqrt(nn1.evaluate(x=X1_test,  y=y1_test,  verbose=0)[1]))
            plt.savefig('figs/nn1/%d_neurons__%d_layers.png' % (nneurons, nhiddenlayers))
    
    
    # Neural net for pitch
    for nneurons in [20, 50, 100]:
        for nhiddenlayers in [2, 3, 4]:
            print("\n\n\n", nneurons, "Neurons per Hidden Layer:")
            print(nhiddenlayers, "Hidden Layers:")
            numClasses = y2.shape[1] # y2, y2_train, y2_test need to be one-hot vectors
            print("\nTraining neural net for pitch.")
            nn2 = Sequential()
            for _ in range(nhiddenlayers):
                nn2.add(Dense(nneurons, activation='relu'))
            nn2.add(Dense(numClasses,  activation='softmax'))
            nn2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            hnn2 = nn2.fit(X2_train, y2_train, epochs=100, batch_size=5000, verbose=2)
            nn2.save('nnModels/nn2/%d_neurons__%d_layers.h5' % (nneurons, nhiddenlayers))

            print("Done training.")

            _, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax2.plot(np.multiply(hnn2.history['accuracy'], 100), color='tab:blue')
            ax1.plot(hnn2.history['loss'], color='tab:red')
            ax1.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)', color='tab:blue')
            ax1.set_ylabel('Categorical Crossentropy', color='tab:red')

            #print("Training set accuracy: %f" % nn2.evaluate(x=X2_train, y=y2_train, verbose=0)[1]) # this is slow
            print("Test set accuracy:     %f" % nn2.evaluate(x=X2_test,  y=y2_test,  verbose=0)[1])
            plt.savefig('figs/nn2/%d_neurons__%d_layers.png' % (nneurons, nhiddenlayers))


    # Generate Confustion matrix
    modelname = 'nnModels/nn2/100_neurons__4_layers.h5'
    nn2 = load_model(modelname)
    util.plot_confusion_matrix(save_name="confusion_100_neurons__4_layers.png",
        y_true=util.one_hot_to_integer(y2_test),
        y_pred=util.one_hot_to_integer(nn2.predict(X2_test)),
        normalize=False)
    '''
def main():
    # collectDatasets()
    trainModels("dataSets/four_notes__no_octave.npz")


if __name__ == "__main__":
    main()