import util
import numpy as np
from tensorflow.keras.models import load_model


class NNPredictor:
    def __init__(self, num_notes, use_octave, nnModelNames):
        model_dir = "models/%d_notes__%s" % (num_notes, "use_octave" if use_octave else "no_octave")
        self.nn0 = load_model(model_dir + "/delay/"    + nnModelNames[0])
        self.nn1 = load_model(model_dir + "/duration/" + nnModelNames[1])
        self.nn2 = load_model(model_dir + "/pitch/"    + nnModelNames[2])

    def predict(self, x):
        print("== Predicting ==")

        delay    = (int) (self.nn0.predict(x.reshape((1,-1)))[0])
        duration = (int) (self.nn1.predict(x.reshape((1,-1)))[0])
        pitch    = self.nn2.predict(x.reshape((1,-1)))
        pitch = util.one_hot_to_integer(pitch)

        print("delay =")
        print(delay)
        print("duration =")
        print(duration)
        print("pitch =")
        print(pitch)

        return delay, duration, pitch


class LinearPredictor:
    def __init__(self, num_notes, use_octave):
        pass

    def predict(self, x):
        delay    = None
        duration = None
        pitch    = None
        # one hot conversion unnecessary here
        return delay, duration, pitch


def generate_song_notes(num_notes, use_NN, nnModelNames, song_length=300, num_throwaway_notes=10, use_octave=False):
    if use_NN:
        predictor = NNPredictor(num_notes, use_octave, nnModelNames)
    else:
        predictor = LinearPredictor(num_notes, use_octave)

    print("=== Initializing Notes === ")

    notes = initialize_notes(num_notes, use_octave, predictor)

    print("inital notes =")
    print(notes)
    print("=== Notes initialized === ")


    for _ in range(num_throwaway_notes):
        print("=== Getting Features ===")
        print("notes = ")
        print(notes)
        features = get_features(notes, use_octave, num_notes)
        print("features =")
        print(features)
        note = predictor.predict(features)
        notes.append(note)
        notes.pop(0)

    for _ in range(song_length):
        features = get_features(notes, use_octave, num_notes)
        note = predictor.predict(features)
        notes.append(note)

    return notes


def initialize_notes(num_notes, use_octave, predictor):
    dataSet = "dataSets/%d_notes__%s.npz" % (num_notes, "use_octave" if use_octave else "no_octave")
    X, _, _, _ = util.get_data_set(dataSet)
    indices = np.random.randint(0, X.shape[0], size=num_notes)
    features =  X[indices,:]

    notes = [] # list of note tuples of form (delay, duration, pitches)
               # note: all three values above should be an int
    for i in range(num_notes):
        note = predictor.predict(features[i,:])
        print("blah blah blah")
        print(features[i,:])
        notes.append(note)

    return notes

# Ben do this function:
def get_features(x, num_notes, use_octave):
    num_pitches = 12 if use_octave is False else 128
    note_len = (2 + num_pitches)
    feature_length = note_len * num_notes
    feature = np.zeros(feature_length, dtype=int)
    noteIdx = 0
    distanceBack = None
    for note in reversed(x):
        if noteIdx == num_notes:
            break
        delay = note[0]
        duration = note[1]
        pitch = note[2] # util.one_hot_to_integer(note[2]) # I don't think you meant this here
        if distanceBack is None:
            distanceBack = delay
        else:
            distanceBack += delay

        relStartIdx = note_len * noteIdx
        durationIdx = note_len * noteIdx + 1
        pitchIdx = note_len * noteIdx + 2 + pitch
        feature[relStartIdx] = distanceBack
        feature[durationIdx] = duration
        feature[pitchIdx] = 1
        noteIdx += 1
    feature = feature.reshape((feature.shape[0], 1)).T
    return feature


# Ben do this function
def generate_midi_file(midi_file_name, notes):
    
    # save to midi_file_name
    pass


def main():
    ### Parameters ###
    num_notes = 3
    use_octave = False
    use_NN = True
    nnModelNames = (  # only used when use_NN == True
        "20_neurons__2_layers.h5",       # delay 
        "20_neurons__2_layers.h5",       # duration
        "20_neurons__2_hiddenlayers.h5") # pitch

    song_length = 300
    num_throwaway_notes = 10
    midi_file_name = "generatedSongs/our_first_song.midi"

    ### Compose a song! ###
    notes = generate_song_notes(num_notes, use_NN, nnModelNames, song_length, num_throwaway_notes, use_octave)
    generate_midi_file(midi_file_name, notes)

if __name__ == '__main__':
    main()
    notes = [(294, 489, 0), (168, 221, 7), (172, 155, 1)] # typical valid input
    features = get_features(x=notes, num_notes=3, use_octave=False)
    print(features)





















