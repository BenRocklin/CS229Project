import util
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import load_model
from mido import Message, MidiFile, MidiTrack
import pickle

class NNPredictor:
    def __init__(self, num_notes, use_octave, nnModelNames):
        model_dir = "models/%d_notes__%s" % (num_notes, "use_octave" if use_octave else "no_octave")
        self.nn0 = load_model(model_dir + "/delay/"    + nnModelNames[0])
        self.nn1 = load_model(model_dir + "/duration/" + nnModelNames[1])
        self.nn2 = load_model(model_dir + "/pitch/"    + nnModelNames[2])

    def predict(self, x, verbose=False):

        delay    = (int) (self.nn0.predict(x.reshape((1,-1)))[0])
        duration = (int) (self.nn1.predict(x.reshape((1,-1)))[0])
        pitch    = self.nn2.predict(x.reshape((1,-1)))
        pitch = util.one_hot_to_integer(pitch)

        if verbose:
            print("== Predicting ==")
            print("delay =")
            print(delay)
            print("duration =")
            print(duration)
            print("pitch =")
            print(pitch)

        return delay, duration, pitch


class LinearPredictor:
    def __init__(self, num_notes, use_octave):
        model_dir = "models/%d_notes__%s" % (num_notes, "use_octave" if use_octave else "no_octave")

        self.linReg0 = pickle.load(open(model_dir + "/delay/linear_model.sav", 'rb'))
        self.linReg2 = pickle.load(open(model_dir + "/duration/linear_model.sav", 'rb'))
        self.logReg3 = pickle.load(open(model_dir + "/pitch/logistic_model.sav", 'rb'))

    def predict(self, x):
        delay    = self.relu(self.linReg0.predict(x.reshape((1,-1)))[0])
        duration = self.relu(self.linReg2.predict(x.reshape((1,-1)))[0])
        pitch    = self.logReg3.predict(x.reshape((1,-1)))[0]
        # one hot conversion unnecessary here

        print("== Predicting ==")
        print("delay =")
        print(delay)
        print("duration =")
        print(duration)
        print("pitch =")
        print(pitch)
        return delay, duration, pitch

    def relu(self, x):
        if x > 0:
            return (int) (x)
        else:
            return (int) (0)


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
        features = get_features(notes, num_notes, use_octave)
        print("features =")
        print(features)
        note = predictor.predict(features)
        notes.append(note)
        notes.pop(0)

    for _ in range(song_length):
        features = get_features(notes, num_notes, use_octave)
        note = predictor.predict(features)
        notes.append(note)

    return notes


def initialize_notes(num_notes, use_octave, predictor):
    dataSet = "dataSets/%d_notes__%s.npz" % (num_notes, "use_octave" if use_octave else "no_octave")
    X, _, _, _ = util.get_data_set(dataSet)
    np.random.seed(5)
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
    # feature = feature.reshape((feature.shape[0], 1)).T
    return feature

def generate_midi_file(midi_file_name, notes, use_octave):
    timedNotes = defaultdict(list)
    currTimestamp = 0
    noteId = 0
    for note in notes:
        delay = note[0]
        duration = note[1]
        pitch = note[2]
        if use_octave is False:
            pitch += 60
        endNoteTime = currTimestamp + duration
        timedNotes[currTimestamp].append((pitch, noteId))
        timedNotes[endNoteTime].append((pitch, noteId))

        print(noteId, pitch, currTimestamp, endNoteTime)
        noteId += 1
        currTimestamp += delay

    print(timedNotes)
    idsToDelete = []
    for timestamp in timedNotes:
        noteList = timedNotes[timestamp]
        newList = []
        seenPitches = []
        for note in noteList:
            if note[0] not in seenPitches:
                seenPitches.append(note[0])
                newList.append(note)
            else:
                idsToDelete.append(note[1])
        timedNotes[timestamp] = newList
    for timestamp in timedNotes:
        noteList = timedNotes[timestamp]
        newList = []
        for note in noteList:
            if note[1] not in idsToDelete:
                newList.append(note)
        timedNotes[timestamp] = newList

    print(timedNotes)
    idsToDelete = []
    activePitches = np.zeros(128, dtype=int)
    for timestamp in sorted(timedNotes.keys()):
        noteList = timedNotes[timestamp]
        for note in noteList:
            if note[1] in idsToDelete:
                continue
            if activePitches[note[0]] == 0:
                activePitches[note[0]] = note[1]
            elif activePitches[note[0]] == note[1]:
                activePitches[note[0]] = 0
            else:
                idsToDelete.append(note[1])
    for timestamp in timedNotes:
        noteList = timedNotes[timestamp]
        newList = []
        for note in noteList:
            if note[1] not in idsToDelete:
                newList.append(note)
        timedNotes[timestamp] = newList

    for timestamp in sorted(timedNotes):
        for note in timedNotes[timestamp]:
            print(note)

    # save to midi_file_name
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(Message('program_change', program=1, time=0))
    sorted_keys = sorted(timedNotes)
    lastStamp = sorted_keys[-1]
    activatedId = -1
    stampIdx = 0
    for timestamp in sorted_keys:
        print(timestamp, timedNotes[timestamp])
        for noteIdx in range(len(timedNotes[timestamp])):
            note = timedNotes[timestamp][noteIdx]
            time = 0
            if noteIdx == len(timedNotes[timestamp]) - 1:
                if stampIdx == len(timedNotes) - 1:
                    time = 32
                else:
                    time = sorted_keys[stampIdx + 1] - sorted_keys[stampIdx]
            if note[1] > activatedId:
                activatedId = note[1]
                track.append(Message('note_on', note=note[0], velocity=127, time=time))
                print("ON", note[0], time)
            else:
                track.append(Message('note_off', note=note[0], velocity=127, time=time))
                print("OFF", note[0], time)
        stampIdx += 1
    mid.save(midi_file_name)


def main():
    ### Parameters ###
    num_notes = 5
    use_octave = False
    use_NN = False
    if num_notes == 3:
        nnModelNames = (  # only used when use_NN == True
            "100_neurons__4_layers.h5",       # delay 
            "100_neurons__3_layers.h5",       # duration
            "50_neurons__4_hiddenlayers.h5") # pitch
    elif num_notes == 5:
        nnModelNames = (  # only used when use_NN == True
            "100_neurons__4_layers.h5",       # delay 
            "100_neurons__4_layers.h5",       # duration
            "100_neurons__4_hiddenlayers.h5") # pitch

    song_length = 300
    num_throwaway_notes = 0
    midi_file_name = ("generatedSongs/%s_song__%d_prior_notes__no_2.mid" % ("NN" if use_NN else "LIN", num_notes))

    ### Compose a song! ###
    notes = generate_song_notes(num_notes, use_NN, nnModelNames, song_length, num_throwaway_notes, use_octave)
    generate_midi_file(midi_file_name, notes, use_octave)

if __name__ == '__main__':
    main()
    #notes = [(294, 489, 0), (168, 221, 7), (172, 155, 1)] # typical valid input
    #features = get_features(x=notes, num_notes=3, use_octave=False)
    #print(features)




















