import numpy as np
import matplotlib.pyplot as plt
from mido import MidiFile
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import random
import os

def get_song_features(song_name, num_notes, use_octave=False, augment=True):
    '''
    returns the feature vectors for the song along with labels for the note duration and note pitch
    '''
    
    random.seed(42069)
    num_augments = 11 # number of extra augments we should do

    midi = MidiFile(song_name)
    notes = defaultdict(list)
    timestamps = [] # needed since defaultdicts don't necessarily have key ordering
    time = 0
    totalNotes = 0
    num_pitches = 12 if use_octave is False else 128
    numFeatures = ((2 + num_pitches) * num_notes)
    X = None
    y0, y1, y2 = None, None, None
    X_aug = None
    y0, y1, y2 = None, None, None
    for i, track in enumerate(midi.tracks):
        for msg in track:
            time += msg.time
            if msg.type == 'note_on':
                pitch = msg.note if use_octave else msg.note % 12
                octave = msg.note // 12 - 1
                if msg.velocity != 0:
                    # note is turning on
                    note = (pitch, -1, octave)
                    totalNotes += 1
                    notes[time].append(note)
                    if time not in timestamps:
                        timestamps.append(time)
                if msg.velocity == 0:
                    # note is turning off
                    noteFound = False
                    for dictIdx in reversed(timestamps):
                        noteList = notes[dictIdx]
                        for note in noteList:
                            if pitch == note[0] and -1 == note[1] and octave == note[2]:
                                # found our note
                                duration = time - dictIdx
                                labelled_note = (note[0], duration, note[2])
                                noteList.remove(note)
                                noteList.append(labelled_note)
                                noteFound = True
                                break
                        if noteFound:
                            break

        num_examples = totalNotes
        for numIdx in range(num_notes):
            num_examples -= len(notes[timestamps[numIdx]])
        num_examples -= len(notes[timestamps[len(timestamps) - 1]])

        X = np.zeros((num_examples, numFeatures), dtype=int)
        y0 = np.zeros(num_examples, dtype=int) # relative starting time
        y1 = np.zeros(num_examples, dtype=int) # duration
        y2 = np.zeros((num_examples, num_pitches), dtype=int) # pitch labels
        X_aug = np.zeros((num_augments * num_examples, numFeatures), dtype=int)
        y0_aug = np.zeros(num_augments * num_examples, dtype=int)
        y1_aug = np.zeros(num_augments * num_examples, dtype=int)
        y2_aug = np.zeros((num_augments * num_examples, num_pitches), dtype=int)
        example = 0
        for timeIdx in range(len(timestamps) - num_notes - 1):
            timestampIdx = timeIdx + num_notes # potentially miss avoidable notes here, but this should be fine
            timestamp = timestamps[timestampIdx]
            for currNote in notes[timestamp]:
                # for each note that starts at this time
                offset = 1
                list_priors = []
                # get the prior notes
                while len(list_priors) < num_notes:
                    priorTimestamp = timestamps[timestampIdx - offset]
                    nextNotes = notes[priorTimestamp].copy()
                    random.shuffle(nextNotes)
                    # delayFactored = False
                    while len(nextNotes) > 0:
                        shortestDuration = float('inf')
                        shortestNote = np.zeros(num_pitches + 2, dtype=int)
                        originalShort = None
                        for priorNote in nextNotes:
                            if priorNote[1] < shortestDuration:
                                shortestDuration = priorNote[1]
                                # features are timestamp relative to curr note's timestamp, duration, pitch of prior note
                                delayParam = priorTimestamp - timestamp
                                # if delayFactored is True:
                                #     delayParam = 0
                                # else:
                                #     delayFactored = True
                                shortestNote[0] = delayParam
                                shortestNote[1] = priorNote[1]
                                shortestNote[2 + priorNote[0]] = 1
                                originalShort = priorNote
                        list_priors.append(shortestNote)
                        nextNotes.remove(originalShort)
                        if len(list_priors) >= num_notes:
                            break
                    offset += 1
                X[example, :] = np.ndarray.flatten(np.asarray(list_priors))
                y0[example] = timestamps[timestampIdx + 1] - timestamp
                y1[example] = currNote[1]
                pitch_vec = np.zeros(num_pitches, dtype=int)
                pitch_vec[currNote[0]] = 1
                y2[example, :] = pitch_vec

                if augment:
                    original_X = X[example, :]
                    original_y0 = y0[example]
                    original_y1 = y1[example]
                    original_y2 = y2[example]
                    for augmentNum in range(num_augments):
                        augmentIdx = example * num_augments + augmentNum
                        bump_amount = augmentNum + 1
                        new_X = np.zeros(num_notes * (num_pitches + 2), dtype=int)
                        new_y2 = np.zeros(num_pitches, dtype=int)
                        note_size = num_pitches + 2
                        for traverse_idx in range(num_notes * note_size):
                            if traverse_idx % note_size == 0 or traverse_idx % note_size == 1:
                                new_X[traverse_idx] = original_X[traverse_idx]
                            elif original_X[traverse_idx] == 1:
                                note_num = traverse_idx // note_size
                                note_idx = traverse_idx % note_size
                                pitch = note_idx - 2
                                augmented_pitch = (pitch + bump_amount) % num_pitches
                                augmented_note_idx = augmented_pitch + 2
                                augment_idx = note_num * note_size + augmented_note_idx
                                new_X[augment_idx] = 1
                        X_aug[augmentIdx, :] = new_X
                        y0_aug[augmentIdx] = original_y0
                        y1_aug[augmentIdx] = original_y1
                        y2_pitch_original = np.argmax(original_y2, axis=0)
                        new_y2[(y2_pitch_original + bump_amount) % num_pitches] = 1
                        y2_aug[augmentIdx, :] = new_y2
                example += 1

    if augment:
        X = np.vstack((X, X_aug))
        y0 = np.concatenate((y0, y0_aug))
        y1 = np.concatenate((y1, y1_aug))
        y2 = np.vstack((y2, y2_aug))

    return X, y0, y1, y2


def extract_dataset_to_file(saveName, num_notes, use_octave=False, songPath="./songs"):
    '''
    Read through all the songs in songPath and save to saveName.
    '''
    songList = []
    for _, _, files in os.walk(songPath):
        songList = songList + files
    X = None
    y0 = None
    y1 = None
    y2 = None
    for songNum, song in enumerate(songList):
        print("Reading song %d of %d: %s" % (songNum + 1, len(songList), song))
        songX, songY0, songY1, songY2 = get_song_features(songPath + "/" + song, num_notes, use_octave)
        if X is None:
            X = songX
            y0 = songY0
            y1 = songY1
            y2 = songY2
        else:
            X = np.vstack((X, songX))
            y0 = np.concatenate((y0, songY0))
            y1 = np.concatenate((y1, songY1))
            y2 = np.vstack((y2, songY2))

    np.savez(saveName, X=X, y0=y0, y1=y1, y2=y2)


def one_hot_to_integer(one_hot):
    if one_hot.ndim >= 2:
        return np.argmax(one_hot, axis=1)[0]
    else:
        return np.argmax(one_hot)[0]

def get_data_set(filename):
    '''
    Get the dataset saved in filename
    '''
    dataset = np.load(filename)
    return dataset['X'], dataset['y0'], dataset['y1'], dataset['y2']

def relu(x):
    for el in x:
        if el < 0:
            el = 0
    return x

# Adapted from scikit-learn confusion matrix page:
def plot_confusion_matrix(save_name, y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ['0 (C)', '1 (C#)', '2 (D)', '3 (D#)', '4 (E)', '5 (F)', '6 (F#)','7 (G)', '8 (G#)', '9 (A)', '10 (A#)', '11 (B)']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_name != None:
        plt.savefig(save_name)
    return ax
