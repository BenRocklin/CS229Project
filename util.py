import numpy as np
from mido import MidiFile
from collections import defaultdict
import random
import os

def get_song_features(song_name, num_notes, use_octave=False):
    '''
    returns the feature vectors for the song along with labels for the note duration and note pitch
    '''
    
    random.seed(42069)

    midi = MidiFile(song_name)
    notes = defaultdict(list)
    timestamps = [] # needed since defaultdicts don't necessarily have key ordering
    time = 0
    totalNotes = 0
    num_pitches = 12 if use_octave is False else 128
    numFeatures = ((2 + num_pitches) * num_notes)
    X = None
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
        y0 = np.zeros(num_examples) # relative starting time
        y1 = np.zeros(num_examples) # duration
        y2 = np.zeros((num_examples, num_pitches)) # pitch labels
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
                example += 1

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


def get_data_set(filename):
    '''
    Get the dataset saved in filename
    '''
    dataset = np.load(filename)
    return dataset['X'], dataset['y0'], dataset['y1'], dataset['y2']


