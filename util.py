import numpy as np
from mido import MidiFile
from collections import defaultdict
import random

# returns the feature vectors for the song along with labels for the note duration and note pitch
def get_song_features(song_name, num_notes, use_octave=False):
    random.seed(42069)

    midi = MidiFile(song_name)
    notes = defaultdict(list)
    timestamps = [] # needed since defaultdicts don't necessarily have key ordering
    time = 0
    totalNotes = 0
    numFeatures = (3 * num_notes)
    X = None
    y1, y2 = None, None
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

        X = np.zeros((num_examples, numFeatures), dtype=int)
        y0 = np.zeros(num_examples) # relative starting time
        y1 = np.zeros(num_examples) # duration
        y2 = np.zeros(num_examples) # pitch labels
        example = 0
        for timeIdx in range(len(timestamps) - num_notes):
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
                    while len(nextNotes) > 0:
                        shortestDuration = float('inf')
                        shortestNote = None
                        originalShort = None
                        for priorNote in nextNotes:
                            if priorNote[1] < shortestDuration:
                                shortestDuration = priorNote[1]
                                # features are timestamp relative to curr note's timestamp, duration, pitch of prior note
                                shortestNote = (priorTimestamp - timestamp, priorNote[1], priorNote[0])
                                originalShort = priorNote
                        list_priors.append(shortestNote)
                        nextNotes.remove(originalShort)
                        if len(list_priors) >= num_notes:
                            break
                    offset += 1
                X[example, :] = np.ndarray.flatten(np.asarray(list_priors))
                y0[example] = timestamp - timestamps[timestampIdx - 1]
                y1[example] = currNote[1]
                y2[example] = currNote[0]
                example += 1
        X_no_stamp = np.delete(X, list(range(0, X.shape[1], 3)), axis=1)

    return X, X_no_stamp, y0, y1, y2