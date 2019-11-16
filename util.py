import mido
from mido import MidiFile

def get_song_features(song_name, num_notes, octave=False, duration=False):
    print(song_name)
    midi = MidiFile(song_name)
    # for i, track in enumerate(midi.tracks):
    #     print(track.name)