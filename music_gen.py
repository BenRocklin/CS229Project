import util
import numpy as np

class NNPredictor:
	def __init__(self, num_notes, use_octave):
		pass

	def predict(self, x):
		delay    = None
		duration = None
		pitch    = None
		pitch = util.one_hot_to_integer(pitch)
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


def generate_song_notes(num_notes, use_NN, song_length=300, num_throwaway_notes=20, use_octave=False):
	notes = [] # list of note tuples of form (delay, duration, pitches)
	# note: all three values above should be an int

	notes = initialize_features(num_notes, use_octave) # make x a list of tuples

	if use_NN:
		predictor = NNPredictor(num_notes, use_octave)
	else:
		predictor = LinearPredictor(num_notes, use_octave)

	for _ in range(num_throwaway_notes):
		features = get_features(notes, use_octave, num_notes)
		note = predictor.predict(features)
		notes.append(note)
		notes.pop(0)

	for _ in range(song_length):
		features = get_features(notes, use_octave, num_notes)
		note = predictor.predict(features)
		notes.append(note)

	return notes


def initialize_features(num_notes, use_octave):
	x = None
	return x


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
		pitch = util.one_hot_to_integer(note[2])
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

	return feature


# Ben do this function
def generate_midi_file(midi_file_name, notes):
	
	# save to midi_file_name
	pass


def main():
	### Parameters ###
	num_notes = 4
	use_octave = False
	use_NN = True
	song_length = 300
	num_throwaway_notes = 20
	midi_file_name = "generatedSongs/our_first_song.midi"

	### Compose a song! ###
	notes = generate_song_notes(num_notes, use_NN, song_length, num_throwaway_notes, use_octave)
	generate_midi_file(midi_file_name, notes)

if __name__ == '__main__':
	main()






















