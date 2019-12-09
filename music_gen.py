import util


class NNPredictor:
	def __init__(self, num_notes, use_octave):
		pass

	def predict(self, x):
		delay    = None
		duration = None
		pitch    = None
		return delay, duration, pitch


class LinearPredictor:
	def __init__(self, num_notes, use_octave):
		pass

	def predict(self, x):
		delay    = None
		duration = None
		pitch    = None
		return delay, duration, pitch


def generate_song_notes(num_notes, use_NN, song_length=300, num_throwaway_notes=20, use_octave=False):
	delays    = [] # list of integers
	durations = [] # list of integers
	pitches   = [] # list of one-hot numpy arrays

	x = initialize_features(num_notes, use_octave)

	if use_NN:
		predictor = NNPredictor(num_notes, use_octave)
	else:
		predictor = LinearPredictor(num_notes, use_octave)

	for _ in range(num_throwaway_notes):
		delay, duration, pitch  = predictor.predict(x)
		x = update_features(x, use_octave, num_notes, delay, duration, pitch)

	for _ in range(song_length):
		delay, duration, pitch  = predictor.predict(x)
		x = update_features(x, use_octave, num_notes, delay, duration, pitch)
		
		delays    += [delay]
		durations += [duration]
		pitches   += [pitch]

	return delays, durations, pitches


def initialize_features(num_notes, use_octave):
	x = None
	return x


# Ben do this function:
def update_features(x, num_notes, use_octave, new_delay, new_duration, new_pitch):
	'''
	Remove oldest note from x and add new note to x
	'''
	return x


# Ben do this function
def generate_midi_file(midi_file_name, num_notes, delays, durations, pitches, use_octave):
	# delays    is a list of integers
	# durations is a list of integers
	# pitches   is a list of one-hot numpy arrays
	
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
	delays, durations, pitches = generate_song_notes(num_notes, use_NN, song_length, num_throwaway_notes, use_octave)
	generate_midi_file(midi_file_name, num_notes, delays, durations, pitches, use_octave)

if __name__ == '__main__':
	main()






















