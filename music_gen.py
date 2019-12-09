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


def generate_song_notes(num_notes, useNN=True, song_length=300, num_throwaway_notes=20, use_octave=False):
	delays    = [] # list of integers
	durations = [] # list of integers
	pitches   = [] # list of one-hot numpy arrays

	if useNN:
		predictor = NNPredictor(num_notes, use_octave)
	else:
		predictor = LinearPredictor(num_notes, use_octave)

	x = initialize_features(num_notes, use_octave)

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



# Ben do this function:
def initialize_features(num_notes, use_octave):
	x = None
	return x



# Ben do this function:
def update_features(x, num_notes, use_octave, new_delay, new_duration, new_pitch):
	'''
	Remove oldest note from x and add new_delay, new_duration, new_pitch (in one-hot form) to x
	'''
	return x
