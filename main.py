import os
import numpy as np
import util

def main():
    songPath = "./songs"
    songList = []
    for path, dirs, files in os.walk("./songs"):
        songList = files
    X = None
    X_no_dur = None
    y0 = None
    y1 = None
    y2 = None
    for song in songList:
        print("Reading ", song)
        songX, songXNoDuration, songY0, songY1, songY2 = util.get_song_features(songPath + "/" + song, 2)
        if X is None:
            X = songX
            X_no_dur = songXNoDuration
            y0 = songY0
            y1 = songY1
            y2 = songY2
        else:
            X = np.vstack((X, songX))
            X_no_dur = np.vstack((X_no_dur, songXNoDuration))
            y0 = np.concatenate((y0, songY0))
            y1 = np.concatenate((y1, songY1))
            y2 = np.concatenate((y2, songY2))

if __name__ == "__main__":
    main()