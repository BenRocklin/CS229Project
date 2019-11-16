import os
import util

def main():
    songPath = "./songs"
    songList = []
    for path, dirs, files in os.walk("./songs"):
        songList = files
    for song in songList:
        util.get_song_features(songPath + "/" + song, 1)

if __name__ == "__main__":
    main()