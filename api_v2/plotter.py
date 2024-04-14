# Logic for plotting spectrograph with labels for repetition score
from process_audio import process, FS, DURATION, N_CHANNELS

import matplotlib.pyplot as plt
import librosa as lbs
import statistics as st

MIN_REP_THRESHOLD = 1350

audio_path = "./voice.wav"

final_matches = process(audio_path, "./output.json", extend_dataset=True)
final_match_keys = final_matches.keys()
print(final_matches)

audio = lbs.load(audio_path, dtype="float32")[0].tolist()

# Plotting graph
time_cd = [i for i in range(len(audio))]
plt.plot(time_cd, audio)
plt.show(block = False)

# Placing labels

# OPTION 1: Plot Maximum repetitions (almost always the silent part)
max_indices = (0, 1000)
max_rep = 0
for key in final_match_keys:
    indices = eval(key)
    if (max_rep < final_matches[key]):
        max_rep = final_matches[key]
        max_indices = indices

plt.plot(list(max_indices), [0, 0], marker = "*")

# OPTION 2: Plot all points which have repetition above a certain threshold

# Finding the minimum threshold
min_indices = (0, 1000)
min_rep = 999999
for key in final_match_keys:
    indices = eval(key)
    if (min_rep > final_matches[key]):
        min_rep = final_matches[key]
        min_indices = indices
MIN_REP_THRESHOLD = min_rep + st.stdev(final_matches.values())*1.2

for key in final_match_keys:
    indices = eval(key)
    if (final_matches[key] > MIN_REP_THRESHOLD):
        plt.plot(list(indices), [0, 0], marker = "*", color = "red")
input("Press ENTER to stop....")
