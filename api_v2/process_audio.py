import json
import math
import statistics as st
import librosa as lbs
import sounddevice as sd
import soundfile as sf
# from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt
import numpy as np

Z_ALPHA = 2.3263  # ALPHA=0.01 <---------------- Control Parameter
FS: int = 44100
DURATION: float = 2.00
N_CHANNELS = 2

sd.default.samplerate = FS
sd.default.channels = N_CHANNELS

def __getMeanSD(audio: int, start_index: int, len_seg: int, D: int):
    next_seg_index = start_index+len_seg
    
    seg1 = audio[start_index:next_seg_index]
    seg2 = audio[next_seg_index:min(next_seg_index+len_seg, D)]

    return st.mean(seg1), st.stdev(seg1), st.mean(seg2), st.stdev(seg2)

def __getSegments(length: int, dur: int) ->list[int]:
    # For 0-indexing, next segment starts from index count+length, else count+length-1
    # dur is int because the data we will be proceesing will be a discrete array, not a continuous distribution
    count = 0
    fully_covered = False
    indices = []
    while (count < dur):
        indices.append(count)
        count += length
    if count+length > dur:
        fully_covered = False
    else:
        fully_covered = True
    
    return fully_covered, indices

def __saveAudio(JSON_path: str, final_matches, file_name: str = "./output.json"):
    # data = json.dumps(final_matches)
    json.dump(final_matches, open(file_name, "w"))

def __rowAverage(row):
    return np.average(row)

def process(audio_path: str, JSON_path: str, extend_dataset: bool = False):
    # Logic for finding which parts (time stamps) of the audio file correspond to repetition of words.
    times_match_i = {}

    final_matches = {}

    audio = []
    # OPTION1:
    audio = lbs.load(audio_path, dtype="float32")[0].tolist()
    
    # TESTING values:
    # audio = [item[0] for item in [[-3.0517578e-05,  0.0000000e+00],
    # [ 0.0000000e+00,  0.0000000e+00],
    # [ 1.7944336e-02,  1.7944336e-02],
    # [ 1.8890381e-02,  1.8890381e-02],
    # [ 1.9653320e-02,  1.9622803e-02]]]
    
    D = len(audio)

    NUM_SEG_THRESHOLD = int(D*0.5)  # MIN. NO. OF SEGMENTS

    Li = 1
    i = 1
    final_i = []
    while (D//Li > NUM_SEG_THRESHOLD):
        Li = 2**i
        i += 1
        final_i.append(i)
        fc, ind = __getSegments(Li, D)

        j = 0
        len_ind = len(ind)
        while (j < len_ind-1):
            curr = ind[j]
            next = ind[j+1]
            times_match_i[f"{(curr, min(next+Li-1, D))}"] = 0
            j += 1
            x1, sd1, x2, sd2 = __getMeanSD(audio = audio, start_index = curr, len_seg = Li, D = D)

            Z = (x1 - x2)/math.sqrt(((sd1**2) + (sd2**2))/Li)
            # print(Z, curr, Li)

            if (abs(Z) >= Z_ALPHA):  # Accepted H1: mean1 != mean2
                continue
            else:
                # Accepted H0: mean1 = mean2
                # SAVE TIME STAMP
                times_match_i[f"{(curr, min(next+Li-1, D))}"] += 1
    
    # Now, all repetitions, observed in different segment lengths, have been recorded
    times_match_i = {key: value for key, value in times_match_i.items() if (value != 0)}

    L: int = D//2**(3)

    fc_final, ind_final = __getSegments(L, D)
    j = 0
    while (j < len(ind_final) - 1):
        curr1 = ind_final[j]
        next1 = ind_final[j+1]
        j += 1

        final_matches[f"{(curr1, next1)}"] = 0
        # Keeping key in this format avoids the following error:
        # TypeError: keys must be str, int, float, bool or None, not tuple
        
        for string in times_match_i.keys():
            start, end = eval(string)
            # Consider if segment lies in this time stamp or vice-versa
            if ( ((start <= curr1) and (end >= next1))
                or ((start >= curr1) and (end <= next1)) ):
                final_matches[f"{(curr1, next1)}"] += 1

        # Remove entries with 0 occurences
        final_matches = {key: value for key, value in final_matches.items() if (value != 0)}
    
    if (extend_dataset):
        __saveAudio(JSON_path, final_matches)
    
    return final_matches

def main():
    print("Started")
    raw_data = sd.rec(int(FS*DURATION))
    sd.wait()
    sf.write("./voice.wav", raw_data, FS)
    final_matches = process("./voice.wav", "./output.json", extend_dataset=True)
    print(final_matches)
    

    time_cd = [i for i in range(len(raw_data))]
    plt.plot(time_cd, raw_data)
    plt.show(block = True)

if __name__ == "__main__":
    main()
