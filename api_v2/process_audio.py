import json
import math
import statistics as st
import librosa as lbs
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt

Z_ALPHA = 1.2816  # ALPHA = 0.10
FS: int = 44100
DURATION: float = 2.00
N_CHANNELS = 2

sd.default.samplerate = FS
sd.default.channels = N_CHANNELS

def __getMeanSD(audio: int, start_index: int, len_seg: int, D: int):
    next_seg_index = start_index+len_seg
    seg1 = audio[start_index:next_seg_index]
    if (next_seg_index+len_seg < len(audio)):
        seg2 = audio[next_seg_index:min(next_seg_index+len_seg, D)]

    return st.mean(seg1), st.stdev(seg1), st.mean(seg2), st.stdev(seg2)

def __getSegments(length: int, dur: int) ->list[int]:
    # For 0-indexing, next segment starts from index count+length, else count+length-1
    # dur is int because the data we will be proceesing will be a discrete array, not a continuous distribution
    count = 0
    fully_covered = False
    indices = []
    while (count+length < dur):
        indices.append(count)
        count += length
    if count+length > dur:
        fully_covered = False
    else:
        fully_covered = True
    
    return fully_covered, indices

def __saveAudio(JSON_path: str, final_matches, file_name: str = "./voice.json"):
    data = json.dumps(final_matches)
    json.dump(data, open("./voice.json", "w"))

def process(audio_path: str, JSON_path: str, extend_dataset: bool = False, dataset_path: str = "./modified_dataset.csv"):
    # Logic for finding which parts (time stamps) of the audio file correspond to repetition of words.
    times_match_i: dict["{int}", (int, int)] = {}

    final_matches: list[tuple[2]] = []

    audio = []
    audio = lbs.load(audio_path, dtype="float32")[0].tolist()
    
    # audio = [item[0] for item in [[-3.0517578e-05,  0.0000000e+00],
    # [ 0.0000000e+00,  0.0000000e+00],
    # [ 1.7944336e-02,  1.7944336e-02],
    # [ 1.8890381e-02,  1.8890381e-02],
    # [ 1.9653320e-02,  1.9622803e-02]]]
    
    D = len(audio)

    NUM_SEG_THRESHOLD = int(D*0.5)  # MIN. NO. OF SEGMENTS

    Li = 1
    i = 1
    while (D//Li > NUM_SEG_THRESHOLD):
        Li = 2**i
        i += 1
        fc, ind = __getSegments(Li, D)

        j = 0
        len_ind = len(ind)
        while (j < len_ind-1):
            curr = ind[j]
            next = ind[j+1]
            j += 1
            x1, sd1, x2, sd2 = __getMeanSD(audio = audio, start_index = curr, len_seg = Li, D = D)

            Z = (x1 - x2)/math.sqrt(((sd1**2) + (sd2**2))/Li)
            # print(Z, curr, Li)

            if (abs(Z) >= Z_ALPHA):  # Accepted H1: mean1 != mean2
                continue
            # else
            # Accepted H0: mean1 = mean2
            # SAVE TIME STAMP
            times_match_i[f"{Li}"] = (curr, min(curr+2*Li, D))
    
    # Now, all repetitions, observed in different segment lengths, have been recorded
    L: int = D//2**(i-1)

    fc_final, ind_final = __getSegments(L, D)
    j = 0
    while (j < len_ind - 1):
        curr1 = ind[j]
        next1 = ind[j+1]
        j += 1
        count = 0
        for Li in times_match_i.keys():
            start, end = times_match_i[Li]
            Li = eval(Li)
            # Consider if more than 5% (from both sides) segment lies in this time stamp
            # if ((start-Li*0.05) <= curr1) and (end <= (next1 + Li*0.05)):
            if (start <= curr1) and (end <= next1):
                final_matches.append((start, end))
    
    if (extend_dataset):
        __saveAudio(JSON_path, final_matches)
    
    return final_matches

def main():
    print("Started")
    raw_data = sd.rec(int(FS*DURATION))
    sd.wait()
    sf.write("./output.wav", raw_data, FS)
    final_matches = process("./output.wav", "./voice.json", extend_dataset=True)
    print(final_matches)
    

    time_cd = [i for i in range(len(raw_data))]
    plt.plot(time_cd, raw_data)
    plt.show(block = True)

if __name__ == "__main__":
    main()
