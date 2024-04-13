'''
To be called only once to initialize all the MFCCs.
'''

import pandas as pd
import librosa  as lbs
import os

# CONSTANTS
SAMPLE_RATE = 22050
# TRACK_DURATION = 30 # measured in seconds  <---------- Change accordingly
# SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def __getMFCC(file_path: str, track_duration: int, n_fft: int, num_mfcc: int, hop_length: int, num_segments: int):
    samples_per_track = track_duration*SAMPLE_RATE
    signal, sample_rate = lbs.load(file_path, sr=SAMPLE_RATE)
    for d in range(num_segments):
        # calculate start and finish sample for current segment
        samples_per_segment = int(samples_per_track / num_segments)
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        # mfcc = lbs.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = lbs.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
    
    return mfcc.tolist()


def initDataFile(dataset_path: str = r"../datasets/Dysarthria and Non Dysarthria/",  n_fft: int = 128, n_mfcc: int = 12, hop_length: int = 32, n_seg = 5) -> bool:
    '''
    Function for preprocessing all files containing data ONCE, so that we can use the results in DNN later.
    Saves the data as a new JSON file and returns true if preprocessing was successful.

    @param dataset_path (str): Path to the file 'data_with_path.csv'
    @param save_file_path (str): Path to JSON file to which data has to be written (overwrite data)
    '''

    '''
    Logic for labelling each MFCC:
    Each .wav file has labels - Gender,Is_dysarthria,Wav_path,Txt_path,Prompts
    (All will be encoded when preprocessing when applying NN, not here)

    Go to path in Wav_path, then generate MFCC and put in the cell.
    '''

    # ---------------------------- GENERATING PATHS TO .wav FILES ----------------------------
    gender_list = []
    is_dysathria_list = []
    wav_path_list = []

    count = 0
    for (root, folders, files) in os.walk(top=dataset_path):
        # if (files[0] == "data_with_path.csv"):
        if (count == 0):
            count += 1
            continue

        if (len(files) == 0):
            continue

        if (len(folders) == 0) and (root[-3:] == "Wav"):
            # Now, we are at a .wav file
            txt_path = root[:-3]+"Txt"
            if (not os.path.exists(txt_path)):
                # If Txt folder does not exist for this Wav folder, skip this folder
                continue
            
            gender = ""
            if ("Female" in root):
                state = "Female"
            else:
                state = "Male"
            state = ""
            if (("Female_dysarthria" in root) or ("Male_Dysarthria" in root)):
                state = "Yes"
            else:
                state = "No"

            for wav_path in files:
                wav_path_list.append(os.path.join(root, wav_path))
                is_dysathria_list.append(state)
                gender_list.append(gender)
    
    dataset = pd.DataFrame(columns=["Index", "Gender", "Is_dysathria", "MFCC", "wav_path"])
    dataset["Index"] = [i for i in range(len(gender_list))]
    dataset["Gender"] = gender_list
    dataset["Is_dysathria"] = is_dysathria_list
    dataset["wav_path"] = wav_path_list


    # ---------------------------- GENERATING MFCCs FOR .wav FILES ----------------------------
    MFCC_list = []

    for index, wav_path in enumerate(dataset["wav_path"]):
        try:
            MFCC_list.append(__getMFCC(wav_path, 
                                        track_duration=lbs.get_duration(path=wav_path),
                                        n_fft=n_fft, num_mfcc=n_mfcc, hop_length=hop_length, num_segments=n_seg)
            )
        except:
            MFCC_list.append("")
    
    dataset["MFCC"] = MFCC_list

    print(dataset[0:10])
    print(dataset["Is_dysathria"][0:10])

    dataset.to_csv("../datasets/data_with_MFCC.csv", header=False, index=False)


initDataFile(n_fft=32, hop_length=8)
