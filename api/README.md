# Description about files:
## initialize_dataset.py
1. RUN this ONLY ONCE.
2. The original audio (.wav) files are not properly arranged and the CSV file provided is not consistent with the data in the folders extracted, hence we need to write our own logic to pick out each audio file which has a corresponding text file (we could have picked all audio files also, but for future expansion compatibility, say when we want to attach a tag about each audio file from the corresponding text file - for such scenarios, we keep the current logic).
3. Each audio file is passed through a function which gives the MFCCs of the file. We can use this data as a feature, or for plotting.
4. Each file is given a tag - Gender + Is_having_dysarthria
5. This [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) object has some errors while saving, so we require some separate logic for handling that file.

## refine_dataset.py
1. RUN this ONLY ONCE, AFTER initialize_dataset.py.
1. This module takes as input the file created by the [initialize_dataset.py](https://github.com/AnshumanAryan24/YANTRA_Hackathon-API/blob/main/api/initialize_dataset.py) module.
2. Then, this is used to create a [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) object.
3. The column "Gender" is not included in the new object as this column appears to (seemingly always) become corrupted while saving the file output from previous module.
4. Now, save this new object as a CSV file which can be directly used by our module containing the model (NN or Random Forest Regression).

Run these commands in the terminal:
```
python initialize_dataset.py
```
```
python refine_dataset.py
```
