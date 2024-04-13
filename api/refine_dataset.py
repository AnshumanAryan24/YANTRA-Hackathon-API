import pandas as pd
import os

dt = pd.read_csv("../datasets/data_with_MFCC.csv")

dataset = pd.DataFrame(columns=["Is_dysathria", "MFCC"])

dataset["Is_dysathria"] = dt.iloc[:, 2].values
dataset["MFCC"] = dt.iloc[:, 3].values

dataset.to_csv("modified_dataset.csv", header=True, index=False)

print(dataset)
