import sys
sys.path.append("../kokoro/")

import kokoro
from orderedset import OrderedSet
import csv
import pandas as pd
import numpy as np

df = pd.read_csv("data/mushrooms-3.csv")

# Acquire new column names
new_cols = OrderedSet()
for col in df:
    for val in df[col]:
        title = col+"_"+val
        new_cols.add(title)

all_new_rows = []

for idx, row in df.iterrows():
    new_row = [0] * len(new_cols)
    row_dict = row.to_dict()

    for key in row_dict:
        val = key + "_" + row_dict[key]
        match_idx = new_cols.index(val)
        new_row[match_idx] = 1

    all_new_rows.append(new_row)

data = np.array(all_new_rows)
new_df = pd.DataFrame(data=data, columns=new_cols)


input_dataset = np.matrix(new_df.loc[:, 'cap-shape_x':])
output_dataset = np.matrix(new_df.loc[:, 'class_p':'class_p'])

input_dataset = input_dataset[:][:1000]
output_dataset = output_dataset[:][:1000]

normed_input = (input_dataset - input_dataset.mean())/input_dataset.std()

ANN = kokoro.ANNetwork(0.5, 112, 20, 2, 1, 0)   
ANN.Train(normed_input, output_dataset, 1000)

