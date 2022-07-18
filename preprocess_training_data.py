import pandas as pd
import numpy as np

df = pd.read_csv("data/training_data.csv")
print(df.head())
print(df.describe(include='all'))
print(len(df))

# runtime_log,ground_truth,entity_log_template,variable_entity_type_array
all_gdths = []
all_split_input = []

for index, row in df.iterrows():
    gdth = []
    for el1, el2 in zip(str(row['runtime_log']).split(" "), str(row['entity_log_template']).split(" ")):
        if "<" in el2 and ">" in el2:
            gdth.append(el2.replace("<", "").replace(">", ""))
        else:
            gdth.append("CONSTANT")
    all_gdths.append(gdth)
    all_split_input.append(str(row['runtime_log']).split(" "))

df['input_data'] = all_split_input
df['ner_labels'] = all_gdths

print(df.head())

# 80, 10, 10 -> train, validate, test
train, validate, test = np.split(df.sample(frac=1, random_state=42),[int(.8*len(df)), int(.9*len(df))])

print("Len train ", len(train))
print("Len validate ", len(validate))
print("Len test ", len(test))

# write training data
train.to_csv("training_data/train.csv", index=False)
validate.to_csv("training_data/validate.csv", index=False)
test.to_csv("training_data/test.csv", index=False)