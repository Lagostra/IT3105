import pandas as pd

data = pd.read_csv('data/adult_raw.txt', header=None)
for i in [1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 14]: #[1, 3, 5, 6, 7, 8, 9, 13, 14]:
    data[i] = pd.Categorical(data[i])
    data[i] = data[i].cat.codes
print(data)
data.to_csv('data/adult.txt', index=False, header=False)
