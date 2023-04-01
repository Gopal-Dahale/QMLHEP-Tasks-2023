import numpy as np
from pathlib import Path
import h5py
from sklearn.model_selection import train_test_split

data_dir = Path(__file__).absolute().parents[1]/'data'/'raw'
path =  data_dir/'QG_jets.npz'
data = np.load(path, allow_pickle=True)
X = data['X']
y = data['y']

print("X: ", X.shape)
print("y: ", y.shape)

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

print("Train: ", x_train.shape, y_train.shape)
print("Val: ", x_val.shape, y_val.shape)
print("Test: ", x_test.shape, y_test.shape)

# Save the data in form of dictionary
with h5py.File(data_dir/'QG_jets_split.h5', "w") as f:
    f.create_dataset("x_train", data=x_train, dtype="f4", compression="lzf")
    f.create_dataset("y_train", data=y_train, dtype="f4", compression="lzf")
    f.create_dataset("x_val", data=x_val, dtype="f4", compression="lzf")
    f.create_dataset("y_val", data=y_val, dtype="f4", compression="lzf")
    f.create_dataset("x_test", data=x_test, dtype="f4", compression="lzf")
    f.create_dataset("y_test", data=y_test, dtype="f4", compression="lzf")