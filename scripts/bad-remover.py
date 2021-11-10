import pickle
import os

with open("bad_file.list", "rb") as p:
    data = pickle.load(p)

for name in data:
    os.remove(os.path.join("non_texture", name))
