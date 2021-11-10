import cv2
import os
import pickle
import multiprocessing as mp

def read(name):
    load = cv2.imread(name, cv2.IMREAD_COLOR)
    shape = cv2.resize(load, (224, 224))
    return shape

p = mp.Pool(8)
texture_files = [f"./texture/{x}" for x in os.listdir("./texture")]
all_textures = p.map(read, texture_files)
np_textures = np.stack(all_textures, 0)
with open("texture-imgs.pkl", "wb") as p:
    pickle.dump(np_textures, p)

ntexture_files = [f"./non_texture/{x}" for x in os.listdir("./non_texture")]
all_ntextures = p.map(read, ntexture_files)
np_ntextures = np.stack(all_ntextures, 0)
with open("non_texture-imgs.pkl", "wb") as p:
    pickle.dump(np_ntextures, p)
