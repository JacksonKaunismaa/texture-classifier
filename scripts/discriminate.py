import os
from PIL import Image
import numpy as np
from mahotas.features.lbp import lbp
import subprocess
import sys

def homogeneity(an_im):
    np_im = np.array(an_im)
    shape = np_im.shape
    if len(shape) != 3:
        np_im = np.repeat(np.expand_dims(np_im, 2), 3, 2)
    if shape[0] != shape[1]:   # crop image to force it to be square
        min_size = min(shape[0], shape[1])
        np_im = np_im[:min_size, :min_size, :]
        shape = np_im.shape
    rand_checks = 2*80
    rand_size = shape[0]//10
    try:
        rand_xs = np.random.randint(0, 9*shape[0]//10, rand_checks)  # image is too small for 9/10*side_length to be greater than 1, so it becomes randint(0,0)
    except ValueError:
        return 0
    rand_ys = np.random.randint(0, 9*shape[1]//10, rand_checks)
    homogeneity = 0.0
    for x1,y1,x2,y2 in zip(rand_xs[::2], rand_ys[::2], rand_xs[1::2], rand_ys[1::2]):
        region1 = np_im[x1:x1+rand_size, y1:y1+rand_size,:]
        region2 = np_im[x2:x2+rand_size, y2:y2+rand_size,:]
        dist = ((region2 - region1)**2).mean()
        if dist > 1e-8:
            homogeneity += 1./(1e-10+dist)
    return homogeneity/rand_checks

def std_feature(an_im):
    np_im = np.array(an_im)
    return np_im.std()

def binary_std(an_im):
    np_im = np.array(an_im)
    if len(np_im.shape) != 3:
        np_im = np.repeat(np.expand_dims(np_im,2), 3, 2)
    result = lbp(np_im.mean(axis=2), 2, 8)
    result /= result.sum()
    return result.std()

def get_im_features(imname):
    try:
        imload = Image.open(imname)
    except OSError:
        return (1,1), imname, 1, 150, 0.
    return imload.size, imname, binary_std(imload), std_feature(imload), homogeneity(imload)

def good_fname(an_fname):
    endings = ['.jpg', '.png', '.bmp', '.ppm', '.jpeg']
    for ending in endings:
        if ending in an_fname:
            return True
    return False

def add_to_file(new_data):
    with open("discriminate.txt", "a") as f:
        f.write(new_data+"\n")


def rule(im_check):
    im_size = im_check[0]
    score = im_check[2]*5500 + 1.3*im_check[3]  + 400*(0.1 - im_check[4])
    threshold = 282
    if im_size[0] < 40 or im_size[1] < 40:  # automatically disqualify if image is too small
        add_to_file(f"{im_check}  too small image")
        return False
    elif im_check[2] > 0.042:     # very high binary std, not good
        add_to_file(f"{im_check}  too high binary_std")
        return False
    elif im_check[2] < 0.02:    # very low binary std, good
        add_to_file(f"{im_check}  low binary_std")
        return True
    elif score < threshold:    # if binary std inconclusive, use weighted sum of std_feature and binary_std and check if it is below a threshold value
        add_to_file(f"{im_check}  weighted avg ({score})")
        return True
    add_to_file(f"{im_check}  default option (score {score})")
    return False


def main(the_dir):
    get_all = [f"{the_dir}/{x}" for x in os.listdir(the_dir) if good_fname(x)]
    print(f"Computing statistics on {the_dir} of size {len(get_all)}")
    all_sizes = list(map(get_im_features, get_all))

    all_good = [x for x in all_sizes if rule(x)]
    all_bad = [x for x in all_sizes if not rule(x)]

    print("Good size:", len(all_good))
    print(" Bad size:", len(all_bad))
    print("Percent good", float(len(all_good))/len(all_sizes))

    all_good_files = [x[1] for x in all_good]
    all_bad_files = [x[1] for x in all_bad]

    print("Displaying set of good images...")
    subprocess.call(["feh"] + all_good_files)
    print("Displaying set of bad images...")
    subprocess.call(["feh"] + all_bad_files)

    return all_good, all_bad



if __name__ == "__main__":
    all_values = []
    if os.path.exists("./discriminate.txt"):
        os.remove("./discriminate.txt")
    if len(sys.argv) >= 2:
        for next_dir in sys.argv[1:]:
            main(next_dir)
    else:
        print("Add directories containg images to get statistics")
        #main(".")

