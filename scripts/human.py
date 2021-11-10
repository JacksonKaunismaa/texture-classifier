import os
import random
import sys
from PIL import Image

def get_list(dirname):
    return [f"{dirname}/{x}" for x in os.listdir(dirname)]

combo_list = (get_list("../texture") + get_list("../non_texture"))
random.shuffle(combo_list)
combo_list = combo_list[:int(sys.argv[1])]
correct = 0
total = len(combo_list)
for fname in combo_list:
    imload = Image.open(fname)
    imload.show()
    correct_resp = "n" if "non_texture" in fname else "t"
    ans = input("Texture(t) or no texture(n)?" ).lower()
    if ans == correct_resp:
        correct += 1
print(f"Out of a total possible {total}, you got {correct} right ({100.*float(correct)/total})")
