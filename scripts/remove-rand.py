import os
import sys
import random

amount = int(sys.argv[1])
files = [f for f in os.listdir(".") if ".py" not in f]
for _ in range(amount):
    rm = random.randint(0,len(files))
    os.remove(files[rm])
    files.pop(rm)
