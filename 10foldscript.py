import numpy as np
import os
import sys
name  = sys.argv[1]

batch = sys.argv[2]
onlines = []
offlines = []

for i in range(1, 11):
     os.system("python3 train.py --nr_gpu=1 --data_dir=./data/" + name + "/" + name + " --dropout_p=0.3 --batch_size="+ batch +"  --test_num=" + str(i) + " | tail -10 > " + name + str(i) + "k.txt")
     os.system("cat " + name + str(i) + "k.txt | tail -2 > " + name + str(i) + ".txt")
     f = open(name + str(i) + ".txt", 'r')
     on = float(f.readline().replace("online: ", ""))
     off = float(f.readline().replace("offline: ", ""))
     onlines.append(on)
     offlines.append(off)

on_std, off_std = np.std(onlines), np.std(offlines)
on_mean, off_mean = np.mean(onlines), np.mean(offlines)

f = open(name + "_res.txt", "w")
f.write("Offline: " + str(off_mean) + " +- "+ str(off_std) + "\n")
f.write("Online: " + str(on_mean) + " +- "+ str(on_std) + "\n")


