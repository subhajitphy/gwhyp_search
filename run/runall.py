import numpy as np
import os
datadir=f"/home/subhajit/Desktop/hypsearch/NanoGrav_open_mdc"

file=np.loadtxt(datadir+'/psrlist.txt',dtype=object)

with open ('run.sh','w') as f:
    f.write('#!/bin/sh\n')
    for index in range(len(file)):
        f.write('python simulated_code.py '+str(index)+' &\n')

os.system('chmod u+x ./run.sh')