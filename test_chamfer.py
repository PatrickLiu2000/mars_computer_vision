import Circle_approach
import numpy as np
import os

path = os.getcwd()
assetsdir = path + "/assets/"


mapname1 = "thm_dir_N-30_060 copy 2.png"
mapname2 = "thm_dir_N-30_060 copy.png"
mapname3 = "thm_dir_N-30_060 copy 3.png"

centers1 = []
for i in range(1,10):
    rad = i*10
    c = np.load(path+"/CircleCenters/" + mapname1 + "_" + str(rad) + ".npy", allow_pickle=False)

    centers1.append(c)


centers2 = []
for i in range(1,10):
    rad = i*10
    c = np.load(path+"/CircleCenters/" + mapname2 + "_" + str(rad) + ".npy", allow_pickle=False)
    centers2.append(c)

centers3 = []
for i in range(1,10):
    rad = i*10
    c = np.load(path+"/CircleCenters/" + mapname3 + "_" + str(rad) + ".npy", allow_pickle=False)
    centers3.append(c)

for i in range(1,10):
    rad = i*10
    print("Chamfer distance 1 to 2 at radius ", rad, " is ", Circle_approach.chamfer_dist(centers1[i-1], centers2[i-1]))
    print("Chamfer distance 1 to 3 at radius ", rad, " is ", Circle_approach.chamfer_dist(centers1[i-1], centers3[i-1]))
    print("Chamfer distance 2 to 3 at radius ", rad, " is ", Circle_approach.chamfer_dist(centers2[i-1], centers3[i-1]))


    print("Chamfer distance within one image (should be 0) is ",Circle_approach.chamfer_dist(centers1[i-1], centers1[i-1]))