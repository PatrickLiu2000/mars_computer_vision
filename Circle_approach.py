import numpy as np
import os
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.feature import canny
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import distance

path = os.getcwd()

assetsdir = path + "/assets/"
mapname = "thm_dir_N-30_060 copy 3.png"
testimg = assetsdir+mapname
testmap= plt.imread(testimg)
def get_gradients(imggray: np.ndarray):
    Mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    My = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    Mx = Mx.astype(np.float64)
    My = My.astype(np.float64)
    img64 = imggray.astype(np.float64)
    gx = ndimage.correlate(img64, Mx, mode='constant', cval=0.0)
    gy = ndimage.correlate(img64, My, mode='constant', cval=0.0)
    return gx, gy
def next_to_black(x, y, imggray):
    shape = imggray.shape
    if x + 1 >= shape[1] or y+1 >= shape[0] or x-1 <0 or y-1<0:
        return False #not quite rright but safe

    if imggray[y][x+1] ==0 or imggray[y][x-1]==0 or imggray[y+1][x]==0 or imggray[y-1][x]==0 or imggray[y][x] == 0:

        # print("true")
        return True
    else:
        return False
# def get_canny(img, sig): #lets you save an edgefile temporarily so canny doesnt have to be run many times
#     edgefile = "curredge.png"
#     if not os.path.isfile(path + "/CannyEdges/" + edgefile):
#         gray = rgb2gray(img)
#         edges = canny(image=gray, sigma=sig)
#         plt.imshow(edges)
#         plt.savefig(path + "/CannyEdges/" + edgefile, dpi = 1200)
#     else:
#         edges = plt.imread(path + "/CannyEdges/" + edgefile)
#     return edges
def detect_circles(img: np.ndarray, radius:int,use_gradients = True, denoise=False, allow_overlap =False):
    sig = 3  #can be changed based on image, sigma for canny edge detector
    numcircles = 3 #Can be changed to reflect any number of circles

    gray = rgb2gray(img)
    edges = canny(image=gray, sigma=sig)

    edgesc = np.copy(edges)
    edgePixels = np.where(edges ==True)
    edgeList = []
    # gray = rgb2gray(img)
    for i in range(len(edgePixels[0])):
        edgeList.append((int(edgePixels[0][i]), int(edgePixels[1][i])))
    j=0
    while j < len(edgeList):
        # print("Len el is ", len(edgeList))
        # print("j is ", j)
        if next_to_black(edgeList[j][1], edgeList[j][0], gray):
            edgesc[edgeList[j][0]][edgeList[j][1]] = 0
            edgeList.pop(j) #remove edges caused by missing photos
        j= j+1

    # fig, ax = plt.subplots(2)
    # ax[0].imshow(edges)
    # ax[1].imshow(edgesc)
    # plt.show()
    edges = edgesc

    H = np.zeros((img.shape[0], img.shape[1])) #the hough accumulator array

    thetas = range(360)
    thetas = np.radians(thetas)
    theta = np.zeros_like(H)

    if use_gradients:
        gx, gy = get_gradients(imggray=gray)
        for i in edgeList:
            theta[i[0],i[1]] = np.arctan2(float(gy[i[0], i[1]]), float(gx[i[0], i[1]]))

    for pixel in edgeList:
        # print("On pixel ", pixel, " of ", len(edgeIndex))
        if use_gradients:
            thetas = [theta[pixel[0]][pixel[1]]]
        for t in thetas:
            a=pixel[1] + (radius*np.cos(t))
            b=pixel[0] + (radius*np.sin(t))
            a = int(a)
            b = int(b)
            if(a<H.shape[1] and b<H.shape[0] and a>=0 and b>=0):
                H[b,a] = H[b,a] + 1

    if denoise:
        percentile = .90 #only value top 10% of H values
        min = np.amin(H)
        max = np.amax(H)
        thresh = ((max-min)*percentile)+min

        Hnew = (H >= thresh) * H
        H = Hnew
        numcircles = int(len(np.where(H != 0))/2)
    centers = []  # the return value, a list of coordinates for circle centers
    Htemp = np.copy(H)
    for i in range(numcircles):
        max = np.amax(Htemp)
        # print("Max votes was ", max)
        center = np.where(Htemp == max)
        # print("center is located ", center )
        center = [center[1][0], center[0][0], radius]
        centers.append(center) #centers holds location and radius
        Htemp[center[1]][center[0]] = -1 #to let us find a different center
        if not allow_overlap:
            for row in range(len(Htemp)):
                for col in range(len(Htemp[0])):
                    point = (col, row)
                    if distance.euclidean(point, [center[0], center[1]]) < (2*radius):
                        Htemp[point[1]][point[0]] = -1



    return centers, H


def display(H, centers, img):
    # print("displaying")

    fig, axs = plt.subplots(3)
    axs[1].imshow(H)
    for i in centers:
        circ = Circle((i[0], i[1]), i[2], color = 'red',fill=False)
        axs[0].add_artist(circ)
    axs[0].imshow(img)
    sig=3
    gray = rgb2gray(img)
    axs[2].imshow(canny(image=gray,sigma=sig))
    # plt.show()
    msg = "Radius = "+ str(centers[0][2])
    axs[0].set_title(msg)
    print("New Figure")
    plt.figure() #should do multiple windows

def main(): # run to find centers of an image and save them
    for i in range(1,10):
        rad = i*10
        print("Finding circles radius ", rad)
        c, h = detect_circles(testmap, rad, use_gradients=False)
        np.save(path+"/CircleCenters/" + mapname + "_" + str(rad) + ".npy", c) #saves centers as the name of the source underscore the radius.npy
        display(h, c, testmap)
    plt.show()

# c, h = detect_circles(testmap, 60, use_gradients=False)
# display(h, c, testmap)

def chamfer_dist(X, Y):# X is points in map, Y is points in image to be searched for
    dists = []
    pts = [] # for recordkeeping
    for point in Y:
        mindist = np.Infinity
        minpt = [0, 0]
        for xpoint in X:
            # print("Point ", point)
            # print("XPoint ", xpoint)
            d = distance.euclidean([point[0], point[1]], [xpoint[0], xpoint[1]])
            if d < mindist:
                minpt = [xpoint[0], xpoint[1]]
                mindist =d
        dists.append(mindist)
        pts.append(minpt)

    cd = 0
    for i in dists:
        cd = cd + i
    cd = float(cd)/len(dists)
    return cd
# main()