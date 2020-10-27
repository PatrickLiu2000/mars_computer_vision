import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.feature import canny
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def get_gradients(imggray: np.ndarray):
    Mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    My = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    Mx = Mx.astype(np.float64)
    My = My.astype(np.float64)
    img64 = imggray.astype(np.float64)
    gx = ndimage.correlate(img64, Mx, mode='constant', cval=0.0)
    gy = ndimage.correlate(img64, My, mode='constant', cval=0.0)
    return gx, gy

def detect_circles(img: np.ndarray, radius:int, use_gradient=True, denoise=False, quant = False):
    sig = 2#can be changed based on image, sigma for canny edge detector
    numcircles = 1 #Can be changed to reflect any number of circles
    centers = np.empty((0,2)) #the return value, a list of coordinates for circle centers
    gray = rgb2gray(img)
    edges = canny(image=gray,sigma=sig)
    edgePixels = np.where(edges ==True)
    edgeIndex = np.zeros((len(edgePixels[0]),2))
    for i in range(len(edgeIndex)):
        edgeIndex[i] = [edgePixels[0][i], edgePixels[1][i]]
    edgeIndex = edgeIndex.astype(np.uint32)
    H = np.zeros((img.shape[0], img.shape[1])) #the hough accumulator array

    thetas = range(360)
    thetas = np.radians(thetas)
    theta = np.zeros_like(H)


    gx, gy = get_gradients(imggray=gray)
    for i in edgeIndex:
        theta[i[0],i[1]] = np.arctan2(float(gy[i[0], i[1]]), float(gx[i[0], i[1]]))


    for pixel in edgeIndex:
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

    for i in range(numcircles):
        max = np.amax(H)
        center = np.where(H == max)
        center = [center[0][0], center[1][0]]
        centers = np.append(centers, (center, radius), axis=0) #centers holds location and radius
        H[center[0]][center[1]] = -1 #to let us find a different center

    return centers

def display(H, centers, img):
    fig, axs = plt.subplots(2)
    axs[1].imshow(H)
    for i in centers:
        circ = Circle((i[0][1], i[0][0]), i[1], color = 'red',fill=False)
        axs[0].add_artist(circ)
    axs[0].imshow(img)
    plt.show()
    # axs[0].set_title("Radius = 100")