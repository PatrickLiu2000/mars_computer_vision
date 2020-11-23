import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time

"""
This approach uses BRIEF descriptors for image matching.
Using opencv, we can detect features using the STAR feature
detector (derived from CenSurE). Then we can extract these
features as BREIF descriptors for matching.
"""

def find_and_match_brief(plane, local, num_matches="all", smoothing=False, orb=False):
    start_time = time.time()

    plane_img = cv.imread(plane, cv.IMREAD_GRAYSCALE)
    local_img = cv.imread(local, cv.IMREAD_GRAYSCALE)

    kp_plane = None
    kp_local = None
    desc_plane = None
    desc_local = None

    if smoothing:
        plane_img = cv.medianBlur(plane_img, 5)
        local_img = cv.medianBlur(local_img, 5)

    if orb:
        orb = cv.ORB_create()

        kp_plane, desc_plane = orb.detectAndCompute(plane_img, None)
        kp_local, desc_local = orb.detectAndCompute(local_img, None)
    else:
        star = cv.xfeatures2d.StarDetector_create()
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create()


        keypoints_plane = star.detect(plane_img)
        keypoints_local = star.detect(local_img)

        kp_plane, desc_plane = brief.compute(plane_img, keypoints_plane)
        kp_local, desc_local = brief.compute(local_img, keypoints_local)


    #brute force matching
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

    matches = bf.match(desc_local, desc_plane)
    matches = sorted(matches, key = lambda x:x.distance)
    if num_matches == "all":
        img_matched = cv.drawMatches(local_img, kp_local, plane_img, kp_plane, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        img_matched = cv.drawMatches(local_img, kp_local, plane_img, kp_plane, matches[:num_matches], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    match_window_size = local_img.shape
    num_matches = len(matches)

    print("Match window size: %d %d" % (match_window_size[0], match_window_size[1]))
    print("Number of matches: %d" % num_matches)

    print(time.time() - start_time, "seconds")

    return local_img, plane_img, img_matched


def show_plots(local_img, plane_img, matched_img):
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(local_img, cmap='gray')
    #ax.flat[0].set_title("Local area image")

    ax[1].imshow(matched_img[:local_img.shape[0], :local_img.shape[1]])
    #ax.flat[1].set_title("Matches on local area image")

    plt.show()

    plt.imshow(matched_img)
    plt.show()

if __name__ == '__main__':
    local, plane, matched = find_and_match_brief('img/im1/plane.png', 'img/im1/local_quarter.png', num_matches=20, orb=True, smoothing=True)
    show_plots(local, plane, matched)
