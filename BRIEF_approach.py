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

def find_and_match_brief(num_matches="all"):
    start_time = time.time()

    plane_img = cv.imread('assets/thm_dir_N-30_030.png', cv.IMREAD_GRAYSCALE)
    local_img = cv.imread('assets/local_test.png', cv.IMREAD_GRAYSCALE)

    star = cv.xfeatures2d.StarDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    orb = cv.ORB_create()

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
    local, plane, matched = find_and_match_brief(num_matches=20)
    show_plots(local, plane, matched)
