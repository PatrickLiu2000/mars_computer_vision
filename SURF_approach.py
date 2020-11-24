#imports
import imageio
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import mahotas
from mahotas.features import surf

#reads image that we want to identify and displays
image_read_time_s = time.perf_counter()
image = imageio.imread('./assets/thm_dir_N-30_000.png')
# image = image[5:200, 5:200]
window = np.zeros((image.shape[0], image.shape[1]))
print(window.shape)
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# plt.show()
image_read_time_d = time.perf_counter() - image_read_time_s

#filtering image if 3 dimensional
image_filter_time_s = time.perf_counter()
if np.ndim(image) == 3:
    image_filter = image[:,:,0]
image_filter_time_d = time.perf_counter() - image_filter_time_s

#applying guassian filter
image_guassian_time_s = time.perf_counter()
guassian = mahotas.gaussian_filter(image, 5)
image_guassian_time_d = time.perf_counter() - image_guassian_time_s

#applying SURF
image_surf_time_s = time.perf_counter()
spoints = surf.surf(guassian)
spoints = np.where(spoints == None, 0, spoints)
desc = spoints[:,:2]
image_surf_time_d = time.perf_counter() - image_surf_time_s
print("Number of feature points: ", len(spoints))
print("\n")

#displaying feature points on guassian image
# fig, ax = plt.subplots(1)
# ax.imshow(guassian, cmap="gray")
# ax.axis('off')
#
# for point in range(len(spoints)):
#     rect = patches.Rectangle((spoints[point][1], spoints[point][0]), 200, 200, linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)
#
# plt.show()

total_time_d = time.perf_counter() - image_read_time_s

#display times
print("image read time: " + "{:.2f}".format(image_read_time_d), " seconds")
print("image filter time: " + "{:.2f}".format(image_filter_time_d), " seconds")
print("image guassian time: " + "{:.2f}".format(image_guassian_time_d), " seconds")
print("image surf time: " + "{:.2f}".format(image_surf_time_d), " seconds")
print("total time: " + "{:.2f}".format(total_time_d), " seconds")


directory = os.fsencode('./assets/')

for file in os.listdir(directory):

    filename = os.fsdecode(file)
    t_image = imageio.imread('./assets/' + filename)

    # filtering image if 3 dimensional
    t_image_filter_time_s = time.perf_counter()
    if np.ndim(t_image) == 3:
        t_image = t_image[:, :, 0]
    t_image_filter_time_d = time.perf_counter() - t_image_filter_time_s

    # applying guassian filter
    t_image_guassian_time_s = time.perf_counter()
    t_guassian = mahotas.gaussian_filter(t_image, 5)
    t_image_guassian_time_d = time.perf_counter() - t_image_guassian_time_s

    # displaying feature points on guassian image
    # fig, ax = plt.subplots(1)
    # ax.imshow(t_guassian, cmap="gray")
    # ax.axis('off')
    #
    # for t_point in range(len(t_spoints)):
    #     t_rect = patches.Rectangle((t_spoints[t_point][1], t_spoints[t_point][0]), 200, 200, linewidth=1, edgecolor='r',
    #                              facecolor='none')
    #     ax.add_patch(t_rect)
    #
    # plt.show()

    # t_total_time_d = time.perf_counter() - t_image_read_time_s

    # display times
    # print("image read time: " + "{:.2f}".format(t_image_read_time_d), " seconds")
    # print("image filter time: " + "{:.2f}".format(t_image_filter_time_d), " seconds")
    # print("image guassian time: " + "{:.2f}".format(t_image_guassian_time_d), " seconds")
    # print("image surf time: " + "{:.2f}".format(t_image_surf_time_d), " seconds")
    # print("total time: " + "{:.2f}".format(t_total_time_d), " seconds")

    for i in range(t_image.shape[0] - window.shape[0] + 1):
        for j in range(t_image.shape[1] - window.shape[0] + 1):

            # applying SURF
            t_image_surf_time_s = time.perf_counter()
            t_spoints = surf.surf(t_guassian[i:window.shape[0]+i,j:window.shape[1]+j])
            t_spoints = np.where(t_spoints == None, 0, t_spoints)
            t_desc = t_spoints[:, :2]
            t_image_surf_time_d = time.perf_counter() - t_image_surf_time_s
            print("Number of feature points: ", len(t_spoints))
            print("\n")

            if(spoints[:,5:].shape == t_spoints[:,5:].shape):
                if np.array_equal(spoints[:,:2], t_spoints[:,:2]):
                    plt.imshow(image, cmap='gray')
                    plt.axis('off')
                    plt.imshow(t_image, cmap='gray')
                    plt.axis('off')
                    plt.show()
                    print("Found matching image")
                    break

    print("Done")