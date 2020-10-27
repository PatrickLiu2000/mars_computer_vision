import urllib.request
# N-30
for i in range(90, 331, 30):
    urllib.request.urlretrieve("http://www.mars.asu.edu/data/thm_dir/large/thm_dir_N-30_{}.png".format(str(i).zfill(3)),
                       "./assets/thm_dir_N-30_{}.jpg".format(str(i).zfill(3)))

# N-60
for i in range(0, 331, 30):
    urllib.request.urlretrieve("http://www.mars.asu.edu/data/thm_dir/large/thm_dir_N-60_{}.png".format(str(i).zfill(3)),
                       "./assets/thm_dir_N-30_{}.jpg".format(str(i).zfill(3)))
# N-90
for i in range(0, 331, 30):
    urllib.request.urlretrieve("http://www.mars.asu.edu/data/thm_dir/large/thm_dir_N-90_{}.png".format(str(i).zfill(3)),
                       "./assets/thm_dir_N-30_{}.jpg".format(str(i).zfill(3)))

# N00
for i in range(0, 331, 30):
    urllib.request.urlretrieve("http://www.mars.asu.edu/data/thm_dir/large/thm_dir_N00_{}.png".format(str(i).zfill(3)),
                       "./assets/thm_dir_N-30_{}.jpg".format(str(i).zfill(3)))

# N30
for i in range(0, 331, 30):
    urllib.request.urlretrieve("http://www.mars.asu.edu/data/thm_dir/large/thm_dir_N30_{}.png".format(str(i).zfill(3)),
                       "./assets/thm_dir_N-30_{}.jpg".format(str(i).zfill(3)))

# N60
for i in range(0, 331, 30):
    urllib.request.urlretrieve("http://www.mars.asu.edu/data/thm_dir/large/thm_dir_N60_{}.png".format(str(i).zfill(3)),
                       "./assets/thm_dir_N-30_{}.jpg".format(str(i).zfill(3)))