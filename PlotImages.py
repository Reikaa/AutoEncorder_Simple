__author__ = 'Thushan Ganegedara'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import sqrt
class PlotImages(object):

    def load_and_plot_img(self,dir_name, f_prefix, num_files = 0):
        fig = plt.figure()
        sqr = sqrt(num_files)
        r_idx = int(sqr) + 1
        c_idx = int(sqr)
        for i in range(num_files):
            a = fig.add_subplot(r_idx,c_idx,i+1)
            img = mpimg.imread(dir_name+"\\"+f_prefix+str(i+1)+".jpg")
            imgplot = plt.imshow(img)


plotImg = PlotImages()
plotImg.load_and_plot_img('Data','image_',20)