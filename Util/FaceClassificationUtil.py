__author__ = 'Thushan Ganegedara'

import shutil as sh
import os
import csv
from PIL import Image

class FaceClassifUtil(object):

    def load_data_to_one_folder(self,data_dir):
        sub_dir_names = [x[0] for x in os.walk(data_dir)]
        for sub_d in sub_dir_names[1:]:
            f = []
            for (dirpath, dirnames, filenames) in os.walk(sub_d):
                f.extend(filenames)
                break
            for f_name in f:
                src = sub_d + '\\' + f_name
                dst = 'all_images\\' + f_name
                if '.jpg' in f_name:
                    sh.copyfile(src, dst)

    def load_data(self,img_dir,lbl_dir):

        def get_thumbnail(filename,size):
            try:
                im = Image.open(filename)
                im.thumbnail(size, Image.ANTIALIAS)
                img = im.convert('LA')
            except IOError:
                print "cannot create thumbnail for '%s'" % filename
            return img

        with open(lbl_dir+'\\fold_0_data.txt') as inputfile:
            results = list(csv.reader(inputfile))

        f_names = []
        f_ids = []
        ages = []
        genders = []
        for s in results[1:]:
            new_s = s[0]+s[1]
            s_tokens = new_s.split('\t')
            f_names.append(s_tokens[1])
            f_ids.append(s_tokens[2])
            ages.append(s_tokens[3])
            genders.append(s_tokens[4])
            img1 = get_thumbnail(img_dir+"\\"+'coarse_tilt_aligned_face.'+f_ids[-1]+'.'+f_names[-1],[64,64])
            l = list(img1.getdata())
            print 'what'



if __name__ == '__main__':
    util = FaceClassifUtil()
    util.load_data("all_images","labels")