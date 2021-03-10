from keras.utils import Sequence
import os
from matplotlib import image


class MPIIDataSet(Sequence):

    def __init__(self, img_dir="./images", annotation_file="./mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"):
        self.img_arr = []



        # for file in os.listdir(img_dir):
        #     self.img_arr.append(image.imread(os.path.join('images', file)))
        #     print(self.img_arr[-1].shape)
        #     assert self.img_arr[-1].shape == (1080, 1920, 3)





if __name__ == "__main__":
    dataset = MPIIDataSet()