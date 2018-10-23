import numpy as np
from common import logger
import os
from PIL import Image
import matplotlib.pyplot as plt

home_dir = os.path.expanduser("~")


# from 10 samples * 10 samples image generate timeSeries data
def gatherimage_to_timeSeries(img_path: str):
    ori_data = np.array(Image.open(img_path))[:, :, 0]
    all_num = 0
    img_size = 10
    res_data = np.empty([0, 1025])
    for row_index in range(img_size):
        for col_index in range(img_size):
            label = all_num % 4
            all_num = all_num + 1
            data = ori_data[
                row_index * 32 : (row_index + 1) * 32,
                col_index * 32 : (col_index + 1) * 32,
            ]
            data = data.flatten()
            res_data = np.append(res_data, np.array([np.append(data, label)]), axis=0)

    print(all_num)
    print(res_data.shape)
    return res_data


if __name__ == "__main__":
    gen_folder = "/home/xyz/Documents/Semi_GANs/semi_gan/function/images/genData"
    gen_file_list = os.listdir(gen_folder)
    gen_data = np.empty([0, 1025])
    for gen_file in gen_file_list:
        gen_temp = gatherimage_to_timeSeries(os.path.join(gen_folder, gen_file))
        gen_data = np.concatenate([gen_data, gen_temp], axis=0)
    print(gen_data.shape)
    np.save("gene_dataset", gen_data)

    print("Finish!")
