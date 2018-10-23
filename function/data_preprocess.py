import numpy as np
from common import logger
import cv2
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

home_dir = os.path.expanduser("~")


def load_data(data_path: str):
    try:
        features_data = np.load(data_path)["features"]
        label_data = np.load(data_path)["labels"]
        label_data = label_data.reshape(label_data.shape[0], 1)
        print("feature shape:", features_data.shape)
        print("label shape:", label_data.shape)

        # shuffle
        feature_record = np.concatenate([features_data, label_data], axis=1)
        idx = np.arange(feature_record.shape[0])
        np.random.shuffle(idx)
        feature_record = feature_record[idx, :]
        features_data = feature_record[:, :-1]
        label_data = feature_record[:, -1]
    except Exception as e:
        logger.error("load ori-data failed!".center(40, "!"))
        return None

    return features_data, label_data


def reshape_timeSeries(timeSeries_data, img_label, training_flag, num):
    """
    reshape 1 * 1098 (183 * 6) to 32 * 32 (171 * 5 + 169)
    """
    os.makedirs("./images/trainData", exist_ok=True)
    os.makedirs("./images/unlabelData", exist_ok=True)
    os.makedirs("./images/testData", exist_ok=True)

    feature_record = timeSeries_data.reshape(6, 183)
    sub_feature_record = feature_record[:, :171]
    sub_feature_record = sub_feature_record.reshape(1, -1)[:, 0:1024]
    # normalized to [0, 255], reshape to 32 * 32
    img_feature_record = ((sub_feature_record * 255).reshape(32, 32)).astype(np.int16)
    if training_flag:
        if num < 1000:
            cv2.imwrite(
                "./images/trainData/train_{0}.jpg".format(img_label), img_feature_record
            )
        else:
            cv2.imwrite(
                "./images/unlabelData/train_{0}.jpg".format(img_label),
                img_feature_record,
            )
    else:
        cv2.imwrite(
            "./images/testData/test_{0}.jpg".format(img_label), img_feature_record
        )
    return img_feature_record.reshape(1, 32, 32)


def timeSeries_to_image(
    timeSeries_dataSet: np.array,
    label_dataSet: np.array,
    img_num: int,
    training_flag: bool,
):
    """
    labels=[
        "Corn": 0,
        "Soybeans": 1,
        "Cotton": 2,
        "Rice": 3,
        "Other": 6
    ]
    """
    record_num = timeSeries_dataSet.shape[0]
    rice_num = 0
    cotton_num = 0
    corn_num = 0
    soybean_num = 0
    train_data = np.empty((0, 1, 32, 32), np.int16)
    train_label = np.empty((0, 1), np.int16)
    for record_index in range(record_num):
        timeSeries_data = timeSeries_dataSet[record_index]
        label_data = label_dataSet[record_index]
        if label_data == 3 and rice_num < img_num:
            # add rice sample to traindata
            train_data = np.append(
                train_data,
                np.array(
                    [
                        reshape_timeSeries(
                            timeSeries_data,
                            "rice_" + str(rice_num),
                            training_flag,
                            rice_num,
                        )
                    ]
                ),
                axis=0,
            )
            train_label = np.append(train_label, np.array([[label_data]]))
            rice_num = rice_num + 1
        elif label_data == 2 and cotton_num < img_num:
            # add cotton sample to traindata
            train_data = np.append(
                train_data,
                np.array(
                    [
                        reshape_timeSeries(
                            timeSeries_data,
                            "cotton_" + str(cotton_num),
                            training_flag,
                            cotton_num,
                        )
                    ]
                ),
                axis=0,
            )
            train_label = np.append(train_label, np.array([[label_data]]))
            cotton_num = cotton_num + 1
        elif label_data == 0 and corn_num < img_num:
            # add cotton sample to traindata
            train_data = np.append(
                train_data,
                np.array(
                    [
                        reshape_timeSeries(
                            timeSeries_data,
                            "corn_" + str(corn_num),
                            training_flag,
                            corn_num,
                        )
                    ]
                ),
                axis=0,
            )
            train_label = np.append(train_label, np.array([[label_data]]))
            corn_num = corn_num + 1
        elif label_data == 1 and soybean_num < img_num:
            # add cotton sample to traindata
            train_data = np.append(
                train_data,
                np.array(
                    [
                        reshape_timeSeries(
                            timeSeries_data,
                            "soybean_" + str(soybean_num),
                            training_flag,
                            soybean_num,
                        )
                    ]
                ),
                axis=0,
            )
            train_label = np.append(train_label, np.array([[label_data]]))
            soybean_num = soybean_num + 1
        else:
            continue

    print(train_data.shape)
    print(train_label.shape)
    return train_data, train_label


def image_to_timeSeries(img_path: str):
    data = np.array(Image.open(img_path))
    data = data.flatten()
    img_file = os.path.basename(img_path)
    if img_file.split("_")[1] == "corn":
        label = 0
    elif img_file.split("_")[1] == "cotton":
        label = 1
    elif img_file.split("_")[1] == "rice":
        label = 2
    elif img_file.split("_")[1] == "soybean":
        label = 3
    else:
        print("fuck label number")
        return None, None
    return np.append(data, label)
    # index = range(data.shape[0])
    # plt.plot(index, data, "r", linewidth=2)
    # plt.savefig(img_path.replace(".jpg", "_curve.jpg"))


def gatherimage_to_timeSeries(img_path: str):
    data = np.array(Image.open(img_path))
    data = data.flatten()
    img_file = os.path.basename(img_path)
    if img_file.split("_")[1] == "corn":
        label = 0
    elif img_file.split("_")[1] == "cotton":
        label = 1
    elif img_file.split("_")[1] == "rice":
        label = 2
    elif img_file.split("_")[1] == "soybean":
        label = 3
    else:
        print("fuck label number")
        return None, None
    return np.append(data, label)
    # index = range(data.shape[0])
    # plt.plot(index, data, "r", linewidth=2)
    # plt.savefig(img_path.replace(".jpg", "_curve.jpg"))


class Spectrum2D(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        spectral = self.data[index]
        label = self.label[index]
        sample = {}
        sample["spectral"] = spectral
        sample["label"] = label
        return sample


if __name__ == "__main__":
    # # generate training dataset
    os.makedirs("images", exist_ok=True)
    train_data_path = "/home/xyz/Documents/Semi_GANs/USA_data/0401_0930_17_1_CoSoOtCoRi_L_REG_TRAIN_141516.npz"
    train_features_data, train_label_data = load_data(train_data_path)
    train_sample_num = 20000
    train_data, train_label = timeSeries_to_image(
        train_features_data, train_label_data, train_sample_num, 1
    )

    # test_data_path = "/home/xyz/Documents/Semi_GANs/USA_data/0401_0930_17_1_CoSoOtCoRi_L_REG_TEST_17.npz"
    # test_feature_data, test_label_data = load_data(test_data_path)
    # test_sample_num = 10000
    # test_data, test_label = timeSeries_to_image(
    #     test_feature_data, test_label_data, test_sample_num, 0
    # )

    # # generate dataset
    # train_folder = "/home/xyz/Documents/Semi_GANs/semi_gan/function/images/trainData"
    # train_file_list = os.listdir(train_folder)
    # train_data = np.empty([0, 1025])
    # for train_file in train_file_list:
    #     train_data = np.append(
    #         train_data,
    #         np.array([image_to_timeSeries(os.path.join(train_folder, train_file))]),
    #         axis=0,
    #     )
    # print(train_data.shape)
    # np.save("train_dataset", train_data)

    # test_folder = "/home/xyz/Documents/Semi_GANs/semi_gan/function/images/testData"
    # test_file_list = os.listdir(test_folder)
    # test_data = np.empty([0, 1025])
    # for test_file in test_file_list:
    #     test_data = np.append(
    #         test_data,
    #         np.array([image_to_timeSeries(os.path.join(test_folder, test_file))]),
    #         axis=0,
    #     )
    # print(test_data.shape)
    # np.save("test_dataset", test_data)

    # res_folder = "/home/xyz/Documents/Semi_GANs/semi_gan/function/images/testData"
    # test_file_list = os.listdir(test_folder)
    # test_data = np.empty([0, 1025])
    # for test_file in test_file_list:
    #     test_data = np.append(
    #         test_data,
    #         np.array([image_to_timeSeries(os.path.join(test_folder, test_file))]),
    #         axis=0,
    #     )
    # print(test_data.shape)
    # np.save("test_dataset", test_data)

    print("Finish!")
