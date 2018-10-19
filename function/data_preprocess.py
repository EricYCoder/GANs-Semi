import numpy as np
from common import logger
import cv2
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

home_dir = os.path.expanduser("~")


def load_data(data_path: str):
    try:
        features_data = np.load(data_path)["features"]
        label_data = np.load(data_path)["labels"]
        print("feature shape:", features_data.shape)
        print("label shape:", label_data.shape)
    except Exception as e:
        logger.error("load ori-data failed!".center(40, "!"))
        return None

    return features_data, label_data


def reshape_timeSeries(timeSeries_data, img_label, training_flag):
    """
    reshape 1 * 1098 (183 * 6) to 32 * 32 (170 * 2 + 171 * 4)
    """
    feature_record = timeSeries_data.reshape(6, 183)
    sub_feature_record = feature_record[:, :171]
    sub_feature_record = sub_feature_record.reshape(1, -1)[:, 0:1024]
    # normalized to [0, 255], reshape to 32 * 32
    img_feature_record = ((sub_feature_record * 255).reshape(32, 32)).astype(np.int16)
    if training_flag:
        cv2.imwrite("./images/{0}_train.jpg".format(img_label), img_feature_record)
    else:
        cv2.imwrite("./images/{0}_test.jpg".format(img_label), img_feature_record)
    return img_feature_record.reshape(1, 32, 32)


def timeSeries_to_image(
    timeSeries_dataSet: np.array,
    label_dataSet: np.array,
    img_num_train: int,
    img_num_test: int,
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
    os.makedirs("images", exist_ok=True)
    record_num = timeSeries_dataSet.shape[0]
    rice_num = 0
    cotton_num = 0
    corn_num = 0
    soybean_num = 0
    img_num = img_num_test + img_num_train
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
                            1 if rice_num < img_num_train else 0,
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
                            1 if cotton_num < img_num_train else 0,
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
                            1 if corn_num < img_num_train else 0,
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
                            1 if soybean_num < img_num_train else 0,
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
    data_path = (
        "/home/xyz/Documents/semi_gan/0401_0930_17_1_CoSoOtCoRi_L_REG_TRAIN_141516.npz"
    )
    features_data, label_data = load_data(data_path)
    train_sample_num = 1000
    test_sample_num = 3000
    train_data, train_label = timeSeries_to_image(
        features_data, label_data, train_sample_num, test_sample_num
    )
    train_dataset = Spectrum2D(train_data, train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    for i, sample in enumerate(train_loader):
        print(i)
        imgs = sample["spectral"]
        labels = sample["label"]
        print(imgs.numpy().shape)

    print("Finish!")
