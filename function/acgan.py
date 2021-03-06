import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch

import time


os.makedirs("crop_images", exist_ok=True)

home_dir = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_epochs", type=int, default=300, help="number of epochs of training"
)
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
)
parser.add_argument(
    "--n_classes", type=int, default=4, help="number of classes for dataset"
)
parser.add_argument(
    "--img_size", type=int, default=32, help="size of each image dimension"
)
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=100, help="interval between image sampling"
)
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax()
        )

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


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


# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

"""
Configure data loader for USA dataset
rice:0 cotton:1 rice:2 soybean:3
"""
# get training dataset
folderPath = os.path.join(
    home_dir, "Documents/Semi_GANs/semi_gan/function/images/trainData"
)
img_list = os.listdir(folderPath)
img_num = len(img_list)
train_data = np.empty((0, 1, 32, 32), np.int16)
train_label = np.empty((0, 1), np.int16)
for img_file in img_list:
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
    data = np.array(Image.open(os.path.join(folderPath, img_file))).reshape(1, 32, 32)
    train_data = np.append(train_data, np.array([data]), axis=0)
    train_label = np.append(train_label, np.array([[label]]))
print(train_data.shape)
train_dataset = Spectrum2D(train_data, train_label)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# get testing dataset
folderPath = os.path.join(
    home_dir, "Documents/Semi_GANs/semi_gan/function/images/testData"
)
img_list = os.listdir(folderPath)
img_num = len(img_list)
test_data = np.empty((0, 1, 32, 32), np.int16)
test_label = np.empty((0, 1), np.int16)
for img_file in img_list:
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
    data = np.array(Image.open(os.path.join(folderPath, img_file))).reshape(1, 32, 32)
    test_data = np.append(test_data, np.array([data]), axis=0)
    test_label = np.append(test_label, np.array([[label]]))
print(test_data.shape)
test_dataset = Spectrum2D(test_data, test_label)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

# get unlabel dataset
folderPath = os.path.join(
    home_dir, "Documents/Semi_GANs/semi_gan/function/images/unlabelData"
)
img_list = os.listdir(folderPath)
img_num = len(img_list)
unlabel_data = np.empty((0, 1, 32, 32), np.int16)
unlabel_label = np.empty((0, 1), np.int16)
for img_file in img_list:
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
    data = np.array(Image.open(os.path.join(folderPath, img_file))).reshape(1, 32, 32)
    unlabel_data = np.append(unlabel_data, np.array([data]), axis=0)
    unlabel_label = np.append(unlabel_label, np.array([[label]]))
print(unlabel_data.shape)
unlabel_dataset = Spectrum2D(unlabel_data, unlabel_label)
unlabel_dataloader = DataLoader(dataset=unlabel_dataset, batch_size=64, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated images ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num % 4 for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(
        gen_imgs.data,
        "crop_images/acgan_%d.png" % batches_done,
        nrow=n_row,
        normalize=True,
        padding=0,
    )


# ----------
#  Training
# ----------

test_acc_list = []
for epoch in range(opt.n_epochs):
    # train model by unlabel dataset
    for i, sample in enumerate(unlabel_dataloader):

        imgs = sample["spectral"]

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))

        # -----------------
        #  Train Generator by unlabel sample
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(
            LongTensor(np.random.randint(0, opt.n_classes, batch_size))
        )

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (
            adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator by unlabel sample
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_dataloader),
                d_loss.item(),
                g_loss.item(),
            )
        )
    # train model by training dataset
    for i, sample in enumerate(train_dataloader):
        imgs = sample["spectral"]
        labels = sample["label"]

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(
            LongTensor(np.random.randint(0, opt.n_classes, batch_size))
        )

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (
            adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (
            adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)
        ) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (
            adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)
        ) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate(
            [real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0
        )
        gt = np.concatenate(
            [labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0
        )
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_dataloader),
                d_loss.item(),
                100 * d_acc,
                g_loss.item(),
            )
        )
        batches_done = epoch * len(train_dataloader) + i

        # get test accuracy and generate images
        if batches_done % opt.sample_interval == 0:
            # generate images
            sample_image(n_row=10, batches_done=batches_done)
            # get test accuracy
            print("calculating test accuracy".center(160, "-"))
            test_accuracy = 0.0
            test_accuracy_corn = 0.0
            test_accuracy_cotton = 0.0
            test_accuracy_rice = 0.0
            test_accuracy_soybean = 0.0
            test_num = 0

            for test_index, test_sample in enumerate(test_dataloader):
                test_imgs = sample["spectral"]
                test_labels = sample["label"]

                test_batch_size = test_imgs.shape[0]

                # Configure input
                test_real_imgs = Variable(test_imgs.type(FloatTensor))
                test_labels = Variable(test_labels.type(LongTensor))

                # Loss for real images
                test_real_pred, test_real_aux = discriminator(test_real_imgs)

                # Calculate discriminator accuracy
                pred = test_real_aux.data.cpu().numpy()
                gt = test_labels.data.cpu().numpy()
                gc = np.argmax(pred, axis=1)

                t_acc = np.mean(gc == gt)
                t_acc_corn = np.sum((gc == gt) & (gt == 0)) / np.sum((gt == 0))
                t_acc_cotton = np.sum((gc == gt) & (gt == 1)) / np.sum((gt == 1))
                t_acc_rice = np.sum((gc == gt) & (gt == 2)) / np.sum((gt == 2))
                t_acc_soybean = np.sum((gc == gt) & (gt == 3)) / np.sum((gt == 3))

                test_accuracy = test_accuracy + t_acc
                test_accuracy_corn = test_accuracy_corn + t_acc_corn
                test_accuracy_cotton = test_accuracy_cotton + t_acc_cotton
                test_accuracy_rice = test_accuracy_rice + t_acc_rice
                test_accuracy_soybean = test_accuracy_soybean + t_acc_soybean

                test_num = test_num + 1

            test_accuracy = 100.0 * (test_accuracy / test_num)
            test_accuracy_corn = 100.0 * (test_accuracy_corn / test_num)
            test_accuracy_cotton = 100.0 * (test_accuracy_cotton / test_num)
            test_accuracy_rice = 100.0 * (test_accuracy_rice / test_num)
            test_accuracy_soybean = 100.0 * (test_accuracy_soybean / test_num)

            test_acc_list.append(
                (
                    test_accuracy,
                    test_accuracy_corn,
                    test_accuracy_cotton,
                    test_accuracy_rice,
                    test_accuracy_soybean,
                )
            )

            print(
                "[D test acc: %f%% | corn acc: %f%% | cotton acc: %f%% | rice acc: %f%% | soybean acc: %f%%]"
                % (
                    test_accuracy,
                    test_accuracy_corn,
                    test_accuracy_cotton,
                    test_accuracy_rice,
                    test_accuracy_soybean,
                )
            )

for img_index in range(120):
    sample_image(n_row=10, batches_done=10000000 + img_index)

np.save(
    "test_acc_acgan" + time.strftime("%Y%m%d%H%M%S", time.localtime()),
    np.array(test_acc_list),
)
