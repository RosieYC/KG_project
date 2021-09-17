# https://www.kaggle.com/c/landmark-recognition-2021
import os
from os import walk

# https://www.kaggle.com/denispotapov/inference-inception-v4-pytorch-lightning
# https://www.kaggle.com/denispotapov/train-inception-v4-pytorch-lightning
# inception_v4 & pytorch_lightning
import os
from os import walk
import cv2
import pickle

import timm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import pytorch_lightning as pl
import timm
import albumentations as A
import numpy as np
import torchmetrics
import datetime
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from pathlib import Path
from torch.utils import data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class Inception4_teacher_Model(pl.LightningModule):
    def __init__(self,
                 model_type,
                 num_classes,
                 classes_weights,
                 learning_rate=0.0001):
        super().__init__()
        # hyperparameters()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.model_type = model_type

        # load nn
        if self.model_type in ['densenet121',  # classifier
                               'densenet161',
                               'densenet169',
                               'densenet201',
                               'densenetblur121d',
                               'efficientnet_b0',
                               'efficientnet_b1',
                               'efficientnet_b1_pruned',
                               'efficientnet_b2',
                               'efficientnet_b2a',
                               'efficientnet_b3',
                               'efficientnet_b3_pruned',
                               'efficientnet_b3a',
                               'efficientnet_em',
                               'efficientnet_es',
                               'efficientnet_lite0',
                               'fbnetc_100',
                               'hrnet_w18',
                               'hrnet_w18_small',
                               'hrnet_w18_small_v2',
                               'hrnet_w30',
                               'hrnet_w32',
                               'hrnet_w40',
                               'hrnet_w44',
                               'hrnet_w48',
                               'hrnet_w64',
                               'mixnet_l',
                               'mixnet_m',
                               'mixnet_s',
                               'mixnet_xl',
                               'mnasnet_100',
                               'mobilenetv2_100',
                               'mobilenetv2_110d',
                               'mobilenetv2_120d',
                               'mobilenetv2_140',
                               'mobilenetv3_large_100',
                               'mobilenetv3_rw',
                               'semnasnet_100',
                               'spnasnet_100',
                               'tf_efficientnet_b0',
                               'tf_efficientnet_b0_ap',
                               'tf_efficientnet_b0_ns',
                               'tf_efficientnet_b1',
                               'tf_efficientnet_b1_ap',
                               'tf_efficientnet_b1_ns',
                               'tf_efficientnet_b2',
                               'tf_efficientnet_b2_ap',
                               'tf_efficientnet_b2_ns',
                               'tf_efficientnet_b3',
                               'tf_efficientnet_b3_ap',
                               'tf_efficientnet_b3_ns',
                               'tf_efficientnet_b4',
                               'tf_efficientnet_b4_ap',
                               'tf_efficientnet_b4_ns',
                               'tf_efficientnet_b5',
                               'tf_efficientnet_b5_ap',
                               'tf_efficientnet_b5_ns',
                               'tf_efficientnet_b6',
                               'tf_efficientnet_b6_ap',
                               'tf_efficientnet_b6_ns',
                               'tf_efficientnet_b7',
                               'tf_efficientnet_b7_ap',
                               'tf_efficientnet_b7_ns',
                               'tf_efficientnet_b8',
                               'tf_efficientnet_b8_ap',
                               'tf_efficientnet_cc_b0_4e',
                               'tf_efficientnet_cc_b0_8e',
                               'tf_efficientnet_cc_b1_8e',
                               'tf_efficientnet_el',
                               'tf_efficientnet_em',
                               'tf_efficientnet_es',
                               'tf_efficientnet_l2_ns',
                               'tf_efficientnet_l2_ns_475',
                               'tf_efficientnet_lite0',
                               'tf_efficientnet_lite1',
                               'tf_efficientnet_lite2',
                               'tf_efficientnet_lite3',
                               'tf_efficientnet_lite4',
                               'tf_mixnet_l',
                               'tf_mixnet_m',
                               'tf_mixnet_s',
                               'tf_mobilenetv3_large_075',
                               'tf_mobilenetv3_large_100',
                               'tf_mobilenetv3_large_minimal_100',
                               'tf_mobilenetv3_small_075',
                               'tf_mobilenetv3_small_100',
                               'tf_mobilenetv3_small_minimal_100',
                               'tv_densenet121',
                               'tf_efficientnetv2_b0',
                               'tf_efficientnetv2_l',
                               'eca_efficientnet_b0',
                               'efficientnet_b2_pruned',
                               'efficientnet_b4',
                               'efficientnet_b5',
                               'efficientnet_b6',
                               'efficientnet_b7',
                               'efficientnet_b8',
                               'efficientnet_cc_b0_4e',
                               'efficientnet_cc_b0_8e',
                               'efficientnet_cc_b1_8e',
                               'efficientnet_el',
                               'efficientnet_el_pruned',
                               'efficientnet_es_pruned',
                               'efficientnet_l2',
                               'efficientnet_lite1',
                               'efficientnet_lite2',
                               'efficientnet_lite3',
                               'efficientnet_lite4',
                               'efficientnetv2_l',
                               'efficientnetv2_m',
                               'efficientnetv2_rw_m',
                               'efficientnetv2_rw_s',
                               'efficientnetv2_s',
                               'gc_efficientnet_b0', ]:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif self.model_type in ['adv_inception_v3',  # fc
                                 'ecaresnet26t',
                                 'ecaresnet50d',
                                 'ecaresnet50d_pruned',
                                 'ecaresnet50t',
                                 'ecaresnet101d',
                                 'ecaresnet101d_pruned',
                                 'ecaresnet269d',
                                 'ecaresnetlight',
                                 'gluon_inception_v3',
                                 'gluon_resnet18_v1b',
                                 'gluon_resnet34_v1b',
                                 'gluon_resnet50_v1b',
                                 'gluon_resnet50_v1c',
                                 'gluon_resnet50_v1d',
                                 'gluon_resnet50_v1s',
                                 'gluon_resnet101_v1b',
                                 'gluon_resnet101_v1c',
                                 'gluon_resnet101_v1d',
                                 'gluon_resnet101_v1s',
                                 'gluon_resnet152_v1b',
                                 'gluon_resnet152_v1c',
                                 'gluon_resnet152_v1d',
                                 'gluon_resnet152_v1s',
                                 'gluon_resnext50_32x4d',
                                 'gluon_resnext101_32x4d',
                                 'gluon_resnext101_64x4d',
                                 'gluon_senet154',
                                 'gluon_seresnext50_32x4d',
                                 'gluon_seresnext101_32x4d',
                                 'gluon_seresnext101_64x4d',
                                 'gluon_xception65',
                                 'ig_resnext101_32x8d',
                                 'ig_resnext101_32x16d',
                                 'ig_resnext101_32x32d',
                                 'ig_resnext101_32x48d',
                                 'inception_v3',
                                 'res2net50_14w_8s',
                                 'res2net50_26w_4s',
                                 'res2net50_26w_6s',
                                 'res2net50_26w_8s',
                                 'res2net50_48w_2s',
                                 'res2net101_26w_4s',
                                 'res2next50',
                                 'resnest14d',
                                 'resnest26d',
                                 'resnest50d',
                                 'resnest50d_1s4x24d',
                                 'resnest50d_4s2x40d',
                                 'resnest101e',
                                 'resnest200e',
                                 'resnest269e',
                                 'resnet18',
                                 'resnet18d',
                                 'resnet26',
                                 'resnet26d',
                                 'resnet34',
                                 'resnet34d',
                                 'resnet50',
                                 'resnet50d',
                                 'resnet101d',
                                 'resnet152d',
                                 'resnet200d',
                                 'resnetblur50',
                                 'resnext50_32x4d',
                                 'resnext50d_32x4d',
                                 'resnext101_32x8d',
                                 'selecsls42b',
                                 'selecsls60',
                                 'selecsls60b',
                                 'seresnet50',
                                 'seresnet152d',
                                 'seresnext26d_32x4d',
                                 'seresnext26t_32x4d',
                                 'seresnext50_32x4d',
                                 'skresnet18',
                                 'skresnet34',
                                 'skresnext50_32x4d',
                                 'ssl_resnet18',
                                 'ssl_resnet50',
                                 'ssl_resnext50_32x4d',
                                 'ssl_resnext101_32x4d',
                                 'ssl_resnext101_32x8d',
                                 'ssl_resnext101_32x16d',
                                 'swsl_resnet18',
                                 'swsl_resnet50',
                                 'swsl_resnext50_32x4d',
                                 'swsl_resnext101_32x4d',
                                 'swsl_resnext101_32x8d',
                                 'swsl_resnext101_32x16d',
                                 'tf_inception_v3',
                                 'tv_resnet34',
                                 'tv_resnet50',
                                 'tv_resnet101',
                                 'tv_resnet152',
                                 'tv_resnext50_32x4d',
                                 'wide_resnet50_2',
                                 'wide_resnet101_2',
                                 'xception', ]:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.fc.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif self.model_type in ['dla34',
                                 'dla46_c',
                                 'dla46x_c',
                                 'dla60',
                                 'dla60_res2net',
                                 'dla60_res2next',
                                 'dla60x',
                                 'dla60x_c',
                                 'dla102',
                                 'dla102x',
                                 'dla102x2',
                                 'dla169',
                                 'dpn68',
                                 'dpn68b',
                                 'dpn92',
                                 'dpn98',
                                 'dpn107',
                                 'dpn131',
                                 ]:
            model = timm.create_model(model_type, pretrained=True)
            if self.model_type == 'dla34':
                model.fc = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            elif self.model_type in ['dla46_c',
                                     'dla46x_c',
                                     'dla60x_c', ]:
                model.fc = nn.Conv2d(256, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            elif self.model_type in ['dla60',
                                     'dla60_res2net',
                                     'dla60_res2next',
                                     'dla60x',
                                     'dla102',
                                     'dla102x',
                                     'dla102x2',
                                     'dla169']:
                model.fc = nn.Conv2d(1024, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            elif self.model_type in ['dpn68', 'dpn68b', ]:
                model.fc = nn.Conv2d(832, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            elif self.model_type in ['dpn92', 'dpn98', 'dpn107', 'dpn131', ]:
                model.fc = nn.Conv2d(2688, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            self.model = model
        elif self.model_type in ['cspdarknet53',  # head.fc
                                 'cspresnet50',
                                 'cspresnext50',
                                 'dm_nfnet_f0',
                                 'dm_nfnet_f1',
                                 'dm_nfnet_f2',
                                 'dm_nfnet_f3',
                                 'dm_nfnet_f4',
                                 'dm_nfnet_f5',
                                 'dm_nfnet_f6',
                                 'ese_vovnet19b_dw',
                                 'ese_vovnet39b',
                                 'gernet_l',
                                 'gernet_m',
                                 'gernet_s',
                                 'nf_regnet_b1',
                                 'nf_resnet50',
                                 'nfnet_l0c',
                                 'regnetx_002',
                                 'regnetx_004',
                                 'regnetx_006',
                                 'regnetx_008',
                                 'regnetx_016',
                                 'regnetx_032',
                                 'regnetx_040',
                                 'regnetx_064',
                                 'regnetx_080',
                                 'regnetx_120',
                                 'regnetx_160',
                                 'regnetx_320',
                                 'regnety_002',
                                 'regnety_004',
                                 'regnety_006',
                                 'regnety_008',
                                 'regnety_016',
                                 'regnety_032',
                                 'regnety_040',
                                 'regnety_064',
                                 'regnety_080',
                                 'regnety_120',
                                 'regnety_160',
                                 'regnety_320',
                                 'repvgg_a2',
                                 'repvgg_b0',
                                 'repvgg_b1',
                                 'repvgg_b1g4',
                                 'repvgg_b2',
                                 'repvgg_b2g4',
                                 'repvgg_b3',
                                 'repvgg_b3g4',
                                 'resnetv2_50x1_bitm',
                                 'resnetv2_50x1_bitm_in21k',
                                 'resnetv2_50x3_bitm',
                                 'resnetv2_50x3_bitm_in21k',
                                 'resnetv2_101x1_bitm',
                                 'resnetv2_101x1_bitm_in21k',
                                 'resnetv2_101x3_bitm',
                                 'resnetv2_101x3_bitm_in21k',
                                 'resnetv2_152x2_bitm',
                                 'resnetv2_152x2_bitm_in21k',
                                 'resnetv2_152x4_bitm',
                                 'resnetv2_152x4_bitm_in21k',
                                 'rexnet_100',
                                 'rexnet_130',
                                 'rexnet_150',
                                 'rexnet_200',
                                 'tresnet_l',
                                 'tresnet_l_448',
                                 'tresnet_m',
                                 'tresnet_m_448',
                                 'tresnet_xl',
                                 'tresnet_xl_448',
                                 'vgg11',
                                 'vgg11_bn',
                                 'vgg13',
                                 'vgg13_bn',
                                 'vgg16',
                                 'vgg16_bn',
                                 'vgg19',
                                 'vgg19_bn',
                                 'xception41',
                                 'xception65',
                                 'xception71', ]:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.head.fc.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif self.model_type in ['deit_base_distilled_patch16_224']:
            model = timm.create_model(model_type, pretrained=True)
            in_features_head = model.head.in_features
            in_features_head_dist = model.head_dist.in_features
            model.head = nn.Linear(in_features_head, self.num_classes)
            model.head_dist = nn.Linear(in_features_head_dist, self.num_classes)
            print(model)
            self.model = model
        elif self.model_type in ['ens_adv_inception_resnet_v2',  # classif
                                 'inception_resnet_v2', ]:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.classif.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif self.model_type in ['inception_v4',  # last_linear
                                 'legacy_senet154',
                                 'legacy_seresnet18',
                                 'legacy_seresnet34',
                                 'legacy_seresnet50',
                                 'legacy_seresnet101',
                                 'legacy_seresnet152',
                                 'legacy_seresnext26_32x4d',
                                 'legacy_seresnext50_32x4d',
                                 'legacy_seresnext101_32x4d',
                                 'nasnetalarge',
                                 'pnasnet5large', ]:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.last_linear.in_features
            print('in_features: ', in_features)
            model.last_linear = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif self.model_type in ['vit_base_patch16_224',  # head
                                 'vit_base_patch16_224_in21k',
                                 'vit_base_patch16_384',
                                 'vit_base_patch32_224_in21k',
                                 'vit_base_patch32_384',
                                 'vit_base_resnet50_224_in21k',
                                 'vit_base_resnet50_384',
                                 'vit_deit_base_distilled_patch16_224',
                                 'vit_deit_base_distilled_patch16_384',
                                 'vit_deit_base_patch16_224',
                                 'vit_deit_base_patch16_384',
                                 'vit_deit_small_distilled_patch16_224',
                                 'vit_deit_small_patch16_224',
                                 'vit_deit_tiny_distilled_patch16_224',
                                 'vit_deit_tiny_patch16_224',
                                 'vit_large_patch16_224',
                                 'vit_large_patch16_224_in21k',
                                 'vit_large_patch16_384',
                                 'vit_large_patch32_224_in21k',
                                 'vit_large_patch32_384',
                                 'vit_small_patch16_224',
                                 'cait_m36_384',
                                 'cait_m48_448',
                                 'cait_s24_224',
                                 'cait_s24_384',
                                 'cait_s36_384',
                                 'cait_xs24_384',
                                 'cait_xxs24_224',
                                 'cait_xxs24_384',
                                 'cait_xxs36_224',
                                 'cait_xxs36_384',
                                 'coat_lite_mini',
                                 'coat_lite_small',
                                 'coat_lite_tiny',
                                 'coat_mini',
                                 'coat_tiny',
                                 'convit_base',
                                 'convit_small',
                                 'convit_tiny', ]:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.head.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
            self.model = model
        else:
            assert (
                False
            ), f"model_type '{self.model_type}' not implemented. Please, choose from {MODELS}"

        if classes_weights:
            self.classes_weights = torch.FloatTensor(classes_weights)  # .cuda()
            self.loss_func = nn.CrossEntropyLoss(weight=self.classes_weights)
        else:
            self.loss_func = nn.CrossEntropyLoss()

            #         self.f1 = torchmetrics.F1(num_classes=self.num_classes)

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

        # will be used during inference

    def forward(self, x):
        return self.model(x)

        # Using custom or multiple metrics (default_hp_metric=False)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

        # logic for a single training step

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        train_loss = self.loss(output, y)

        # training metrics
        output = torch.argmax(output, dim=1)
        acc = accuracy(output, y)

        self.log('train_loss', train_loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return train_loss

        # logic for a single validation step

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        val_loss = self.loss(output, y)

        # validation metrics
        output = torch.argmax(output, dim=1)
        acc = accuracy(output, y)

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return val_loss

        # logic for a single testing step

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        test_loss = self.loss(output, y)

        # validation metrics
        output = torch.argmax(output, dim=1)
        acc = accuracy(output, y)

        self.log('test_loss', test_loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

        return test_loss

        # def training_epoch_end(self, outputs):
        #     self.log('train_f1_epoch', self.f1.compute())
        #     self.f1.reset()
        #
        # def validation_epoch_end(self, outputs):
        #     self.log('val_f1_epoch', self.f1.compute(), prog_bar=True)
        #     self.f1.reset()
        #
        # def test_epoch_end(self, outputs):
        #     self.log('test_f1_epoch', self.f1.compute())
        #     self.f1.reset()

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        gen_sched = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(gen_opt, gamma=0.999),  # , verbose=False),
                     'interval': 'epoch'}

        return [gen_opt], [gen_sched]


class Inception4_Dataset(data.Dataset):
    def __init__(self,
                 data,
                 input_resize,
                 augments=None,
                 preprocessing=None):
        super().__init__()
        self.imgs, self.labels = data
        self.input_resize = (input_resize, input_resize)
        self.augments = augments
        self.preprocessing = preprocessing

    def __len__(self):
        return (len(self.imgs))

    def __getitem__(self, item):
        img_path = str(self.imgs[item])
        label = self.labels[item]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, self.input_resize, interpolation=cv2.INTER_NEAREST)
        if self.augments:
            augmented = self.augments(image=img)
            img = augmented['image']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img)
            img = preprocessed['image']
        return img, label


class Inception4_DataModule(pl.LightningDataModule):
    def __init__(self,
                 model_type,
                 batch_size,
                 data_dir,
                 input_resize,
                 input_resize_test,
                 mean,
                 std,
                 augment_p=0.7,
                 images_ext='jpg'
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_resize = input_resize
        self.input_resize_test = input_resize_test
        self.mean = mean,
        self.std = std,
        self.augment_p = augment_p
        self.images_ext = images_ext

        # get preprocessing and augmentation
        transforms_composed = self._get_transforms()
        self.augments, self.preprocessing = transforms_composed

        self.dataset_train, self.dataset_val, self.dataset_test = None, None, None

        # dataset transforms: normalize and toTensor

    def _get_transforms(self):
        transforms = []

        if self.mean is not None:
            transforms += [A.Normalize(mean=self.mean, std=self.std)]

        transforms += [ToTensorV2(transpose_mask=True)]
        preprocessing = A.Compose(transforms)

        return self._get_train_transforms(self.augment_p), preprocessing

        # dataset augmentation. I used albumentation library. You can add your own augmentations.

    def _get_train_transforms(self, p):
        return A.Compose([
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.3),
            A.GaussNoise(p=0.4),
            A.OneOf([A.MotionBlur(p=0.5),
                     A.MedianBlur(blur_limit=3, p=0.5),
                     A.Blur(blur_limit=3, p=0.1)], p=0.5),
            A.OneOf([A.CLAHE(clip_limit=2),
                     A.Sharpen(),
                     A.Emboss(),
                     A.RandomBrightnessContrast()], p=0.5),
        ], p=p)

    # It's a main procedure for preparing your data. I have writen it especially for landmark-recognition-2021 dataset.
    # Below this procedure there is another one. You should check it too.
    def setup(self, stage=None):
        path = Path(self.data_dir)
        train_df = pd.read_csv(r"/data/ppr/inception_v4/DATA/train.csv")
        landmark = train_df.landmark_id.value_counts()
        # 138982    6272
        # 126637    2231
        # 20409     1758
        # 83144     1741
        # 113209    1135

        # we take only 5 most frequent classes. Your can change count of classes - for example to 1000.
        l = landmark[:1000].index.values
        # [138982 126637  20409  83144 113209]
        freq_landmarks_df = train_df[train_df['landmark_id'].isin(l)]
        # 157294   004d531bd0f43001        20409
        # 157295   00817c48cc418bf3        20409
        # 157296   00b6e3050572f46f        20409
        # 157297   00cd3cd816e56e2b        20409
        # 157298   01294d171c6a5a99        20409
        # ...                   ...          ...
        # 1083511  ffc7eb8c2f216c85       138982
        # 1083512  ffcc4c0f61b254c0       138982
        # 1083513  ffdb7ec1b164f10f       138982
        # 1083514  ffe4999cd81b3775       138982
        # 1083515  fffa5d86cbbae4cb       138982
        image_ids = freq_landmarks_df['id'].tolist()
        # ['004d531bd0f43001', '00817c48cc418bf3', '00b6e3050572f46f', '00cd3cd816e56e2b', '01294d171c6a5a99', '01384cd07195f5c5', '018c8d3ab336c05b', '01a1c3c6ff3ebf36']
        landmark_ids = freq_landmarks_df['landmark_id'].tolist()
        # [20409, 20409, 20409, 20409, 20409, 20409, 20409, 20409, 20409, 20409, 20409, 20409, 20409, 20409, 20409, ...]
        # convert from classes to codes 0, 1, 2, 3, ... etc.
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(l)
        # [4 3 0 1 2]
        self.num_classes = len(np.unique(encoded))
        # 5
        # save labels dict to file. We will use this file during inference.
        with open(r"/data/ppr/inception_v4/label_encoder.pkl", 'wb') as le_dump_file:
            pickle.dump(label_encoder, le_dump_file)

        # mapping classes and codes
        mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        #  {20409: 0, 83144: 1, 113209: 2, 126637: 3, 138982: 4}

        # image_ids landmark_ids dict
        im_land_dict = dict((k, i) for k, i in zip(image_ids, landmark_ids))
        # {'004d531bd0f43001': 20409, '00817c48cc418bf3': 20409}

        # get paths of all images in dataset
        print('Unpacking images...')
        path_list = [os.path.join(dirpath, filename) for dirpath, _, filenames in os.walk(str(path)) for filename in
                     filenames if filename.endswith('.jpg')]

        # get filenames from paths
        filenames = []
        for path in tqdm(path_list, disable=True):
            filename, _ = os.path.splitext(path.split('/')[-1])  # /
            filenames.append(filename)

        # find intersection of images filenames of our frequent classes and all filenames
        ind_dict = dict((k, i) for i, k in enumerate(filenames))
        inter = set(ind_dict).intersection(image_ids)
        indices = [ind_dict[x] for x in inter]

        # find paths of images of our frequent classes
        image_ids_paths = []
        for ind in indices:
            image_ids_paths.append(path_list[ind])

        # find landmarks ids for our images
        labels_ids = []
        for img in tqdm(image_ids_paths, disable=True):
            filename, _ = os.path.splitext(img.split('/')[-1])  # /
            land_id = im_land_dict[filename]
            labels_ids.append(mapping[int(land_id)])

        # you can set classes_weights but I skipped this step
        self.classes_weights = None

        image_ids_paths = [Path(p) for p in image_ids_paths]
        print(image_ids_paths)
        print(labels_ids)
        # set train images and labels
        train_files, val_test_files, train_labels, val_test_labels = train_test_split(image_ids_paths, labels_ids,
                                                                                      test_size=0.3, random_state=42,
                                                                                      stratify=landmark_ids)
        print(f'train_files: {len(train_files)}, train_labels: {len(train_labels)}')

        train_data = train_files, train_labels

        # set val and test images and labels
        val_files, test_files, val_labels, test_labels = train_test_split(val_test_files, val_test_labels,
                                                                          test_size=0.5, random_state=42,
                                                                          stratify=val_test_labels)

        print(f'val_files: {len(val_files)}, val_labels: {len(val_labels)}')
        val_data = val_files, val_labels

        print(f'test_files: {len(test_files)}, test_labels: {len(test_labels)}')
        test_data = test_files, test_labels

        self.sampler = None
        # self.sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        if stage == 'fit' or stage is None:
            self.dataset_train = Inception4_Dataset(
                data=train_data,
                input_resize=self.input_resize,
                augments=self.augments,
                preprocessing=self.preprocessing)

            # notice that we don't add augments for val dataset but only for training
            self.dataset_val = Inception4_Dataset(
                data=val_data,
                input_resize=self.input_resize,
                preprocessing=self.preprocessing)

            self.dims = tuple(self.dataset_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.dataset_test = Inception4_Dataset(
                data=test_data,
                input_resize=self.input_resize_test,
                preprocessing=self.preprocessing)

            self.dims = tuple(self.dataset_test[0][0].shape)

    # !!!ATENTION!!! Below there is another setup procedure. If you plan to use this pipeline for your Image Classification problems
    # just put images of every class to separate folders with names of classes.
    # Uncomment this procedure and everything will work.

    # Example of folder structure:
    # -- animals
    #    -- cat
    #       -- cat1.png
    #       -- cat2.png
    #       -- cat3.png
    #    -- dog
    #       -- dog1.png
    #       -- dog2.png
    #       -- dog3.png

    # You can see such folder structure in "input" section on the right side: folder - simpsons.

    #     def setup(self, stage=None):
    #         # Assign train/val datasets for use in dataloaders

    #         path = Path(self.data_dir)

    #         train_val_files = list(path.rglob('*.' + self.images_ext))
    #         train_val_labels = [path.parent.name for path in train_val_files]

    #         label_encoder = LabelEncoder()
    #         encoded = label_encoder.fit_transform(train_val_labels)
    #         self.num_classes = len(np.unique(encoded))

    #         # save labels dict to file
    #         with open('label_encoder.pkl', 'wb') as le_dump_file:
    #             pickle.dump(label_encoder, le_dump_file)

    #         train_files, val_test_files = train_test_split(train_val_files, test_size=0.3, stratify=train_val_labels)

    #         train_labels = [path.parent.name for path in train_files]
    #         train_labels = label_encoder.transform(train_labels)
    #         train_data = train_files, train_labels

    #         class_weights = []
    #         count_all_files = 0
    #         for root, subdir, files in os.walk(self.data_dir):
    #             if len(files) > 0:
    #                 class_weights.append(len(files))
    #                 count_all_files += len(files)

    #         self.classes_weights = [x / count_all_files for x in class_weights]
    #         print('classes_weights', self.classes_weights)

    #         sample_weights = [0] * len(train_files)

    #         for idx, (data, label) in enumerate(zip(train_files, train_labels)):
    #             class_weight = self.classes_weights[label]
    #             sample_weights[idx] = class_weight

    #         self.sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    #         # self.classes_weights = [round(x / sum(list(Counter(sorted(train_labels)).values())), 2) for x in
    #         #                        list(Counter(sorted(train_labels)).values())]

    #         # without test step
    #         # val_labels = [path.parent.name for path in val_test_files]
    #         # val_labels = label_encoder.transform(val_labels)
    #         # val_data = val_test_files, val_labels

    #         # with test step
    #         val_test_labels = [path.parent.name for path in val_test_files]
    #         val_files, test_files = train_test_split(val_test_files, test_size=0.5, stratify=val_test_labels)

    #         val_labels = [path.parent.name for path in val_files]
    #         val_labels = label_encoder.transform(val_labels)

    #         test_labels = [path.parent.name for path in test_files]
    #         test_labels = label_encoder.transform(test_labels)

    #         val_data = val_files, val_labels
    #         test_data = test_files, test_labels

    #         if stage == 'fit' or stage is None:
    #             self.dataset_train = ICPDataset(
    #                 data=train_data,
    #                 input_resize=self.input_resize,
    #                 augments=self.augments,
    #                 preprocessing=self.preprocessing)

    #             self.dataset_val = ICPDataset(
    #                 data=val_data,
    #                 input_resize=self.input_resize,
    #                 preprocessing=self.preprocessing)

    #             self.dims = tuple(self.dataset_train[0][0].shape)

    #         # Assign test dataset for use in dataloader(s)
    #         if stage == 'test' or stage is None:
    #             self.dataset_test = ICPDataset(
    #                 data=test_data,
    #                 input_resize=self.input_resize_test,
    #                 preprocessing=self.preprocessing)

    #             self.dims = tuple(self.dataset_test[0][0].shape)

    def train_dataloader(self):
        if self.sampler:
            loader = DataLoader(self.dataset_train, batch_size=self.batch_size, sampler=self.sampler, num_workers=4)
        else:
            loader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

        return loader

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=4)


def train():
    project_name = 'landmarks'
    data_dir = r"/data/ppr/inception_v4/DATA/train/"
    image_ext = 'jpg'
    augment_p = 0.8
    init_lr = 0.0003
    early_stop_patience = 5
    max_epochs = 5
    progress_bar_refresh_rate = 10
    model_name = ['inception_v4']
    models = []
    for m in model_name:
        if m in ['deit_base_distilled_patch16_224',
                 'deit_base_distilled_patch16_384',
                 'deit_base_patch16_224',
                 'deit_base_patch16_384',
                 'deit_small_distilled_patch16_224',
                 'deit_small_patch16_224',
                 'deit_tiny_distilled_patch16_224',
                 'deit_tiny_patch16_224', ]:
            continue
        model_dict = {}
        model_dict['batch_size'] = 8
        model_dict['model_type'] = m
        model_dict['im_size'] = None
        model_dict['im_size_test'] = None
        models.append(model_dict)

    models_for_training = []
    for m in tqdm(models, disable=True):
        model_data = {'model': m}
        # get mean, std, image_size of every model
        mod = timm.create_model(model_data['model']['model_type'], pretrained=False)
        model_mean = list(mod.default_cfg['mean'])
        model_std = list(mod.default_cfg['std'])

        # get input size
        im_size = 0
        im_size_test = 0

        print(model_data['model']['model_type'] + ' input size is ' + str(mod.default_cfg['input_size']))

        if model_data['model']['im_size']:
            im_size = model_data['model']['im_size']
        else:
            im_size = mod.default_cfg['input_size'][1]

        if model_data['model']['im_size_test']:
            im_size_test = model_data['model']['im_size']
        else:
            im_size_test = mod.default_cfg['input_size'][1]
        print('img_size: ', im_size)

        # create datamodule object
        print(data_dir, augment_p, model_data['model']['model_type'], \
              model_data['model']['batch_size'], im_size, im_size_test, model_mean, model_std)

        dm = Inception4_DataModule(data_dir=data_dir,
                                   augment_p=augment_p,
                                   model_type=model_data['model']['model_type'],
                                   batch_size=model_data['model']['batch_size'],
                                   input_resize=im_size,
                                   input_resize_test=im_size_test,
                                   mean=model_mean,
                                   std=model_std)

        # To access the x_dataloader we need to call prepare_data and setup.
        dm.setup()

        # Init our model
        model = Inception4_teacher_Model(model_type=model_data['model']['model_type'],
                                 num_classes=dm.num_classes,
                                 classes_weights=None,
                                 learning_rate=init_lr)

        # Initialize a trainer
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=early_stop_patience,
            verbose=True,
            mode='min'
        )

        # logs for tensorboard
        experiment_name = model_data['model']['model_type']
        logger = TensorBoardLogger(os.path.join(r"/data/ppr/inception_v4/", project_name, experiment_name))

        checkpoint_name = experiment_name + '_' + '_{epoch}_{val_loss:.3f}_{val_acc:.3f}_{val_f1_epoch:.3f}'

        # mertics
        checkpoint_callback_loss = ModelCheckpoint(monitor='val_loss',
                                                   mode='min',
                                                   filename=checkpoint_name,
                                                   verbose=True,
                                                   save_top_k=3,
                                                   save_last=False)

        checkpoint_callback_acc = ModelCheckpoint(monitor='val_acc',
                                                  mode='max',
                                                  filename=checkpoint_name,
                                                  verbose=True,
                                                  save_top_k=3,
                                                  save_last=False)

        checkpoints = [checkpoint_callback_acc, checkpoint_callback_loss, early_stop_callback]
        callbacks = checkpoints

        # create Trainer
        trainer = pl.Trainer(max_epochs=max_epochs,
                             progress_bar_refresh_rate=progress_bar_refresh_rate,
                             gpus=1,
                             logger=logger,
                             callbacks=callbacks,
                             # amp_level='02',
                             # precision=16
                             )

        model_data['icp_datamodule'] = dm
        model_data['icp_model'] = model
        model_data['icp_trainer'] = trainer

        models_for_training.append(model_data)

    # train process
    for model in models_for_training:
        print('##################### START Training ' + model['model']['model_type'] + '... #####################')

        # Train the model
        model['icp_trainer'].fit(model['icp_model'], model['icp_datamodule'])

        # Evaluate the model on the held out test set
        results = model['icp_trainer'].test()[0]

        # save test results
        best_checkpoint = 'best_checkpoint: ' + model['icp_trainer'].checkpoint_callback.best_model_path
        results['best_checkpoint'] = best_checkpoint

        filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '__test_acc_' + str(
            round(results.get('test_acc'), 4)) + '.txt'

        path = os.path.join(r"/data/ppr/inception_v4/", project_name, model['model']['model_type'])
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + '/' + filename, 'w+') as f:
            print(results, file=f)

        print('##################### END Training ' + model['model']['model_type'] + '... #####################')


#### MODELS LIST
'''
MODELS = [
    'adv_inception_v3',
    'cspdarknet53',
    'cspresnet50',
    'cspresnext50',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'densenetblur121d',
    'dla102',
    'dla102x',
    'dla102x2',
    'dla169',
    'dla34',
    'dla46_c',
    'dla46x_c',
    'dla60_res2net',
    'dla60_res2next',
    'dla60',
    'dla60x_c',
    'dla60x',
    'dm_nfnet_f0',
    'dm_nfnet_f1',
    'dm_nfnet_f2',
    'dm_nfnet_f3',
    'dm_nfnet_f4',
    'dm_nfnet_f5',
    'dm_nfnet_f6',
    'dpn107',
    'dpn131',
    'dpn68',
    'dpn68b',
    'dpn92',
    'dpn98',
    'ecaresnet101d_pruned',
    'ecaresnet101d',
    'ecaresnet269d',
    'ecaresnet26t',
    'ecaresnet50d_pruned',
    'ecaresnet50d',
    'ecaresnet50t',
    'ecaresnetlight',
    'efficientnet_b0',
    'efficientnet_b1_pruned',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b2a',
    'efficientnet_b3_pruned',
    'efficientnet_b3',
    'efficientnet_b3a',
    'efficientnet_em',
    'efficientnet_es',
    'efficientnet_lite0',
    'ens_adv_inception_resnet_v2',
    'ese_vovnet19b_dw',
    'ese_vovnet39b',
    'fbnetc_100',
    'gernet_l',
    'gernet_m',
    'gernet_s',
    'gluon_inception_v3',
    'gluon_resnet101_v1b',
    'gluon_resnet101_v1c',
    'gluon_resnet101_v1d',
    'gluon_resnet101_v1s',
    'gluon_resnet152_v1b',
    'gluon_resnet152_v1c',
    'gluon_resnet152_v1d',
    'gluon_resnet152_v1s',
    'gluon_resnet18_v1b',
    'gluon_resnet34_v1b',
    'gluon_resnet50_v1b',
    'gluon_resnet50_v1c',
    'gluon_resnet50_v1d',
    'gluon_resnet50_v1s',
    'gluon_resnext101_32x4d',
    'gluon_resnext101_64x4d',
    'gluon_resnext50_32x4d',
    'gluon_senet154',
    'gluon_seresnext101_32x4d',
    'gluon_seresnext101_64x4d',
    'gluon_seresnext50_32x4d',
    'gluon_xception65',
    'hrnet_w18_small_v2',
    'hrnet_w18_small',
    'hrnet_w18',
    'hrnet_w30',
    'hrnet_w32',
    'hrnet_w40',
    'hrnet_w44',
    'hrnet_w48',
    'hrnet_w64',
    'ig_resnext101_32x16d',
    'ig_resnext101_32x32d',
    'ig_resnext101_32x48d',
    'ig_resnext101_32x8d',
    'inception_resnet_v2',
    'inception_v3',
    'inception_v4',
    'legacy_senet154',
    'legacy_seresnet101',
    'legacy_seresnet152',
    'legacy_seresnet18',
    'legacy_seresnet34',
    'legacy_seresnet50',
    'legacy_seresnext101_32x4d',
    'legacy_seresnext26_32x4d',
    'legacy_seresnext50_32x4d',
    'mixnet_l',
    'mixnet_m',
    'mixnet_s',
    'mixnet_xl',
    'mnasnet_100',
    'mobilenetv2_100',
    'mobilenetv2_110d',
    'mobilenetv2_120d',
    'mobilenetv2_140',
    'mobilenetv3_large_100',
    'mobilenetv3_rw',
    'nasnetalarge',
    'nf_regnet_b1',
    'nf_resnet50',
    'nfnet_l0c',
    'pnasnet5large',
    'regnetx_002',
    'regnetx_004',
    'regnetx_006',
    'regnetx_008',
    'regnetx_016',
    'regnetx_032',
    'regnetx_040',
    'regnetx_064',
    'regnetx_080',
    'regnetx_120',
    'regnetx_160',
    'regnetx_320',
    'regnety_002',
    'regnety_004',
    'regnety_006',
    'regnety_008',
    'regnety_016',
    'regnety_032',
    'regnety_040',
    'regnety_064',
    'regnety_080',
    'regnety_120',
    'regnety_160',
    'regnety_320',
    'repvgg_a2',
    'repvgg_b0',
    'repvgg_b1',
    'repvgg_b1g4',
    'repvgg_b2',
    'repvgg_b2g4',
    'repvgg_b3',
    'repvgg_b3g4',
    'res2net101_26w_4s',
    'res2net50_14w_8s',
    'res2net50_26w_4s',
    'res2net50_26w_6s',
    'res2net50_26w_8s',
    'res2net50_48w_2s',
    'res2next50',
    'resnest101e',
    'resnest14d',
    'resnest200e',
    'resnest269e',
    'resnest26d',
    'resnest50d_1s4x24d',
    'resnest50d_4s2x40d',
    'resnest50d',
    'resnet101d',
    'resnet152d',
    'resnet18',
    'resnet18d',
    'resnet200d',
    'resnet26',
    'resnet26d',
    'resnet34',
    'resnet34d',
    'resnet50',
    'resnet50d',
    'resnetblur50',
    'resnetv2_101x1_bitm_in21k',
    'resnetv2_101x1_bitm',
    'resnetv2_101x3_bitm_in21k',
    'resnetv2_101x3_bitm',
    'resnetv2_152x2_bitm_in21k',
    'resnetv2_152x2_bitm',
    'resnetv2_152x4_bitm_in21k',
    'resnetv2_152x4_bitm',
    'resnetv2_50x1_bitm_in21k',
    'resnetv2_50x1_bitm',
    'resnetv2_50x3_bitm_in21k',
    'resnetv2_50x3_bitm',
    'resnext101_32x8d',
    'resnext50_32x4d',
    'resnext50d_32x4d',
    'rexnet_100',
    'rexnet_130',
    'rexnet_150',
    'rexnet_200',
    'selecsls42b',
    'selecsls60',
    'selecsls60b',
    'semnasnet_100',
    'seresnet152d',
    'seresnet50',
    'seresnext26d_32x4d',
    'seresnext26t_32x4d',
    'seresnext50_32x4d',
    'skresnet18',
    'skresnet34',
    'skresnext50_32x4d',
    'spnasnet_100',
    'ssl_resnet18',
    'ssl_resnet50',
    'ssl_resnext101_32x16d',
    'ssl_resnext101_32x4d',
    'ssl_resnext101_32x8d',
    'ssl_resnext50_32x4d',
    'swsl_resnet18',
    'swsl_resnet50',
    'swsl_resnext101_32x16d',
    'swsl_resnext101_32x4d',
    'swsl_resnext101_32x8d',
    'swsl_resnext50_32x4d',
    'tf_efficientnet_b0_ap',
    'tf_efficientnet_b0_ns',
    'tf_efficientnet_b0',
    'tf_efficientnet_b1_ap',
    'tf_efficientnet_b1_ns',
    'tf_efficientnet_b1',
    'tf_efficientnet_b2_ap',
    'tf_efficientnet_b2_ns',
    'tf_efficientnet_b2',
    'tf_efficientnet_b3_ap',
    'tf_efficientnet_b3_ns',
    'tf_efficientnet_b3',
    'tf_efficientnet_b4_ap',
    'tf_efficientnet_b4_ns',
    'tf_efficientnet_b4',
    'tf_efficientnet_b5_ap',
    'tf_efficientnet_b5_ns',
    'tf_efficientnet_b5',
    'tf_efficientnet_b6_ap',
    'tf_efficientnet_b6_ns',
    'tf_efficientnet_b6',
    'tf_efficientnet_b7_ap',
    'tf_efficientnet_b7_ns',
    'tf_efficientnet_b7',
    'tf_efficientnet_b8_ap',
    'tf_efficientnet_b8',
    'tf_efficientnet_cc_b0_4e',
    'tf_efficientnet_cc_b0_8e',
    'tf_efficientnet_cc_b1_8e',
    'tf_efficientnet_el',
    'tf_efficientnet_em',
    'tf_efficientnet_es',
    'tf_efficientnet_l2_ns_475',
    'tf_efficientnet_l2_ns',
    'tf_efficientnet_lite0',
    'tf_efficientnet_lite1',
    'tf_efficientnet_lite2',
    'tf_efficientnet_lite3',
    'tf_efficientnet_lite4',
    'tf_inception_v3',
    'tf_mixnet_l',
    'tf_mixnet_m',
    'tf_mixnet_s',
    'tf_mobilenetv3_large_075',
    'tf_mobilenetv3_large_100',
    'tf_mobilenetv3_large_minimal_100',
    'tf_mobilenetv3_small_075',
    'tf_mobilenetv3_small_100',
    'tf_mobilenetv3_small_minimal_100',
    'tresnet_l_448',
    'tresnet_l',
    'tresnet_m_448',
    'tresnet_m',
    'tresnet_xl_448',
    'tresnet_xl',
    'tv_densenet121',
    'tv_resnet101',
    'tv_resnet152',
    'tv_resnet34',
    'tv_resnet50',
    'tv_resnext50_32x4d',
    'vgg11_bn',
    'vgg11',
    'vgg13_bn',
    'vgg13',
    'vgg16_bn',
    'vgg16',
    'vgg19_bn',
    'vgg19',
    'vit_base_patch16_224_in21k',
    'vit_base_patch16_224',
    'vit_base_patch16_384',
    'vit_base_patch32_224_in21k',
    'vit_base_patch32_384',
    'vit_base_resnet50_224_in21k',
    'vit_base_resnet50_384',
    'vit_deit_base_distilled_patch16_224',
    'vit_deit_base_distilled_patch16_384',
    'vit_deit_base_patch16_224',
    'vit_deit_base_patch16_384',
    'vit_deit_small_distilled_patch16_224',
    'vit_deit_small_patch16_224',
    'vit_deit_tiny_distilled_patch16_224',
    'vit_deit_tiny_patch16_224',
    'vit_large_patch16_224_in21k',
    'vit_large_patch16_224',
    'vit_large_patch16_384',
    'vit_large_patch32_224_in21k',
    'vit_large_patch32_384',
    'vit_small_patch16_224',
    'wide_resnet101_2',
    'wide_resnet50_2',
    'xception',
    'xception41',
    'xception65',
    'xception71',
    'tf_efficientnetv2_b0',
    'tf_efficientnetv2_l',
    'cait_m36_384',
    'cait_m48_448',
    'cait_s24_224',
    'cait_s24_384',
    'cait_s36_384',
    'cait_xs24_384',
    'cait_xxs24_224',
    'cait_xxs24_384',
    'cait_xxs36_224',
    'cait_xxs36_384',
    'coat_lite_mini',
    'coat_lite_small',
    'coat_lite_tiny',
    'coat_mini',
    'coat_tiny',
    'convit_base',
    'convit_small',
    'convit_tiny',
    'deit_base_distilled_patch16_224',
    'eca_efficientnet_b0',
    'efficientnet_b2_pruned',
    'efficientnet_b4',
    'efficientnet_b5',
    'efficientnet_b6',
    'efficientnet_b7',
    'efficientnet_b8',
    'efficientnet_cc_b0_4e',
    'efficientnet_cc_b0_8e',
    'efficientnet_cc_b1_8e',
    'efficientnet_el',
    'efficientnet_el_pruned',
    'efficientnet_es_pruned',
    'efficientnet_l2',
    'efficientnet_lite1',
    'efficientnet_lite2',
    'efficientnet_lite3',
    'efficientnet_lite4',
    'efficientnetv2_l',
    'efficientnetv2_m',
    'efficientnetv2_rw_m',
    'efficientnetv2_rw_s',
    'efficientnetv2_s',
    'gc_efficientnet_b0',
]

'''
if __name__ == "__main__":
    train()
    # model = 'inception_v4'
    # Inception4_Model()
    # m = timm.create_model(model, pretrained=True )
    # print('model: ', model)
    # print('m.default_cfg: ' , m.default_cfg)
## DATASET
# class InferenceDataset(Dataset):
