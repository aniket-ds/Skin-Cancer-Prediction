import os
import torch
from torch.utils import data

import albumentations
import pretrainedmodels

import itertools
import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn import metrics
from torch.nn import functional as F

from wtfml.data_loaders.image.classification import ClassificationDataset
from wtfml.engine.engine import Engine
from wtfml.utils import EarlyStopping


class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=pretrained)
        self.out = nn.Linear(2048, 1)
    
    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(
            out, targets.reshape(-1, 1).type_as(out)
        )
        return out, loss


def train(fold):
    training_data_path = "../input/siic-isic-224x224-images/train/"
    df = pd.read_csv("../input/train-foldscsv/train_folds.csv")  
    device = "cuda"
    epochs = 50
    train_bs = 32
    valid_bs = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    df_train = df[df.kfold != fold].reset_index(drop=True) 
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".png") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".png") for i in valid_images]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=4
    )

    valid_dataset = ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug
    )

    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=4
    )

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode="max"
    )
    
    eng = Engine(model, optimizer, device=device)

    es = EarlyStopping(patience=5, mode="max")

    for epoch in range(epochs):
        train_loss = eng.train(train_loader)
        
        predictions, valid_loss = eng.evaluate(valid_loader)
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epoch = {epoch}, AUC = {auc}")
        scheduler.step(auc)

        es(auc, model, model_path=f"model_fold_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break


def predict(fold):
    test_data_path = "../input/siic-isic-224x224-images/test"
    df_test = pd.read_csv("../input/siic-isic-224x224-images/test")
    df_test.loc[:, "target"] = 0

    device = "cuda"
    epochs = 10
    test_bs = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    test_images = df_test.image_name.values.tolist()
    test_images = [os.path.join(test_data_path, i + ".png") for i in test_images]
    test_targets = df_test.target.values

    test_dataset = ClassificationDataset( 
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=test_aug
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        num_workers=4
    )
    
    eng = Engine(model, optimizer, device=device)

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.load_state_dict(torch.load(os.path.join(model_path, f"model{fold}.bin")))
    model.to(device)

    predictions = eng.predict(test_loader)
    return np.vstack((predictions)).ravel()


if __name__ == "__main__":
    train(fold=0)