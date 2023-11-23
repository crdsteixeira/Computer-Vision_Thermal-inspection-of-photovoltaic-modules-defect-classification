import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import torch
import json
import os
from torch.utils.data import random_split
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from collections import defaultdict
import random
from torch.utils.data import Subset
import cv2
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

class SolarDataset(Dataset):
  def __init__(self, metadata_path, images_path, random_state=42, gan_df=False):
    self.full_df = pd.read_json(metadata_path, orient='index').sort_index()
    self.full_df['image_filepath'] = self.full_df.image_filepath.apply(lambda x: os.path.join(images_path, x))
    self.train_df = self.full_df.groupby('anomaly_class').sample(frac=0.80, random_state=random_state)
    self.test_df = self.full_df.drop(self.train_df.index)
    self.val_df = self.train_df.groupby('anomaly_class').sample(frac=0.20, random_state=random_state)
    self.train_df = self.train_df.drop(self.val_df.index)
    self.classes_list = self.full_df.anomaly_class.value_counts().keys().tolist()
    self.num_classes = len(self.classes_list)
    if gan_df:
      self.train_df_gan = self.train_df.copy()
      max_per_class = int(self.train_df_gan.anomaly_class.value_counts().max())
      self.train_df_gan = self.train_df_gan.groupby('anomaly_class').sample(max_per_class, replace=True, random_state=random_state)

  def get_XY(self, in_df, pre_process=None, augment=False):
    def shuffle_within_group(group):
      return group.sample(frac=1).reset_index(drop=True)

    imgs = []
    labels = []
    class_counts_list = in_df.anomaly_class.value_counts()
    class_counts = max(class_counts_list) - class_counts_list

    for i, (p, c) in enumerate(in_df.values):
      image = cv2.imread(p, cv2.IMREAD_COLOR)
      if pre_process:
        imgs_aug = pre_process(image, aug_number=0, label=c)
        for im_aug in imgs_aug:
          labels.append(self.classes_list.index(c))
          imgs.append(im_aug)
      else:
        labels.append(self.classes_list.index(c))
        imgs.append(image)

    if augment:
      grouped_df = in_df.groupby('anomaly_class')
      for i, (c, group) in enumerate(grouped_df):
        imgs_to_gen = class_counts[c]
        # Shuffle the group to diversify augmentation
        group_count = 0
        while group_count < imgs_to_gen:
          group = shuffle_within_group(group)
          for j, (p, c) in enumerate(group.values):
            if group_count >= imgs_to_gen:
              break
            imgs_aug = pre_process(cv2.imread(p, cv2.IMREAD_COLOR), aug_number=1, label=c)
            for im_aug in imgs_aug:
              labels.append(self.classes_list.index(c))
              imgs.append(im_aug)
              group_count += 1

    return torch.tensor(np.array(imgs, dtype=np.float32)), torch.tensor(np.array(labels, dtype=np.float32))

class TrainTest:
  def __init__(self, model, loss_fn, batch_size, solar_dataset, train_df, device, pre_process=None, augment=False, random_seed=42):
    self.model = model
    self.batch_size = batch_size
    self.solar_dataset = solar_dataset
    self.device = device
    self.loss_fn = loss_fn

    # Get Train dataloader
    x_train, y_train = solar_dataset.get_XY(train_df, pre_process, augment=augment)
    train_dataset = TensorDataset(x_train, y_train)
    self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
 
    # Get Validation datal
    self.val_data = solar_dataset.get_XY(solar_dataset.val_df, pre_process, augment=False)

    # Get Test datal
    self.test_data = solar_dataset.get_XY(solar_dataset.test_df, pre_process, augment=False)

  # Function to save the model
  def saveModel(self, path):
    torch.save(self.model.state_dict(), path)

  # Function to load the model
  def loadModel(self, path):
     self.model.load_state_dict(torch.load(path))

  # Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
  def train(self, optimizer, num_epochs, path_model, scheduler=None, verbatim=True):
    self.model.train()
    self.model.to(self.device)

    best_accuracy = 0.0
    best_loss = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
      train_loss = 0
      train_acc = 0

      if scheduler:
        scheduler.step()

      self.model.train()
      for i, (images, labels) in enumerate(self.train_dataloader, 0):
        # 0. Get the inputs
        data_inputs = images.to(self.device)
        data_labels = labels.to(self.device)
        data_labels_one_hot = nn.functional.one_hot(data_labels.to(torch.int64), num_classes=self.solar_dataset.num_classes)

        # 1. Forward pass
        y_pred = self.model(data_inputs)
        y_pred = y_pred.squeeze(dim=1)

        # 2. Calculate  and accumulate loss
        loss = self.loss_fn(y_pred, data_labels_one_hot.to(torch.float32))
        train_loss += loss.item()

        train_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (train_pred_class == data_labels).sum().item() / len(train_pred_class)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward() #(create_graph=True)

        # 5. Optimizer step
        optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      with torch.no_grad():
        _, val_loss, val_acc = self.test()

      # Adjust metrics to get average loss and accuracy per batch
      train_acc = train_acc / len(self.train_dataloader)
      train_loss = train_loss / len(self.train_dataloader)
      train_losses.append(train_loss)
      train_accs.append(train_acc)
      val_losses.append(val_loss)
      val_accs.append(val_acc)

      # we want to save the model if the accuracy is the best
      path = "./myModel_" +str(epoch)+ ".pth"
      self.saveModel(path)
      if val_acc > best_accuracy:
        self.saveModel(path_model)
        best_loss = val_loss
        best_accuracy = val_acc
        best_epoch = epoch
        if verbatim:
          print('Best Epoch #', epoch,' Validation Loss=', best_loss, " Validation Accu=", best_accuracy )

    # Plot the loss and accuracy curves
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(range(1, num_epochs + 1), train_losses, color='orange', label="Training Loss")
    axs[0].plot(range(1, num_epochs + 1), val_losses, color='c', label="Validation Loss")
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(range(1, num_epochs + 1), train_accs, color='orange', label="Training Accuracy")
    axs[1].plot(range(1, num_epochs + 1), val_accs, color='c', label="Validation Accuracy")
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    plt.tight_layout()  # ensure the two axes are on the same scale
    plt.title('Training Validation Loss and Accuracy Curves')
    plt.show()
    return best_loss, best_accuracy, best_epoch

  def test(self):
    # Put model in eval mode
    self.model.eval()

    x_test, y_test = self.test_data
    test_dataset = TensorDataset(x_test, y_test)
    dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    test_loss = 0
    test_acc = 0
    pred_classes = []
    with torch.no_grad():
      for (x, y) in dataloader:
        test_labels = y.to(self.device)
        test_labels_one_hot = nn.functional.one_hot(test_labels.to(torch.int64), num_classes=self.solar_dataset.num_classes)
        y_test_pred = self.model(x.to(self.device))
        test_loss += self.loss_fn(y_test_pred, test_labels_one_hot.to(torch.float32)).item()
        y_test_pred_class = torch.argmax(torch.softmax(y_test_pred, dim=1), dim=1)
        test_acc += (y_test_pred_class == test_labels).sum().item() / len(y_test_pred_class)
        pred_classes.extend(y_test_pred_class)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    print("Loss =", test_loss, "  Accuracy=", test_acc)
    return pred_classes, test_loss, test_acc

  def get_predictions(self, data):
    # Put model in eval mode
    self.model.eval()

    x_in, y_in = data
    dataset = TensorDataset(x_in, y_in)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    labels = []
    probs = []
    preds = []
    with torch.no_grad():
      for (x, y) in dataloader:
        y_prob = torch.softmax(self.model(x.to(self.device)), dim=1)
        y_pred = torch.argmax(y_prob, dim=1)
        labels.append(y.cpu())
        probs.append(y_prob.cpu())
        preds.append(y_pred.cpu())

    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)
    preds = torch.cat(preds, dim=0)
    return labels.numpy(), probs.numpy(), preds.numpy()

  def plot_confusion_matrix(self):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    # Build confusion matrix
    labels, _, pred_labels = self.get_predictions(self.test_data)
    cm = confusion_matrix(labels, pred_labels)
    df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=self.solar_dataset.classes_list, columns=self.solar_dataset.classes_list)

    # Plot confusion matrix using seaborn
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='.2f', ax=ax)
    plt.show()

  def plot_class_metrics(self):
    f1s = []
    aucs = []
    accuracies = []
    labels, pred_probs, pred_labels = self.get_predictions(self.test_data)
    for i in range(len(self.solar_dataset.classes_list)):
        f1 = f1_score(labels, pred_labels, average=None)
        auc = roc_auc_score(labels, pred_probs, average=None, multi_class='ovr')
        accuracy = np.mean(pred_labels[labels == i] == i)
        f1s.append(np.round(f1[i], 2))
        aucs.append(np.round(auc[i], 2))
        accuracies.append(np.round(accuracy, 2))

    df_m = pd.DataFrame({'F1-Score': f1s, 'AUC': aucs, 'Accuracy': accuracies}, index=self.solar_dataset.classes_list)
    ax = df_m.plot.barh(figsize=(10,10))
    for container in ax.containers:
        ax.bar_label(container)
    plt.show()
