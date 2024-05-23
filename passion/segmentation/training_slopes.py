# import pathlib
# import cv2
# import pickle
# import numpy as np
# from torch.utils.data import Subset
# import random
# import torch
# import torchvision
# import torchmetrics
# import os
# import time
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# import torch.optim
# import torch.nn.functional as F
# import torch.nn as nn
# import matplotlib.pyplot as plt

# from passion.segmentation import models
# from passion.segmentation import models_dropouts
# from passion.segmentation import models_batch
# from passion.segmentation import models_resnet152
# from passion.segmentation import models_resnet50
# from passion.segmentation import model_ankit
# from passion.segmentation import model_slopes

# cv2.setNumThreads(4)
# # determine the device to be used for training and evaluation
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # determine if we will be pinning memory during data loading
# PIN_MEMORY = True if DEVICE == "cuda" else False

# class SegmentationDataset(torch.utils.data.Dataset):
#   def __init__(self, image_paths, label_paths, slope_paths, transforms):
#     self.image_paths = image_paths
#     self.label_paths = label_paths
#     self.slope_paths = slope_paths
#     self.transforms = transforms
    
#   def __len__(self):
# 		# return the number of total samples contained in the dataset
#     return len(self.image_paths)
#   def __getitem__(self, idx):
# 		# grab the image path from the current index
#     imagePath = self.image_paths[idx]
# 		# load the image from disk, swap its channels from BGR to RGB,
# 		# and read the associated mask from disk in grayscale mode
#     image = cv2.imread(str(imagePath))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     mask = cv2.imread(self.label_paths[idx], 0)
#     mask2 = cv2.imread(self.slope_paths[idx], 0)/99.0
# 		# check to see if we are applying any transformations
#     if self.transforms is not None:
# 			# apply the transformations to both image and its mask
#       image = self.transforms(image)
#       mask = self.transforms(mask)
#       # mask2 = self.transforms(mask2).float() #original
#       mask2 = (torch.from_numpy(mask2)).float()

#     return (image, mask, mask2)
  
# class AutomaticWeightedLoss(nn.Module):
#     """automatically weighted multi-task loss

#     Params：
#         num: int，the number of loss
#         x: multi-task loss
#     Examples：
#         loss1=1
#         loss2=2
#         awl = AutomaticWeightedLoss(2)
#         loss_sum = awl(loss1, loss2)
#     """
#     def __init__(self, num=2):
#         super(AutomaticWeightedLoss, self).__init__()
#         params = torch.ones(num, requires_grad=True)
#         self.params = torch.nn.Parameter(params)

#     def forward(self, *x):
#         loss_sum = 0
#         for i, loss in enumerate(x):
#             loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
#         return loss_sum

# if __name__ == '__main__':
#     awl = AutomaticWeightedLoss(2)
#     print(awl.parameters())

# def intersection_over_union(confusion_matrix):
#     # Intersection = TP Union = TP + FP + FN
#     intersection = np.diag(confusion_matrix)
#     union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    
#     # Aggiungi un piccolo valore epsilon per evitare divisione per zero
#     epsilon = 1e-9
#     iou = intersection / (union + epsilon)
    
#     # Sostituisci eventuali NaN con 0 (o un altro valore a tua scelta)
#     iou = np.nan_to_num(iou, nan=0.0)
    
#     return iou

# # def intersection_over_union(confusion_matrix):
# # 	# Intersection = TP Union = TP + FP + FN
# # 	# IoU = TP / (TP + FP + FN)
# # 	intersection = np.diag(confusion_matrix)
# # 	union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
# # 	iou = intersection / union
# # 	return iou

# def depth_error(x_pred, x_output):
#     epsilon = 1e-9
#     x_pred = x_pred * 99.0
#     x_output = x_output * 99.0
  
#     # relative_error = torch.abs(x_pred- x_output) / (x_output + epsilon)
#     # relative_percent_error = relative_error * 100  # Converti in percentuale
#     # return relative_percent_error.mean().item()

#     # Calcola l'errore assoluto tra la predizione e il valore reale
#     abs_err = torch.abs(x_pred - x_output)
#     # Calcola e ritorna la media degli errori assoluti e relativi
#     total_elements = x_output.numel()  # Numero totale di elementi nel tensore
#     return (torch.sum(abs_err) / total_elements).item()

# def train_model(train_data_path: pathlib.Path,
#                 val_data_path: pathlib.Path,
#                 model_output_path: pathlib.Path,
#                 model_name: str,
#                 num_classes: int = 18,
#                 batch_size: int = 1,
#                 learning_rate: float = 0.00001,
#                 n_epochs: int = 10):
#   '''
#   Takes a training and a validation data paths, their parameters,
#   and trains a segmentation model based on it.

#   Data paths must follow the following structure:
#   data_path
#   |-image
#     |-<filename>.png
#   |-label
#     |-<filename>.png
  
#   ---
  
#   train_data_path     -- Path, path of the training data.
#   val_data_path       -- Path, path of the validation data.
#   model_output_path   -- Path, folder in which the model will be stored
#   model_name          -- str, name for the model file
#   num_classes         -- int, number of classes in the segmentation task
#   batch_size          -- int, number of samples to take at a time for training
#   learning_rate       -- float, rate of the optimizer
#   n_epochs            -- int, number of full dataset iterations
#   '''
#   print(f"Training Configuration:")
#   print(f"Model Name: {model_name}")
#   print(f"Number of Classes: {num_classes}")
#   print(f"Batch Size: {batch_size}")
#   print(f"Learning Rate: {learning_rate}")
#   print(f"Number of Epochs: {n_epochs}")
#   print("-" * 50)

#   model_output_path.mkdir(parents=True, exist_ok=True)
  
#   train_image_paths = sorted(list((train_data_path / 'image').glob('*.png')))
#   train_image_paths = [str(path) for path in train_image_paths]
#   train_label_paths = sorted(list((train_data_path / 'label').glob('*.png')))
#   train_label_paths = [str(path) for path in train_label_paths]
#   train_slope_paths = sorted(list((train_data_path / 'slope').glob('*.png')))
#   train_slope_paths = [str(path) for path in train_slope_paths]
#   val_image_paths = sorted(list((val_data_path / 'image').glob('*.png')))
#   val_image_paths = [str(path) for path in val_image_paths]
#   val_label_paths = sorted(list((val_data_path / 'label').glob('*.png')))
#   val_label_paths = [str(path) for path in val_label_paths]
#   val_slope_paths = sorted(list((val_data_path / 'slope').glob('*.png')))
#   val_slope_paths = [str(path) for path in val_slope_paths]
  
#   trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
#   train_dataset = SegmentationDataset(image_paths=train_image_paths,
#                                      label_paths=train_label_paths,
#                                      slope_paths=train_slope_paths,
#                                      transforms=trans)
#   val_dataset = SegmentationDataset(image_paths=val_image_paths,
#                                     label_paths=val_label_paths,
#                                     slope_paths=val_slope_paths,
#                                     transforms=trans)

#   print(f"[INFO] found {len(train_dataset)}, examples in the training set...")
#   print(f"[INFO] {len(train_image_paths)},{len(train_label_paths)}")
#   print(f"[INFO] found {len(val_dataset)}, examples in the validation set...")
#   print(f"[INFO] {len(val_image_paths)},{len(val_label_paths)}")

#   train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
#     batch_size=batch_size, pin_memory=PIN_MEMORY,
#     num_workers=os.cpu_count())
#   val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
#     batch_size=batch_size, pin_memory=PIN_MEMORY,
#     num_workers=os.cpu_count())

#   focal_loss = torch.hub.load(
#     '/scratch/clear/aboccala/PASSION/passion/segmentation/pytorch-multi-class-focal-loss',
#     model='focal_loss',
#     source='local',
#     alpha=None,
#     gamma=2,
#     reduction='mean',
#     device=DEVICE,
#     dtype=torch.float32,
#     force_reload=False
#   )
#   loss_func = focal_loss

#   print(f'Initializing ResNetUNet with {num_classes} classes...')
#   #unet = models.ResNetUNet(num_classes).to(DEVICE)
#   unet = model_slopes.ResNetUNet(num_classes).to(DEVICE)
#   unet.to(DEVICE)
#   optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate, weight_decay=1e-5)
#   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
#   awl = AutomaticWeightedLoss(2).to(DEVICE)

#   train_steps = len(train_dataset) // batch_size
#   val_steps = len(val_dataset) // batch_size
#   # initialize a dictionary to store training history
#   H = {"train_loss": [], "val_loss": [], "train_acc":[], "val_acc":[]}

#   best_val_score = 0.0

#   # Tensorboard writer
#   tensorboard_path = pathlib.Path(model_output_path / (model_name.split('.')[0]))
#   model_output_path.mkdir(parents=True, exist_ok=True)
#   writer = SummaryWriter(log_dir=str(tensorboard_path))

#   # loop over epochs
#   print("[INFO] training the network...")
#   start_time = time.time()
#   for e in tqdm(range(n_epochs)):
#     confmat = torchmetrics.ConfusionMatrix(task='multiclass',num_classes=num_classes).to(DEVICE)
#     cm = torch.zeros((num_classes,num_classes)).to(DEVICE)
#     # set the model in training mode
#     unet.train()
#     # initialize the total training and validation loss
#     total_train_loss = 0
#     total_train_abs_err = 0

#     total_val_loss = 0
#     total_correct = 0
#     total = 0
#     jaccard = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(DEVICE)
#     total_val, total_val_correct = 0, 0
#     # loop over the training set
#     for (i, (x, y, z)) in enumerate(train_loader):
#       y *= 255
#       # send the input to the device
#       (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
#       # perform a forward pass and calculate the training loss
#       pred, pred2 = unet(x)

#       # if i % 10 ==0:  # Adjust the frequency according to your preference
#       #     plt.figure()
#       #     plt.hist(pred2.detach().cpu().numpy().flatten(), bins=30, alpha=0.75)
#       #     plt.title(f'Histogram of Slope Predictions at Epoch {e+1}, Batch {i+1}')
#       #     plt.xlabel('Slope Value')
#       #     plt.ylabel('Frequency')
#       #     plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/slope_pred2.png', bbox_inches='tight')
#       #     plt.close()

#       # if i % 10 ==0:  # Adjust the frequency according to your preference
#       #     plt.figure()
#       #     plt.hist(z.detach().cpu().numpy().flatten(), bins=30, alpha=0.75)
#       #     plt.title(f'Histogram of Slope Predictions at Epoch {e+1}, Batch {i+1}')
#       #     plt.xlabel('Slope Value')
#       #     plt.ylabel('Frequency')
#       #     plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/slope_z.png', bbox_inches='tight')
#       #     plt.close()

#       # if i % 10 ==0:  # Adjust the frequency according to your preference
#       #     plt.figure()
#       #     plt.hist(pred.detach().cpu().numpy().flatten(), bins=30, alpha=0.75)
#       #     plt.title(f'Histogram of Slope Predictions at Epoch {e+1}, Batch {i+1}')
#       #     plt.xlabel('Slope Value')
#       #     plt.ylabel('Frequency')
#       #     plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/slope_pred.png', bbox_inches='tight')
#       #     plt.close()

#       # if i % 10 ==0:  # Adjust the frequency according to your preference
#       #     plt.figure()
#       #     plt.hist(y.detach().cpu().numpy().flatten(), bins=30, alpha=0.75)
#       #     plt.title(f'Histogram of Slope Predictions at Epoch {e+1}, Batch {i+1}')
#       #     plt.xlabel('Slope Value')
#       #     plt.ylabel('Frequency')
#       #     plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/slope_y.png', bbox_inches='tight')
#       #     plt.close()

#       if DEVICE == "cuda":
#         y = y.type(torch.cuda.LongTensor).squeeze()
#       else:
#         y = y.type(torch.LongTensor).squeeze()
        
#       loss_seg = loss_func(pred, (y))
#       loss_slope = F.mse_loss(pred2 , (z))
#       # print(f"Epoch {e+1}, Batch {i+1}: Seg Loss: {loss_seg.item():.4f}")
#       # print(f"Epoch {e+1}, Batch {i+1}: Slope MSE Loss: {loss_slope.item():.4f}")
#       loss = awl(loss_seg, loss_slope)
#       # total_slope_error += loss_slope.item()
     
#       # loss=0.1*loss_seg + loss_slope 
#       optimizer.zero_grad()
#       loss.backward()
#       optimizer.step()
#       # add the loss to the total training loss so far
#       total_train_loss += loss.detach()
#       train_abs_err = depth_error(pred2, z)
#       total_train_abs_err += train_abs_err


#       pred = torch.argmax(pred, dim=1)
#       total += y.size().numel()
#       total_correct += (pred == y).sum().item()
#       if DEVICE == "cuda":
#         pred = pred.type(torch.cuda.LongTensor).squeeze()
#       else:
#         pred = pred.type(torch.LongTensor).squeeze()
#       cm += confmat(pred, y)

#     train_iou = intersection_over_union(cm.cpu().detach().numpy())
#     filtered_train_iou = train_iou[train_iou != 0]  # Filtra per rimuovere gli zeri
#     mean_train_iou = np.mean(filtered_train_iou)
#     mean_abs_slope_error = total_train_abs_err / len(train_loader)
#     print('Train Class IOU', filtered_train_iou)
#     print('Train Mean Class IOU:{}'.format(mean_train_iou))
#     print('Train Mean Abs Slope Error: {:.4f}'.format(mean_abs_slope_error)) 
#     # switch off autograd
#     with torch.no_grad():
#       confmat = torchmetrics.ConfusionMatrix(task='multiclass',num_classes=num_classes).to(DEVICE)
#       cm = torch.zeros((num_classes,num_classes)).to(DEVICE)
#       # set the model in evaluation mode
#       unet.eval()
#       # loop over the validation set
#       total_val_loss = 0
#       total_val_abs_err = 0
#       total_val_correct = 0
#       total_val = 0
#       for (x, y, z) in val_loader:
#         # y = adjust_labels(y, exclude_class=1, new_class=0)
#          # send the input to the device
#         y *= 255

#         (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
#         # make the predictions and calculate the validation loss
#         pred, pred2 = unet(x)

#         if DEVICE == "cuda":
#           y = y.type(torch.cuda.LongTensor).squeeze()
#         else:
#           y = y.type(torch.LongTensor).squeeze()
#           #y = replaceTensor(y)  
#         loss_seg = loss_func(pred, y)
#         loss_slope = F.mse_loss(pred2, z)
#         loss = awl(loss_seg, loss_slope)  # Utilizza AutomaticWeightedLoss
#         total_val_loss += loss.item()

#         val_abs_err= depth_error(pred2, z)
#         total_val_abs_err += val_abs_err



#         pred = torch.argmax(pred, dim=1)
#         total_val += y.size().numel()
#         total_val_correct += (pred == y).sum().item()
        
#         if DEVICE == "cuda":
#           pred = pred.type(torch.cuda.LongTensor).squeeze()
#         else:
#           pred = pred.type(torch.LongTensor).squeeze()

#         j_s = jaccard((pred), (y))
#         cm += confmat(pred, y) 

#       val_iou = intersection_over_union(cm.cpu().detach().numpy())
#       filtered_val_iou = val_iou[val_iou != 0]  # Filtra per rimuovere gli zeri
#       mean_val_iou = np.mean(filtered_val_iou)
#       mean_abs_slope_error = total_val_abs_err / len(val_loader)
#       print('Val Class IOU', filtered_val_iou)
#       print('Val Mean Class IOU:{}'.format(mean_val_iou))
#       print('Val Mean Abs Slope Error: {:.4f}'.format(mean_abs_slope_error)) 


#       if(mean_val_iou > best_val_score):
#         print('Better Model Found: {} > {}'.format(mean_val_iou, best_val_score))
#         best_val_score = mean_val_iou
#         torch.save(unet, model_output_path / model_name)

#     # calculate the average training and validation loss
#     avg_train_loss = total_train_loss / train_steps
#     avg_val_loss = total_val_loss / val_steps
#     avg_train_acc = total_correct / total
#     avg_val_acc = total_val_correct / total_val

#     # update our training history
#     H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
#     H["val_loss"].append(avg_val_loss)
#     H["train_acc"].append(avg_train_acc)
#     H["val_acc"].append(avg_val_acc)
#     # print the model training and validation information
#     print("[INFO] EPOCH: {}/{}".format(e + 1, n_epochs))
#     print("Train loss: {:.6f}, Val loss: {:.4f} Train Acc: {:.6f}, Val Acc: {:.4f}".format(
#       avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc))
#     scheduler.step(avg_val_loss)
#   # display the total time needed to perform the training
#   end_time = time.time()
#   print("[INFO] total time taken to train the model: {:.2f}s".format(
#     end_time - start_time))

#   # serialize the model to disk
#   torch.save(unet, model_output_path / 'model_last.pth')
#   writer.flush()
#   print('Best Val Score:{}'.format(best_val_score))

# import pathlib
# import cv2
# import numpy as np
# import torch
# import torchvision
# import torchmetrics
# import os
# import time
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# import torch.optim
# import torch.nn.functional as F
# import torch.nn as nn

# from passion.segmentation import model_slopes

# cv2.setNumThreads(4)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PIN_MEMORY = True if DEVICE == "cuda" else False

# class SegmentationDataset(torch.utils.data.Dataset):
#     def __init__(self, image_paths, label_paths, slope_paths, transforms, max_slope):
#         self.image_paths = image_paths
#         self.label_paths = label_paths
#         self.slope_paths = slope_paths
#         self.transforms = transforms
#         self.max_slope = max_slope
    
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         imagePath = self.image_paths[idx]
#         image = cv2.imread(str(imagePath))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(self.label_paths[idx], 0)
#         slope = cv2.imread(self.slope_paths[idx], 0)
        
#         if self.transforms is not None:
#             image = self.transforms(image)
#             mask = self.transforms(mask)
#             slope = torch.from_numpy(slope).float() / self.max_slope  # Normalize slope values to [0, 1]

#         return (image, mask, slope)

# def find_slope_max(slope_paths):
#     max_value = float('-inf')
#     for slope_path in tqdm(slope_paths):
#         slope = cv2.imread(str(slope_path), cv2.IMREAD_GRAYSCALE)
#         max_value = max(max_value, slope.max())
#     return max_value

# train_slope_paths = sorted(list((pathlib.Path('/scratch/clear/aboccala/training/RID/output_mix_ge_va/train_ge/slope')).glob('*.png')))
# val_slope_paths = sorted(list((pathlib.Path('/scratch/clear/aboccala/training/RID/output_mix_ge_va/test_va/slope')).glob('*.png')))

# train_max = find_slope_max(train_slope_paths)
# val_max = find_slope_max(val_slope_paths)
# overall_max = max(train_max, val_max)

# print(f"Overall Max Slope: {overall_max}")

# class AutomaticWeightedLoss(nn.Module):
#     def __init__(self, num=2):
#         super(AutomaticWeightedLoss, self).__init__()
#         params = torch.ones(num, requires_grad=True)
#         self.params = torch.nn.Parameter(params)

#     def forward(self, *x):
#         loss_sum = 0
#         for i, loss in enumerate(x):
#             loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
#         return loss_sum

# if __name__ == '__main__':
#     awl = AutomaticWeightedLoss(2)
#     print(awl.parameters())

# def intersection_over_union(confusion_matrix):
#     intersection = np.diag(confusion_matrix)
#     union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    
#     epsilon = 1e-9
#     iou = intersection / (union + epsilon)
#     iou = np.nan_to_num(iou, nan=0.0)
#     return iou

# def depth_error(x_pred, x_output, max_slope):
#     x_pred = x_pred * max_slope
#     x_output = x_output * max_slope

#     abs_err = torch.abs(x_pred - x_output)
#     total_elements = x_output.numel()
#     return (torch.sum(abs_err) / total_elements).item()

# def train_model(train_data_path: pathlib.Path,
#                 val_data_path: pathlib.Path,
#                 model_output_path: pathlib.Path,
#                 model_name: str,
#                 num_classes: int = 18,
#                 batch_size: int = 16,
#                 learning_rate: float = 0.00001,
#                 n_epochs: int = 10):
#     print(f"Training Configuration:")
#     print(f"Model Name: {model_name}")
#     print(f"Number of Classes: {num_classes}")
#     print(f"Batch Size: {batch_size}")
#     print(f"Learning Rate: {learning_rate}")
#     print(f"Number of Epochs: {n_epochs}")
#     print("-" * 50)

#     model_output_path.mkdir(parents=True, exist_ok=True)
    
#     train_image_paths = sorted(list((train_data_path / 'image').glob('*.png')))
#     train_image_paths = [str(path) for path in train_image_paths]
#     train_label_paths = sorted(list((train_data_path / 'label').glob('*.png')))
#     train_label_paths = [str(path) for path in train_label_paths]
#     train_slope_paths = sorted(list((train_data_path / 'slope').glob('*.png')))
#     train_slope_paths = [str(path) for path in train_slope_paths]
#     val_image_paths = sorted(list((val_data_path / 'image').glob('*.png')))
#     val_image_paths = [str(path) for path in val_image_paths]
#     val_label_paths = sorted(list((val_data_path / 'label').glob('*.png')))
#     val_label_paths = [str(path) for path in val_label_paths]
#     val_slope_paths = sorted(list((val_data_path / 'slope').glob('*.png')))
#     val_slope_paths = [str(path) for path in val_slope_paths]
    
#     trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    
#     train_dataset = SegmentationDataset(image_paths=train_image_paths,
#                                         label_paths=train_label_paths,
#                                         slope_paths=train_slope_paths,
#                                         transforms=trans,
#                                         max_slope=overall_max)
    
#     val_dataset = SegmentationDataset(image_paths=val_image_paths,
#                                       label_paths=val_label_paths,
#                                       slope_paths=val_slope_paths,
#                                       transforms=trans,
#                                       max_slope=overall_max)

#     print(f"[INFO] found {len(train_dataset)}, examples in the training set...")
#     print(f"[INFO] {len(train_image_paths)},{len(train_label_paths)}")
#     print(f"[INFO] found {len(val_dataset)}, examples in the validation set...")
#     print(f"[INFO] {len(val_image_paths)},{len(val_label_paths)}")

#     train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
#         batch_size=batch_size, pin_memory=PIN_MEMORY,
#         num_workers=os.cpu_count())
#     val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
#         batch_size=batch_size, pin_memory=PIN_MEMORY,
#         num_workers=os.cpu_count())

#     focal_loss = torch.hub.load(
#         '/scratch/clear/aboccala/PASSION/passion/segmentation/pytorch-multi-class-focal-loss',
#         model='focal_loss',
#         source='local',
#         alpha=None,
#         gamma=2,
#         reduction='mean',
#         device=DEVICE,
#         dtype=torch.float32,
#         force_reload=False
#     )
#     loss_func = focal_loss

#     print(f'Initializing ResNetUNet with {num_classes} classes...')
#     unet = model_slopes.ResNetUNet(num_classes).to(DEVICE)
#     unet.to(DEVICE)
#     optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate, weight_decay=1e-5)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
#     awl = AutomaticWeightedLoss(2).to(DEVICE)

#     train_steps = len(train_dataset) // batch_size
#     val_steps = len(val_dataset) // batch_size
#     H = {"train_loss": [], "val_loss": [], "train_acc":[], "val_acc":[]}

#     best_val_score = 0.0

#     tensorboard_path = pathlib.Path(model_output_path / (model_name.split('.')[0]))
#     tensorboard_path.mkdir(parents=True, exist_ok=True)
#     writer = SummaryWriter(log_dir=str(tensorboard_path))

#     print("[INFO] training the network...")
#     start_time = time.time()
#     for e in tqdm(range(n_epochs)):
#         confmat = torchmetrics.ConfusionMatrix(task='multiclass',num_classes=num_classes).to(DEVICE)
#         cm = torch.zeros((num_classes,num_classes)).to(DEVICE)
#         unet.train()
#         total_train_loss = 0
#         total_train_abs_err = 0
#         total_val_loss = 0
#         total_correct = 0
#         total = 0
#         total_seg_loss = 0
#         total_slope_loss = 0
#         jaccard = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(DEVICE)
#         total_val, total_val_correct = 0, 0

#         for (i, (x, y, z)) in enumerate(train_loader):
#             y *= 255
#             (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
#             pred, pred2 = unet(x)

#             y = y.long().squeeze()
#             loss_seg = loss_func(pred, y)
#             loss_slope = F.mse_loss(pred2 * overall_max, z * overall_max)
#             # loss = awl(loss_seg, loss_slope)
#             loss = awl(loss_seg, loss_slope*0.1)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.detach()
#             total_seg_loss += loss_seg.detach()
#             total_slope_loss += loss_slope.detach()
#             train_abs_err = depth_error(pred2, z, overall_max)
#             total_train_abs_err += train_abs_err

#             pred = torch.argmax(pred, dim=1)
#             total += y.size().numel()
#             total_correct += (pred == y).sum().item()
#             cm += confmat(pred, y)

#         avg_seg_loss = total_seg_loss / train_steps
#         avg_slope_loss = total_slope_loss / train_steps
#         train_iou = intersection_over_union(cm.cpu().detach().numpy())
#         filtered_train_iou = train_iou[train_iou != 0]
#         mean_train_iou = np.mean(filtered_train_iou)
#         mean_abs_slope_error = total_train_abs_err / len(train_loader)
#         print('Train Class IOU', filtered_train_iou)
#         print('Train Mean Class IOU:{}'.format(mean_train_iou))
#         print('Train Mean Abs Slope Error: {:.4f}'.format(mean_abs_slope_error))
#         print('Train Avg Segmentation Loss: {:.4f}'.format(avg_seg_loss))
#         print('Train Avg Slope Loss: {:.4f}'.format(avg_slope_loss))
#         print('AWL Params:', awl.params.detach().cpu().numpy())

#         with torch.no_grad():
#             confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes).to(DEVICE)
#             cm = torch.zeros((num_classes, num_classes)).to(DEVICE)
#             unet.eval()
#             total_val_loss = 0
#             total_val_abs_err = 0
#             total_val_correct = 0
#             total_val = 0
#             total_val_seg_loss = 0
#             total_val_slope_loss = 0
#             for (x, y, z) in val_loader:
#                 y *= 255
#                 (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
#                 pred, pred2 = unet(x)

#                 y = y.long().squeeze()
#                 loss_seg = loss_func(pred, y)
#                 loss_slope = F.mse_loss(pred2 * overall_max, z * overall_max)
#                 # loss = awl(loss_seg, loss_slope)
#                 loss = awl(loss_seg, loss_slope*0.1)
#                 total_val_loss += loss.item()
#                 total_val_seg_loss += loss_seg.item()
#                 total_val_slope_loss += loss_slope.item()

#                 val_abs_err = depth_error(pred2, z, overall_max)
#                 total_val_abs_err += val_abs_err

#                 pred = torch.argmax(pred, dim=1)
#                 total_val += y.size().numel()
#                 total_val_correct += (pred == y).sum().item()

#                 cm += confmat(pred, y)

#             avg_val_seg_loss = total_val_seg_loss / val_steps
#             avg_val_slope_loss = total_val_slope_loss / val_steps
#             val_iou = intersection_over_union(cm.cpu().detach().numpy())
#             filtered_val_iou = val_iou[val_iou != 0]
#             mean_val_iou = np.mean(filtered_val_iou)
#             mean_abs_slope_error = total_val_abs_err / len(val_loader)
#             print('Val Class IOU', filtered_val_iou)
#             print('Val Mean Class IOU:{}'.format(mean_val_iou))
#             print('Val Mean Abs Slope Error: {:.4f}'.format(mean_abs_slope_error))
#             print('Val Avg Segmentation Loss: {:.4f}'.format(avg_val_seg_loss))
#             print('Val Avg Slope Loss: {:.4f}'.format(avg_val_slope_loss))
#             print('AWL Params:', awl.params.detach().cpu().numpy())

#             if mean_val_iou > best_val_score:
#                 print('Better Model Found: {} > {}'.format(mean_val_iou, best_val_score))
#                 best_val_score = mean_val_iou
#                 torch.save(unet, model_output_path / model_name)

#         avg_train_loss = total_train_loss / train_steps
#         avg_val_loss = total_val_loss / val_steps
#         avg_train_acc = total_correct / total
#         avg_val_acc = total_val_correct / total_val

#         H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
#         H["val_loss"].append(avg_val_loss)
#         H["train_acc"].append(avg_train_acc)
#         H["val_acc"].append(avg_val_acc)
#         print("[INFO] EPOCH: {}/{}".format(e + 1, n_epochs))
#         print("Train loss: {:.6f}, Val loss: {:.4f} Train Acc: {:.6f}, Val Acc: {:.4f}".format(
#             avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc))
#         scheduler.step(avg_val_loss)

#     end_time = time.time()
#     print("[INFO] total time taken to train the model: {:.2f}s".format(
#         end_time - start_time))

#     torch.save(unet, model_output_path / 'model_last.pth')
#     writer.flush()
#     print('Best Val Score:{}'.format(best_val_score))



# import pathlib
# import cv2
# import numpy as np
# import torch
# import torchvision
# import torchmetrics
# import os
# import time
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# import torch.optim
# import torch.nn.functional as F
# import torch.nn as nn

# from passion.segmentation import model_slopes

# cv2.setNumThreads(4)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PIN_MEMORY = True if DEVICE == "cuda" else False

# class SegmentationDataset(torch.utils.data.Dataset):
#     def __init__(self, image_paths, label_paths, slope_paths, transforms, max_slope):
#         self.image_paths = image_paths
#         self.label_paths = label_paths
#         self.slope_paths = slope_paths
#         self.transforms = transforms
#         self.max_slope = max_slope
    
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         imagePath = self.image_paths[idx]
#         image = cv2.imread(str(imagePath))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(self.label_paths[idx], 0)
#         slope = cv2.imread(self.slope_paths[idx], 0)
        
#         if self.transforms is not None:
#             image = self.transforms(image)
#             mask = self.transforms(mask)
#             slope = torch.from_numpy(slope).float() / self.max_slope  # Normalize slope values to [0, 1]

#         return (image, mask, slope)

# def find_slope_max(slope_paths):
#     max_value = float('-inf')
#     for slope_path in tqdm(slope_paths):
#         slope = cv2.imread(str(slope_path), cv2.IMREAD_GRAYSCALE)
#         max_value = max(max_value, slope.max())
#     return max_value

# train_slope_paths = sorted(list((pathlib.Path('/scratch/clear/aboccala/training/RID/output_mix_ge_va/train_ge/slope')).glob('*.png')))
# val_slope_paths = sorted(list((pathlib.Path('/scratch/clear/aboccala/training/RID/output_mix_ge_va/test_va/slope')).glob('*.png')))

# train_max = find_slope_max(train_slope_paths)
# val_max = find_slope_max(val_slope_paths)
# overall_max = max(train_max, val_max)

# print(f"Overall Max Slope: {overall_max}")

# class AutomaticWeightedLoss(nn.Module):
#     def __init__(self, num=2):
#         super(AutomaticWeightedLoss, self).__init__()
#         params = torch.ones(num, requires_grad=True)
#         self.params = torch.nn.Parameter(params)

#     def forward(self, *x):
#         loss_sum = 0
#         for i, loss in enumerate(x):
#             loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
#         return loss_sum

# if __name__ == '__main__':
#     awl = AutomaticWeightedLoss(2)
#     print(awl.parameters())

# def intersection_over_union(confusion_matrix):
#     intersection = np.diag(confusion_matrix)
#     union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    
#     epsilon = 1e-9
#     iou = intersection / (union + epsilon)
#     iou = np.nan_to_num(iou, nan=0.0)
#     return iou

# def depth_error(x_pred, x_output, max_slope):
#     x_pred = x_pred * max_slope
#     x_output = x_output * max_slope

#     abs_err = torch.abs(x_pred - x_output)
#     total_elements = x_output.numel()
#     return (torch.sum(abs_err) / total_elements).item()

# def smooth_l1_loss_with_center_penalty(pred, target, max_slope, beta=1.0):
#     # Normalize predictions and targets
#     pred = pred * max_slope
#     target = target * max_slope

#     # Smooth L1 Loss (Huber Loss)
#     diff = torch.abs(pred - target)
#     smooth_l1_loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    
#     # Calculate the distance penalty based on distance from the center
#     batch_size, channels, height, width = pred.shape
#     y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
#     y_coords = y_coords.float().to(pred.device) / height
#     x_coords = x_coords.float().to(pred.device) / width
#     distance_penalty = torch.sqrt((x_coords - 0.5)**2 + (y_coords - 0.5)**2)
#     distance_penalty = distance_penalty.unsqueeze(0).unsqueeze(0)  # Make it (1, 1, height, width)

#     # Apply the distance penalty to the loss
#     penalized_loss = smooth_l1_loss * distance_penalty

#     return penalized_loss.mean()

# def train_model(train_data_path: pathlib.Path,
#                 val_data_path: pathlib.Path,
#                 model_output_path: pathlib.Path,
#                 model_name: str,
#                 num_classes: int = 18,
#                 batch_size: int = 16,
#                 learning_rate_seg: float = 0.00001,
#                 learning_rate_slope: float = 0.00001,
#                 n_epochs: int = 10):
#     print(f"Training Configuration:")
#     print(f"Model Name: {model_name}")
#     print(f"Number of Classes: {num_classes}")
#     print(f"Batch Size: {batch_size}")
#     print(f"Learning Rate (Segmentation): {learning_rate_seg}")
#     print(f"Learning Rate (Slope): {learning_rate_slope}")
#     print(f"Number of Epochs: {n_epochs}")
#     print("-" * 50)

#     model_output_path.mkdir(parents=True, exist_ok=True)
    
#     train_image_paths = sorted(list((train_data_path / 'image').glob('*.png')))
#     train_image_paths = [str(path) for path in train_image_paths]
#     train_label_paths = sorted(list((train_data_path / 'label').glob('*.png')))
#     train_label_paths = [str(path) for path in train_label_paths]
#     train_slope_paths = sorted(list((train_data_path / 'slope').glob('*.png')))
#     train_slope_paths = [str(path) for path in train_slope_paths]
#     val_image_paths = sorted(list((val_data_path / 'image').glob('*.png')))
#     val_image_paths = [str(path) for path in val_image_paths]
#     val_label_paths = sorted(list((val_data_path / 'label').glob('*.png')))
#     val_label_paths = [str(path) for path in val_label_paths]
#     val_slope_paths = sorted(list((val_data_path / 'slope').glob('*.png')))
#     val_slope_paths = [str(path) for path in val_slope_paths]
    
#     trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    
#     train_dataset = SegmentationDataset(image_paths=train_image_paths,
#                                         label_paths=train_label_paths,
#                                         slope_paths=train_slope_paths,
#                                         transforms=trans,
#                                         max_slope=overall_max)
    
#     val_dataset = SegmentationDataset(image_paths=val_image_paths,
#                                       label_paths=val_label_paths,
#                                       slope_paths=val_slope_paths,
#                                       transforms=trans,
#                                       max_slope=overall_max)

#     print(f"[INFO] found {len(train_dataset)}, examples in the training set...")
#     print(f"[INFO] {len(train_image_paths)},{len(train_label_paths)}")
#     print(f"[INFO] found {len(val_dataset)}, examples in the validation set...")
#     print(f"[INFO] {len(val_image_paths)},{len(val_label_paths)}")

#     train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
#         batch_size=batch_size, pin_memory=PIN_MEMORY,
#         num_workers=os.cpu_count())
#     val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
#         batch_size=batch_size, pin_memory=PIN_MEMORY,
#         num_workers=os.cpu_count())

#     focal_loss = torch.hub.load(
#         '/scratch/clear/aboccala/PASSION/passion/segmentation/pytorch-multi-class-focal-loss',
#         model='focal_loss',
#         source='local',
#         alpha=None,
#         gamma=2,
#         reduction='mean',
#         device=DEVICE,
#         dtype=torch.float32,
#         force_reload=False
#     )
#     loss_func = focal_loss

#     print(f'Initializing ResNetUNet with {num_classes} classes...')
#     unet = model_slopes.ResNetUNet(num_classes).to(DEVICE)
#     unet.to(DEVICE)

#     # Separate optimizers for segmentation and slope
#     optimizer_seg = torch.optim.Adam(unet.parameters(), lr=learning_rate_seg, weight_decay=1e-5)
#     optimizer_slope = torch.optim.Adam(unet.parameters(), lr=learning_rate_slope, weight_decay=1e-5)

#     scheduler_seg = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_seg, 'min', factor=0.1, patience=5, verbose=True)
#     scheduler_slope = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_slope, 'min', factor=0.1, patience=5, verbose=True)

#     awl = AutomaticWeightedLoss(2).to(DEVICE)

#     train_steps = len(train_dataset) // batch_size
#     val_steps = len(val_dataset) // batch_size
#     H = {"train_loss": [], "val_loss": [], "train_acc":[], "val_acc":[]}

#     best_val_score = 0.0

#     tensorboard_path = pathlib.Path(model_output_path / (model_name.split('.')[0]))
#     tensorboard_path.mkdir(parents=True, exist_ok=True)
#     writer = SummaryWriter(log_dir=str(tensorboard_path))

#     print("[INFO] training the network...")
#     start_time = time.time()
#     for e in tqdm(range(n_epochs)):
#         confmat = torchmetrics.ConfusionMatrix(task='multiclass',num_classes=num_classes).to(DEVICE)
#         cm = torch.zeros((num_classes,num_classes)).to(DEVICE)
#         unet.train()
#         total_train_loss = 0
#         total_train_abs_err = 0
#         total_val_loss = 0
#         total_correct = 0
#         total = 0
#         total_seg_loss = 0
#         total_slope_loss = 0
#         jaccard = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(DEVICE)
#         total_val, total_val_correct = 0, 0

#         for (i, (x, y, z)) in enumerate(train_loader):
#             y *= 255
#             (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
#             pred, pred2 = unet(x)

#             y = y.long().squeeze()
#             loss_seg = loss_func(pred, y)
#             # loss_slope = F.mse_loss(pred2 * overall_max, z * overall_max)
#             loss_slope = smooth_l1_loss_with_center_penalty(pred2, z, overall_max)

#             loss = awl(loss_seg, loss_slope)

#             optimizer_seg.zero_grad()
#             optimizer_slope.zero_grad()
#             loss.backward()
#             optimizer_seg.step()
#             optimizer_slope.step()

#             total_train_loss += loss.detach()
#             total_seg_loss += loss_seg.detach()
#             total_slope_loss += loss_slope.detach()
#             train_abs_err = depth_error(pred2, z, overall_max)
#             total_train_abs_err += train_abs_err

#             pred = torch.argmax(pred, dim=1)
#             total += y.size().numel()
#             total_correct += (pred == y).sum().item()
#             cm += confmat(pred, y)

#         avg_seg_loss = total_seg_loss / train_steps
#         avg_slope_loss = total_slope_loss / train_steps
#         train_iou = intersection_over_union(cm.cpu().detach().numpy())
#         filtered_train_iou = train_iou[train_iou != 0]
#         mean_train_iou = np.mean(filtered_train_iou)
#         mean_abs_slope_error = total_train_abs_err / len(train_loader)
#         print('Train Class IOU', filtered_train_iou)
#         print('Train Mean Class IOU:{}'.format(mean_train_iou))
#         print('Train Mean Abs Slope Error: {:.4f}'.format(mean_abs_slope_error))
#         print('Train Avg Segmentation Loss: {:.4f}'.format(avg_seg_loss))
#         print('Train Avg Slope Loss: {:.4f}'.format(avg_slope_loss))
#         print('AWL Params:', awl.params.detach().cpu().numpy())

#         with torch.no_grad():
#             confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes).to(DEVICE)
#             cm = torch.zeros((num_classes, num_classes)).to(DEVICE)
#             unet.eval()
#             total_val_loss = 0
#             total_val_abs_err = 0
#             total_val_correct = 0
#             total_val = 0
#             total_val_seg_loss = 0
#             total_val_slope_loss = 0
#             for (x, y, z) in val_loader:
#                 y *= 255
#                 (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
#                 pred, pred2 = unet(x)

#                 y = y.long().squeeze()
#                 loss_seg = loss_func(pred, y)
#                 # loss_slope = F.mse_loss(pred2 * overall_max, z * overall_max)
#                 loss_slope = smooth_l1_loss_with_center_penalty(pred2, z, overall_max)
#                 loss = awl(loss_seg, loss_slope)
#                 total_val_loss += loss.item()
#                 total_val_seg_loss += loss_seg.item()
#                 total_val_slope_loss += loss_slope.item()

#                 val_abs_err = depth_error(pred2, z, overall_max)
#                 total_val_abs_err += val_abs_err

#                 pred = torch.argmax(pred, dim=1)
#                 total_val += y.size().numel()
#                 total_val_correct += (pred == y).sum().item()

#                 cm += confmat(pred, y)

#             avg_val_seg_loss = total_val_seg_loss / val_steps
#             avg_val_slope_loss = total_val_slope_loss / val_steps
#             val_iou = intersection_over_union(cm.cpu().detach().numpy())
#             filtered_val_iou = val_iou[val_iou != 0]
#             mean_val_iou = np.mean(filtered_val_iou)
#             mean_abs_slope_error = total_val_abs_err / len(val_loader)
#             print('Val Class IOU', filtered_val_iou)
#             print('Val Mean Class IOU:{}'.format(mean_val_iou))
#             print('Val Mean Abs Slope Error: {:.4f}'.format(mean_abs_slope_error))
#             print('Val Avg Segmentation Loss: {:.4f}'.format(avg_val_seg_loss))
#             print('Val Avg Slope Loss: {:.4f}'.format(avg_val_slope_loss))
#             print('AWL Params:', awl.params.detach().cpu().numpy())

#             if mean_val_iou > best_val_score:
#                 print('Better Model Found: {} > {}'.format(mean_val_iou, best_val_score))
#                 best_val_score = mean_val_iou
#                 torch.save(unet, model_output_path / model_name)

#         avg_train_loss = total_train_loss / train_steps
#         avg_val_loss = total_val_loss / val_steps
#         avg_train_acc = total_correct / total
#         avg_val_acc = total_val_correct / total_val

#         H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
#         H["val_loss"].append(avg_val_loss)
#         H["train_acc"].append(avg_train_acc)
#         H["val_acc"].append(avg_val_acc)
#         print("[INFO] EPOCH: {}/{}".format(e + 1, n_epochs))
#         print("Train loss: {:.6f}, Val loss: {:.4f} Train Acc: {:.6f}, Val Acc: {:.4f}".format(
#             avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc))
#         scheduler_seg.step(avg_val_loss)
#         scheduler_slope.step(avg_val_loss)

#     end_time = time.time()
#     print("[INFO] total time taken to train the model: {:.2f}s".format(
#         end_time - start_time))

#     torch.save(unet, model_output_path / 'model_last.pth')
#     writer.flush()
#     print('Best Val Score:{}'.format(best_val_score))

#------------------------------------------------------------------------MODIFIED LOSS

# import pathlib
# import cv2
# import numpy as np
# import torch
# import torchvision
# import torchmetrics
# import os
# import time
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# import torch.optim
# import torch.nn.functional as F
# import torch.nn as nn

# from passion.segmentation import model_slopes

# cv2.setNumThreads(4)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PIN_MEMORY = True if DEVICE == "cuda" else False

# class SegmentationDataset(torch.utils.data.Dataset):
#     def __init__(self, image_paths, label_paths, slope_paths, transforms, max_slope):
#         self.image_paths = image_paths
#         self.label_paths = label_paths
#         self.slope_paths = slope_paths
#         self.transforms = transforms
#         self.max_slope = max_slope
    
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         imagePath = self.image_paths[idx]
#         image = cv2.imread(str(imagePath))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(self.label_paths[idx], 0)
#         slope = cv2.imread(self.slope_paths[idx], 0)
        
#         if self.transforms is not None:
#             image = self.transforms(image)
#             mask = self.transforms(mask)
#             slope = torch.from_numpy(slope).float() / self.max_slope  # Normalize slope values to [0, 1]

#         return (image, mask, slope)

# def find_slope_max(slope_paths):
#     max_value = float('-inf')
#     for slope_path in tqdm(slope_paths):
#         slope = cv2.imread(str(slope_path), cv2.IMREAD_GRAYSCALE)
#         max_value = max(max_value, slope.max())
#     return max_value

# train_slope_paths = sorted(list((pathlib.Path('/scratch/clear/aboccala/training/RID/output_mix_ge_va/train_ge/slope')).glob('*.png')))
# val_slope_paths = sorted(list((pathlib.Path('/scratch/clear/aboccala/training/RID/output_mix_ge_va/test_va/slope')).glob('*.png')))

# train_max = find_slope_max(train_slope_paths)
# val_max = find_slope_max(val_slope_paths)
# overall_max = max(train_max, val_max)

# print(f"Overall Max Slope: {overall_max}")

# class AutomaticWeightedLoss(nn.Module):
#     def __init__(self, num=2):
#         super(AutomaticWeightedLoss, self).__init__()
#         params = torch.ones(num, requires_grad=True)
#         self.params = torch.nn.Parameter(params)

#     def forward(self, *x):
#         loss_sum = 0
#         for i, loss in enumerate(x):
#             loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
#         return loss_sum

# if __name__ == '__main__':
#     awl = AutomaticWeightedLoss(2)
#     print(awl.parameters())

# def intersection_over_union(confusion_matrix):
#     intersection = np.diag(confusion_matrix)
#     union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    
#     epsilon = 1e-9
#     iou = intersection / (union + epsilon)
#     iou = np.nan_to_num(iou, nan=0.0)
#     return iou

# def depth_error(x_pred, x_output, max_slope):
#     x_pred = x_pred * max_slope
#     x_output = x_output * max_slope

#     abs_err = torch.abs(x_pred - x_output)
#     total_elements = x_output.numel()
#     return (torch.sum(abs_err) / total_elements).item()

# def smooth_l1_loss_with_center_penalty(pred, target, beta=1.0):
#     # Smooth L1 Loss (Huber Loss)
#     diff = torch.abs(pred - target)
#     smooth_l1_loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    
#     # Calculate the distance penalty based on distance from the center
#     batch_size, channels, height, width = pred.shape
#     y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
#     y_coords = y_coords.float().to(pred.device) / height
#     x_coords = x_coords.float().to(pred.device) / width
#     distance_penalty = torch.sqrt((x_coords - 0.5)**2 + (y_coords - 0.5)**2)
#     distance_penalty = distance_penalty.unsqueeze(0).unsqueeze(0)  # Make it (1, 1, height, width)

#     # Apply the distance penalty to the loss
#     penalized_loss = smooth_l1_loss * distance_penalty

#     return penalized_loss.mean()

# def train_model(train_data_path: pathlib.Path,
#                 val_data_path: pathlib.Path,
#                 model_output_path: pathlib.Path,
#                 model_name: str,
#                 num_classes: int = 18,
#                 batch_size: int = 16,
#                 learning_rate_seg: float = 0.00001,
#                 learning_rate_slope: float = 0.00001,
#                 n_epochs: int = 10):
#     print(f"Training Configuration:")
#     print(f"Model Name: {model_name}")
#     print(f"Number of Classes: {num_classes}")
#     print(f"Batch Size: {batch_size}")
#     print(f"Learning Rate (Segmentation): {learning_rate_seg}")
#     print(f"Learning Rate (Slope): {learning_rate_slope}")
#     print(f"Number of Epochs: {n_epochs}")
#     print("-" * 50)

#     model_output_path.mkdir(parents=True, exist_ok=True)
    
#     train_image_paths = sorted(list((train_data_path / 'image').glob('*.png')))
#     train_image_paths = [str(path) for path in train_image_paths]
#     train_label_paths = sorted(list((train_data_path / 'label').glob('*.png')))
#     train_label_paths = [str(path) for path in train_label_paths]
#     train_slope_paths = sorted(list((train_data_path / 'slope').glob('*.png')))
#     train_slope_paths = [str(path) for path in train_slope_paths]
#     val_image_paths = sorted(list((val_data_path / 'image').glob('*.png')))
#     val_image_paths = [str(path) for path in val_image_paths]
#     val_label_paths = sorted(list((val_data_path / 'label').glob('*.png')))
#     val_label_paths = [str(path) for path in val_label_paths]
#     val_slope_paths = sorted(list((val_data_path / 'slope').glob('*.png')))
#     val_slope_paths = [str(path) for path in val_slope_paths]
    
#     trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    
#     train_dataset = SegmentationDataset(image_paths=train_image_paths,
#                                         label_paths=train_label_paths,
#                                         slope_paths=train_slope_paths,
#                                         transforms=trans,
#                                         max_slope=overall_max)
    
#     val_dataset = SegmentationDataset(image_paths=val_image_paths,
#                                       label_paths=val_label_paths,
#                                       slope_paths=val_slope_paths,
#                                       transforms=trans,
#                                       max_slope=overall_max)

#     print(f"[INFO] found {len(train_dataset)}, examples in the training set...")
#     print(f"[INFO] {len(train_image_paths)},{len(train_label_paths)}")
#     print(f"[INFO] found {len(val_dataset)}, examples in the validation set...")
#     print(f"[INFO] {len(val_image_paths)},{len(val_label_paths)}")

#     train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
#         batch_size=batch_size, pin_memory=PIN_MEMORY,
#         num_workers=os.cpu_count())
#     val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
#         batch_size=batch_size, pin_memory=PIN_MEMORY,
#         num_workers=os.cpu_count())

#     focal_loss = torch.hub.load(
#         '/scratch/clear/aboccala/PASSION/passion/segmentation/pytorch-multi-class-focal-loss',
#         model='focal_loss',
#         source='local',
#         alpha=None,
#         gamma=2,
#         reduction='mean',
#         device=DEVICE,
#         dtype=torch.float32,
#         force_reload=False
#     )
#     loss_func = focal_loss

#     print(f'Initializing ResNetUNet with {num_classes} classes...')
#     unet = model_slopes.ResNetUNet(num_classes).to(DEVICE)
#     unet.to(DEVICE)

#     # Separate optimizers for segmentation and slope
#     optimizer_seg = torch.optim.Adam(unet.parameters(), lr=learning_rate_seg, weight_decay=1e-5)
#     optimizer_slope = torch.optim.Adam(unet.parameters(), lr=learning_rate_slope, weight_decay=1e-5)

#     scheduler_seg = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_seg, 'min', factor=0.1, patience=5, verbose=True)
#     scheduler_slope = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_slope, 'min', factor=0.1, patience=5, verbose=True)

#     awl = AutomaticWeightedLoss(2).to(DEVICE)

#     train_steps = len(train_dataset) // batch_size
#     val_steps = len(val_dataset) // batch_size
#     H = {"train_loss": [], "val_loss": [], "train_acc":[], "val_acc":[]}

#     best_val_score = 0.0

#     tensorboard_path = pathlib.Path(model_output_path / (model_name.split('.')[0]))
#     tensorboard_path.mkdir(parents=True, exist_ok=True)
#     writer = SummaryWriter(log_dir=str(tensorboard_path))

#     print("[INFO] training the network...")
#     start_time = time.time()
#     for e in tqdm(range(n_epochs)):
#         confmat = torchmetrics.ConfusionMatrix(task='multiclass',num_classes=num_classes).to(DEVICE)
#         cm = torch.zeros((num_classes,num_classes)).to(DEVICE)
#         unet.train()
#         total_train_loss = 0
#         total_train_abs_err = 0
#         total_val_loss = 0
#         total_correct = 0
#         total = 0
#         total_seg_loss = 0
#         total_slope_loss = 0
#         jaccard = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(DEVICE)
#         total_val, total_val_correct = 0, 0

#         for (i, (x, y, z)) in enumerate(train_loader):
#             y *= 255
#             (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
#             pred, pred2 = unet(x)

#             y = y.long().squeeze()
#             loss_seg = loss_func(pred, y)
#             loss_slope = smooth_l1_loss_with_center_penalty(pred2 * overall_max, z * overall_max)

#             loss = awl(loss_seg, loss_slope*1.0)  # Increase the influence of slope loss

#             optimizer_seg.zero_grad()
#             optimizer_slope.zero_grad()
#             loss.backward()
#             optimizer_seg.step()
#             optimizer_slope.step()

#             total_train_loss += loss.detach()
#             total_seg_loss += loss_seg.detach()
#             total_slope_loss += loss_slope.detach()
#             train_abs_err = depth_error(pred2, z, overall_max)
#             total_train_abs_err += train_abs_err

#             pred = torch.argmax(pred, dim=1)
#             total += y.size().numel()
#             total_correct += (pred == y).sum().item()
#             cm += confmat(pred, y)

#             # Debugging: Print slope predictions for the first batch
#             if i == 0:
#                 print("Slope Predictions (first batch):")
#                 print(pred2[0].cpu().detach().numpy() * overall_max)  # Denormalize for debugging

#         avg_seg_loss = total_seg_loss / train_steps
#         avg_slope_loss = total_slope_loss / train_steps
#         train_iou = intersection_over_union(cm.cpu().detach().numpy())
#         filtered_train_iou = train_iou[train_iou != 0]
#         mean_train_iou = np.mean(filtered_train_iou)
#         mean_abs_slope_error = total_train_abs_err / len(train_loader)
#         print('Train Class IOU', filtered_train_iou)
#         print('Train Mean Class IOU:{}'.format(mean_train_iou))
#         print('Train Mean Abs Slope Error: {:.4f}'.format(mean_abs_slope_error))
#         print('Train Avg Segmentation Loss: {:.4f}'.format(avg_seg_loss))
#         print('Train Avg Slope Loss: {:.4f}'.format(avg_slope_loss))
#         print('AWL Params:', awl.params.detach().cpu().numpy())

#         with torch.no_grad():
#             confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes).to(DEVICE)
#             cm = torch.zeros((num_classes, num_classes)).to(DEVICE)
#             unet.eval()
#             total_val_loss = 0
#             total_val_abs_err = 0
#             total_val_correct = 0
#             total_val = 0
#             total_val_seg_loss = 0
#             total_val_slope_loss = 0
#             for (x, y, z) in val_loader:
#                 y *= 255
#                 (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
#                 pred, pred2 = unet(x)

#                 y = y.long().squeeze()
#                 loss_seg = loss_func(pred, y)
#                 loss_slope = smooth_l1_loss_with_center_penalty(pred2 * overall_max, z * overall_max)
#                 loss = awl(loss_seg, loss_slope*1.0)  # Increase the influence of slope loss
#                 total_val_loss += loss.item()
#                 total_val_seg_loss += loss_seg.item()
#                 total_val_slope_loss += loss_slope.item()

#                 val_abs_err = depth_error(pred2, z, overall_max)
#                 total_val_abs_err += val_abs_err

#                 pred = torch.argmax(pred, dim=1)
#                 total_val += y.size().numel()
#                 total_val_correct += (pred == y).sum().item()

#                 cm += confmat(pred, y)

#             avg_val_seg_loss = total_val_seg_loss / val_steps
#             avg_val_slope_loss = total_val_slope_loss / val_steps
#             val_iou = intersection_over_union(cm.cpu().detach().numpy())
#             filtered_val_iou = val_iou[val_iou != 0]
#             mean_val_iou = np.mean(filtered_val_iou)
#             mean_abs_slope_error = total_val_abs_err / len(val_loader)
#             print('Val Class IOU', filtered_val_iou)
#             print('Val Mean Class IOU:{}'.format(mean_val_iou))
#             print('Val Mean Abs Slope Error: {:.4f}'.format(mean_abs_slope_error))
#             print('Val Avg Segmentation Loss: {:.4f}'.format(avg_val_seg_loss))
#             print('Val Avg Slope Loss: {:.4f}'.format(avg_val_slope_loss))
#             print('AWL Params:', awl.params.detach().cpu().numpy())

#             if mean_val_iou > best_val_score:
#                 print('Better Model Found: {} > {}'.format(mean_val_iou, best_val_score))
#                 best_val_score = mean_val_iou
#                 torch.save(unet, model_output_path / model_name)

#         avg_train_loss = total_train_loss / train_steps
#         avg_val_loss = total_val_loss / val_steps
#         avg_train_acc = total_correct / total
#         avg_val_acc = total_val_correct / total_val

#         H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
#         H["val_loss"].append(avg_val_loss)
#         H["train_acc"].append(avg_train_acc)
#         H["val_acc"].append(avg_val_acc)
#         print("[INFO] EPOCH: {}/{}".format(e + 1, n_epochs))
#         print("Train loss: {:.6f}, Val loss: {:.4f} Train Acc: {:.6f}, Val Acc: {:.4f}".format(
#             avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc))
#         scheduler_seg.step(avg_val_loss)
#         scheduler_slope.step(avg_val_loss)

#     end_time = time.time()
#     print("[INFO] total time taken to train the model: {:.2f}s".format(
#         end_time - start_time))

#     torch.save(unet, model_output_path / 'model_last.pth')
#     writer.flush()
#     print('Best Val Score:{}'.format(best_val_score))


# import pathlib
# import cv2
# import numpy as np
# import torch
# import torchvision
# import torchmetrics
# import os
# import time
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# import torch.optim
# import torch.nn.functional as F
# import torch.nn as nn

# from passion.segmentation import model_slopes

# cv2.setNumThreads(4)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PIN_MEMORY = True if DEVICE == "cuda" else False

# class SegmentationDataset(torch.utils.data.Dataset):
#     def __init__(self, image_paths, label_paths, slope_paths, transforms, max_slope):
#         self.image_paths = image_paths
#         self.label_paths = label_paths
#         self.slope_paths = slope_paths
#         self.transforms = transforms
#         self.max_slope = max_slope
    
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         imagePath = self.image_paths[idx]
#         image = cv2.imread(str(imagePath))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(self.label_paths[idx], 0)
#         slope = cv2.imread(self.slope_paths[idx], 0)
        
#         if self.transforms is not None:
#             image = self.transforms(image)
#             mask = self.transforms(mask)
#             slope = torch.from_numpy(slope).float() / self.max_slope  # Normalize slope values to [0, 1]

#         return (image, mask, slope)

# def find_slope_max(slope_paths):
#     max_value = float('-inf')
#     for slope_path in tqdm(slope_paths):
#         slope = cv2.imread(str(slope_path), cv2.IMREAD_GRAYSCALE)
#         max_value = max(max_value, slope.max())
#     return max_value

# train_slope_paths = sorted(list((pathlib.Path('/scratch/clear/aboccala/training/RID/output_mix_ge_va/train_ge/slope')).glob('*.png')))
# val_slope_paths = sorted(list((pathlib.Path('/scratch/clear/aboccala/training/RID/output_mix_ge_va/test_va/slope')).glob('*.png')))

# train_max = find_slope_max(train_slope_paths)
# val_max = find_slope_max(val_slope_paths)
# overall_max = max(train_max, val_max)

# print(f"Overall Max Slope: {overall_max}")

# class AutomaticWeightedLoss(nn.Module):
#     def __init__(self, num=2):
#         super(AutomaticWeightedLoss, self).__init__()
#         params = torch.ones(num, requires_grad=True)
#         self.params = torch.nn.Parameter(params)

#     def forward(self, *x):
#         loss_sum = 0
#         for i, loss in enumerate(x):
#             loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
#         return loss_sum

# if __name__ == '__main__':
#     awl = AutomaticWeightedLoss(2)
#     print(awl.parameters())

# def intersection_over_union(confusion_matrix):
#     intersection = np.diag(confusion_matrix)
#     union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    
#     epsilon = 1e-9
#     iou = intersection / (union + epsilon)
#     iou = np.nan_to_num(iou, nan=0.0)
#     return iou

# def depth_error(x_pred, x_output, max_slope):
#     x_pred = x_pred * max_slope
#     x_output = x_output * max_slope

#     abs_err = torch.abs(x_pred - x_output)
#     total_elements = x_output.numel()
#     return (torch.sum(abs_err) / total_elements).item()

# def smooth_l1_loss_with_center_penalty(pred, target, beta=1.0):
#     # Smooth L1 Loss (Huber Loss)
#     diff = torch.abs(pred - target)
#     smooth_l1_loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    
#     # Calculate the distance penalty based on distance from the center
#     batch_size, channels, height, width = pred.shape
#     y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
#     y_coords = y_coords.float().to(pred.device) / height
#     x_coords = x_coords.float().to(pred.device) / width
#     distance_penalty = torch.sqrt((x_coords - 0.5)**2 + (y_coords - 0.5)**2)
#     distance_penalty = distance_penalty.unsqueeze(0).unsqueeze(0)  # Make it (1, 1, height, width)

#     # Apply the distance penalty to the loss
#     penalized_loss = smooth_l1_loss * distance_penalty

#     return penalized_loss.mean()

# def train_model(train_data_path: pathlib.Path,
#                 val_data_path: pathlib.Path,
#                 model_output_path: pathlib.Path,
#                 model_name: str,
#                 num_classes: int = 18,
#                 batch_size: int = 16,
#                 learning_rate_seg: float = 0.0001,
#                 learning_rate_slope: float = 0.0001,
#                 n_epochs: int = 20):
#     print(f"Training Configuration:")
#     print(f"Model Name: {model_name}")
#     print(f"Number of Classes: {num_classes}")
#     print(f"Batch Size: {batch_size}")
#     print(f"Learning Rate (Segmentation): {learning_rate_seg}")
#     print(f"Learning Rate (Slope): {learning_rate_slope}")
#     print(f"Number of Epochs: {n_epochs}")
#     print("-" * 50)

#     model_output_path.mkdir(parents=True, exist_ok=True)
    
#     train_image_paths = sorted(list((train_data_path / 'image').glob('*.png')))
#     train_image_paths = [str(path) for path in train_image_paths]
#     train_label_paths = sorted(list((train_data_path / 'label').glob('*.png')))
#     train_label_paths = [str(path) for path in train_label_paths]
#     train_slope_paths = sorted(list((train_data_path / 'slope').glob('*.png')))
#     train_slope_paths = [str(path) for path in train_slope_paths]
#     val_image_paths = sorted(list((val_data_path / 'image').glob('*.png')))
#     val_image_paths = [str(path) for path in val_image_paths]
#     val_label_paths = sorted(list((val_data_path / 'label').glob('*.png')))
#     val_label_paths = [str(path) for path in val_label_paths]
#     val_slope_paths = sorted(list((val_data_path / 'slope').glob('*.png')))
#     val_slope_paths = [str(path) for path in val_slope_paths]
    
#     trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    
#     train_dataset = SegmentationDataset(image_paths=train_image_paths,
#                                         label_paths=train_label_paths,
#                                         slope_paths=train_slope_paths,
#                                         transforms=trans,
#                                         max_slope=overall_max)
    
#     val_dataset = SegmentationDataset(image_paths=val_image_paths,
#                                       label_paths=val_label_paths,
#                                       slope_paths=val_slope_paths,
#                                       transforms=trans,
#                                       max_slope=overall_max)

#     print(f"[INFO] found {len(train_dataset)}, examples in the training set...")
#     print(f"[INFO] {len(train_image_paths)},{len(train_label_paths)}")
#     print(f"[INFO] found {len(val_dataset)}, examples in the validation set...")
#     print(f"[INFO] {len(val_image_paths)},{len(val_label_paths)}")

#     train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
#         batch_size=batch_size, pin_memory=PIN_MEMORY,
#         num_workers=os.cpu_count())
#     val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
#         batch_size=batch_size, pin_memory=PIN_MEMORY,
#         num_workers=os.cpu_count())

#     focal_loss = torch.hub.load(
#         '/scratch/clear/aboccala/PASSION/passion/segmentation/pytorch-multi-class-focal-loss',
#         model='focal_loss',
#         source='local',
#         alpha=None,
#         gamma=2,
#         reduction='mean',
#         device=DEVICE,
#         dtype=torch.float32,
#         force_reload=False
#     )
#     loss_func = focal_loss

#     print(f'Initializing ResNetUNet with {num_classes} classes...')
#     unet = model_slopes.ResNetUNet(num_classes).to(DEVICE)
#     unet.to(DEVICE)

#     # Separate optimizers for segmentation and slope
#     optimizer_seg = torch.optim.Adam(unet.parameters(), lr=learning_rate_seg, weight_decay=1e-5)
#     optimizer_slope = torch.optim.Adam(unet.parameters(), lr=learning_rate_slope, weight_decay=1e-5)

#     scheduler_seg = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_seg, 'min', factor=0.1, patience=5, verbose=True)
#     scheduler_slope = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_slope, 'min', factor=0.1, patience=5, verbose=True)

#     awl = AutomaticWeightedLoss(2).to(DEVICE)

#     train_steps = len(train_dataset) // batch_size
#     val_steps = len(val_dataset) // batch_size
#     H = {"train_loss": [], "val_loss": [], "train_acc":[], "val_acc":[]}

#     best_val_score = 0.0

#     tensorboard_path = pathlib.Path(model_output_path / (model_name.split('.')[0]))
#     tensorboard_path.mkdir(parents=True, exist_ok=True)
#     writer = SummaryWriter(log_dir=str(tensorboard_path))

#     print("[INFO] training the network...")
#     start_time = time.time()
#     for e in tqdm(range(n_epochs)):
#         confmat = torchmetrics.ConfusionMatrix(task='multiclass',num_classes=num_classes).to(DEVICE)
#         cm = torch.zeros((num_classes,num_classes)).to(DEVICE)
#         unet.train()
#         total_train_loss = 0
#         total_train_abs_err = 0
#         total_val_loss = 0
#         total_correct = 0
#         total = 0
#         total_seg_loss = 0
#         total_slope_loss = 0
#         jaccard = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(DEVICE)
#         total_val, total_val_correct = 0, 0

#         for (i, (x, y, z)) in enumerate(train_loader):
#             y *= 255
#             (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
#             pred, pred2 = unet(x)

#             y = y.long().squeeze()
#             loss_seg = loss_func(pred, y)
#             pred2_denorm = pred2 * overall_max  # Denormalize pred2
#             z_denorm = z * overall_max  # Denormalize z
#             loss_slope = smooth_l1_loss_with_center_penalty(pred2_denorm, z_denorm)

#             loss = awl(loss_seg, loss_slope * 1.0)  # Adjust the influence of slope loss

#             optimizer_seg.zero_grad()
#             optimizer_slope.zero_grad()
#             loss.backward()
#             optimizer_seg.step()
#             optimizer_slope.step()

#             total_train_loss += loss.detach()
#             total_seg_loss += loss_seg.detach()
#             total_slope_loss += loss_slope.detach()
#             train_abs_err = depth_error(pred2, z, overall_max)
#             total_train_abs_err += train_abs_err

#             pred = torch.argmax(pred, dim=1)
#             total += y.size().numel()
#             total_correct += (pred == y).sum().item()
#             cm += confmat(pred, y)

#             # Debugging: Print slope predictions for the first batch
#             if i == 0:
#                 print("Slope Predictions (first batch):")
#                 print(pred2[0].cpu().detach().numpy() * overall_max)  # Denormalize for debugging

#         avg_seg_loss = total_seg_loss / train_steps
#         avg_slope_loss = total_slope_loss / train_steps
#         train_iou = intersection_over_union(cm.cpu().detach().numpy())
#         filtered_train_iou = train_iou[train_iou != 0]
#         mean_train_iou = np.mean(filtered_train_iou)
#         mean_abs_slope_error = total_train_abs_err / len(train_loader)
#         print('Train Class IOU', filtered_train_iou)
#         print('Train Mean Class IOU:{}'.format(mean_train_iou))
#         print('Train Mean Abs Slope Error: {:.4f}'.format(mean_abs_slope_error))
#         print('Train Avg Segmentation Loss: {:.4f}'.format(avg_seg_loss))
#         print('Train Avg Slope Loss: {:.4f}'.format(avg_slope_loss))
#         print('AWL Params:', awl.params.detach().cpu().numpy())

#         with torch.no_grad():
#             confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes).to(DEVICE)
#             cm = torch.zeros((num_classes, num_classes)).to(DEVICE)
#             unet.eval()
#             total_val_loss = 0
#             total_val_abs_err = 0
#             total_val_correct = 0
#             total_val = 0
#             total_val_seg_loss = 0
#             total_val_slope_loss = 0
#             for (x, y, z) in val_loader:
#                 y *= 255
#                 (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
#                 pred, pred2 = unet(x)

#                 y = y.long().squeeze()
#                 loss_seg = loss_func(pred, y)
#                 pred2_denorm = pred2 * overall_max  # Denormalize pred2
#                 z_denorm = z * overall_max  # Denormalize z
#                 loss_slope = smooth_l1_loss_with_center_penalty(pred2_denorm, z_denorm)
#                 loss = awl(loss_seg, loss_slope * 1.0)  # Adjust the influence of slope loss
#                 total_val_loss += loss.item()
#                 total_val_seg_loss += loss_seg.item()
#                 total_val_slope_loss += loss_slope.item()

#                 val_abs_err = depth_error(pred2, z, overall_max)
#                 total_val_abs_err += val_abs_err

#                 pred = torch.argmax(pred, dim=1)
#                 total_val += y.size().numel()
#                 total_val_correct += (pred == y).sum().item()

#                 cm += confmat(pred, y)

#             avg_val_seg_loss = total_val_seg_loss / val_steps
#             avg_val_slope_loss = total_val_slope_loss / val_steps
#             val_iou = intersection_over_union(cm.cpu().detach().numpy())
#             filtered_val_iou = val_iou[val_iou != 0]
#             mean_val_iou = np.mean(filtered_val_iou)
#             mean_abs_slope_error = total_val_abs_err / len(val_loader)
#             print('Val Class IOU', filtered_val_iou)
#             print('Val Mean Class IOU:{}'.format(mean_val_iou))
#             print('Val Mean Abs Slope Error: {:.4f}'.format(mean_abs_slope_error))
#             print('Val Avg Segmentation Loss: {:.4f}'.format(avg_val_seg_loss))
#             print('Val Avg Slope Loss: {:.4f}'.format(avg_val_slope_loss))
#             print('AWL Params:', awl.params.detach().cpu().numpy())

#             if mean_val_iou > best_val_score:
#                 print('Better Model Found: {} > {}'.format(mean_val_iou, best_val_score))
#                 best_val_score = mean_val_iou
#                 torch.save(unet, model_output_path / model_name)

#         avg_train_loss = total_train_loss / train_steps
#         avg_val_loss = total_val_loss / val_steps
#         avg_train_acc = total_correct / total
#         avg_val_acc = total_val_correct / total_val

#         H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
#         H["val_loss"].append(avg_val_loss)
#         H["train_acc"].append(avg_train_acc)
#         H["val_acc"].append(avg_val_acc)
#         print("[INFO] EPOCH: {}/{}".format(e + 1, n_epochs))
#         print("Train loss: {:.6f}, Val loss: {:.4f} Train Acc: {:.6f}, Val Acc: {:.4f}".format(
#             avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc))
#         scheduler_seg.step(avg_val_loss)
#         scheduler_slope.step(avg_val_loss)

#     end_time = time.time()
#     print("[INFO] total time taken to train the model: {:.2f}s".format(
#         end_time - start_time))

#     torch.save(unet, model_output_path / 'model_last.pth')
#     writer.flush()
#     print('Best Val Score:{}'.format(best_val_score))


# import pathlib
# import cv2
# import numpy as np
# import torch
# import torchvision
# import torchmetrics
# import os
# import time
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# import torch.optim
# import torch.nn.functional as F
# import torch.nn as nn

# from passion.segmentation import model_slopes

# cv2.setNumThreads(4)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PIN_MEMORY = True if DEVICE == "cuda" else False

# class SegmentationDataset(torch.utils.data.Dataset):
#     def __init__(self, image_paths, label_paths, slope_paths, transforms):
#         self.image_paths = image_paths
#         self.label_paths = label_paths
#         self.slope_paths = slope_paths
#         self.transforms = transforms
    
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         imagePath = self.image_paths[idx]
#         image = cv2.imread(str(imagePath))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(self.label_paths[idx], 0)
#         slope = cv2.imread(self.slope_paths[idx], 0)
        
#         if self.transforms is not None:
#             image = self.transforms(image)
#             mask = self.transforms(mask)
#             slope = torch.from_numpy(slope).float()  # Use raw slope values

#         return (image, mask, slope)

# # def find_slope_max(slope_paths):
# #     max_value = float('-inf')
# #     for slope_path in tqdm(slope_paths):
# #         slope = cv2.imread(str(slope_path), cv2.IMREAD_GRAYSCALE)
# #         max_value = max(max_value, slope.max())
# #     return max_value

# # train_slope_paths = sorted(list((pathlib.Path('/scratch/clear/aboccala/training/RID/output_mix_ge_va/train_ge/slope')).glob('*.png')))
# # val_slope_paths = sorted(list((pathlib.Path('/scratch/clear/aboccala/training/RID/output_mix_ge_va/test_va/slope')).glob('*.png')))

# # train_max = find_slope_max(train_slope_paths)
# # val_max = find_slope_max(val_slope_paths)
# # overall_max = max(train_max, val_max)

# # print(f"Overall Max Slope: {overall_max}")

# class AutomaticWeightedLoss(nn.Module):
#     def __init__(self, num=2):
#         super(AutomaticWeightedLoss, self).__init__()
#         params = torch.ones(num, requires_grad=True)
#         self.params = torch.nn.Parameter(params)

#     def forward(self, *x):
#         loss_sum = 0
#         for i, loss in enumerate(x):
#             loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
#         return loss_sum

# if __name__ == '__main__':
#     awl = AutomaticWeightedLoss(2)
#     print(awl.parameters())

# def intersection_over_union(confusion_matrix):
#     intersection = np.diag(confusion_matrix)
#     union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    
#     epsilon = 1e-9
#     iou = intersection / (union + epsilon)
#     iou = np.nan_to_num(iou, nan=0.0)
#     return iou

# def depth_error(x_pred, x_output):
#     abs_err = torch.abs(x_pred - x_output)
#     total_elements = x_output.numel()
#     return (torch.sum(abs_err) / total_elements).item()

# def smooth_l1_loss_with_center_penalty(pred, target, beta=1.0):
#     # Smooth L1 Loss (Huber Loss)
#     diff = torch.abs(pred - target)
#     smooth_l1_loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    
#     # Calculate the distance penalty based on distance from the center
#     batch_size, channels, height, width = pred.shape
#     y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
#     y_coords = y_coords.float().to(pred.device) / height
#     x_coords = x_coords.float().to(pred.device) / width
#     distance_penalty = torch.sqrt((x_coords - 0.5)**2 + (y_coords - 0.5)**2)
#     distance_penalty = distance_penalty.unsqueeze(0).unsqueeze(0)  # Make it (1, 1, height, width)

#     # Apply the distance penalty to the loss
#     penalized_loss = smooth_l1_loss * distance_penalty

#     return penalized_loss.mean()

# def train_model(train_data_path: pathlib.Path,
#                 val_data_path: pathlib.Path,
#                 model_output_path: pathlib.Path,
#                 model_name: str,
#                 num_classes: int = 18,
#                 batch_size: int = 16,
#                 learning_rate_seg: float = 0.0001,
#                 learning_rate_slope: float = 0.0001,
#                 n_epochs: int = 20):
#     print(f"Training Configuration:")
#     print(f"Model Name: {model_name}")
#     print(f"Number of Classes: {num_classes}")
#     print(f"Batch Size: {batch_size}")
#     print(f"Learning Rate (Segmentation): {learning_rate_seg}")
#     print(f"Learning Rate (Slope): {learning_rate_slope}")
#     print(f"Number of Epochs: {n_epochs}")
#     print("-" * 50)

#     model_output_path.mkdir(parents=True, exist_ok=True)
    
#     train_image_paths = sorted(list((train_data_path / 'image').glob('*.png')))
#     train_image_paths = [str(path) for path in train_image_paths]
#     train_label_paths = sorted(list((train_data_path / 'label').glob('*.png')))
#     train_label_paths = [str(path) for path in train_label_paths]
#     train_slope_paths = sorted(list((train_data_path / 'slope').glob('*.png')))
#     train_slope_paths = [str(path) for path in train_slope_paths]
#     val_image_paths = sorted(list((val_data_path / 'image').glob('*.png')))
#     val_image_paths = [str(path) for path in val_image_paths]
#     val_label_paths = sorted(list((val_data_path / 'label').glob('*.png')))
#     val_label_paths = [str(path) for path in val_label_paths]
#     val_slope_paths = sorted(list((val_data_path / 'slope').glob('*.png')))
#     val_slope_paths = [str(path) for path in val_slope_paths]
    
#     trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    
#     train_dataset = SegmentationDataset(image_paths=train_image_paths,
#                                         label_paths=train_label_paths,
#                                         slope_paths=train_slope_paths,
#                                         transforms=trans,
#                                        )
    
#     val_dataset = SegmentationDataset(image_paths=val_image_paths,
#                                       label_paths=val_label_paths,
#                                       slope_paths=val_slope_paths,
#                                       transforms=trans,
#                                  )

#     print(f"[INFO] found {len(train_dataset)}, examples in the training set...")
#     print(f"[INFO] {len(train_image_paths)},{len(train_label_paths)}")
#     print(f"[INFO] found {len(val_dataset)}, examples in the validation set...")
#     print(f"[INFO] {len(val_image_paths)},{len(val_label_paths)}")

#     train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
#         batch_size=batch_size, pin_memory=PIN_MEMORY,
#         num_workers=os.cpu_count())
#     val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
#         batch_size=batch_size, pin_memory=PIN_MEMORY,
#         num_workers=os.cpu_count())

#     focal_loss = torch.hub.load(
#         '/scratch/clear/aboccala/PASSION/passion/segmentation/pytorch-multi-class-focal-loss',
#         model='focal_loss',
#         source='local',
#         alpha=None,
#         gamma=2,
#         reduction='mean',
#         device=DEVICE,
#         dtype=torch.float32,
#         force_reload=False
#     )
#     loss_func = focal_loss

#     print(f'Initializing ResNetUNet with {num_classes} classes...')
#     unet = model_slopes.ResNetUNet(num_classes).to(DEVICE)
#     unet.to(DEVICE)

#     # Separate optimizers for segmentation and slope
#     optimizer_seg = torch.optim.Adam(unet.parameters(), lr=learning_rate_seg, weight_decay=1e-5)
#     optimizer_slope = torch.optim.Adam(unet.parameters(), lr=learning_rate_slope, weight_decay=1e-5)

#     scheduler_seg = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_seg, 'min', factor=0.1, patience=5, verbose=True)
#     scheduler_slope = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_slope, 'min', factor=0.1, patience=5, verbose=True)

#     awl = AutomaticWeightedLoss(2).to(DEVICE)

#     train_steps = len(train_dataset) // batch_size
#     val_steps = len(val_dataset) // batch_size
#     H = {"train_loss": [], "val_loss": [], "train_acc":[], "val_acc":[]}

#     best_val_score = 0.0

#     tensorboard_path = pathlib.Path(model_output_path / (model_name.split('.')[0]))
#     tensorboard_path.mkdir(parents=True, exist_ok=True)
#     writer = SummaryWriter(log_dir=str(tensorboard_path))

#     print("[INFO] training the network...")
#     start_time = time.time()
#     for e in tqdm(range(n_epochs)):
#         confmat = torchmetrics.ConfusionMatrix(task='multiclass',num_classes=num_classes).to(DEVICE)
#         cm = torch.zeros((num_classes,num_classes)).to(DEVICE)
#         unet.train()
#         total_train_loss = 0
#         total_train_abs_err = 0
#         total_val_loss = 0
#         total_correct = 0
#         total = 0
#         total_seg_loss = 0
#         total_slope_loss = 0
#         jaccard = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(DEVICE)
#         total_val, total_val_correct = 0, 0

#         for (i, (x, y, z)) in enumerate(train_loader):
#             y *= 255
#             (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
#             pred, pred2 = unet(x)

#             y = y.long().squeeze()
#             loss_seg = loss_func(pred, y)
#             loss_slope = smooth_l1_loss_with_center_penalty(pred2, z)

#             loss = awl(loss_seg, loss_slope * 1.0)  # Adjust the influence of slope loss

#             optimizer_seg.zero_grad()
#             optimizer_slope.zero_grad()
#             loss.backward()
#             optimizer_seg.step()
#             optimizer_slope.step()

#             total_train_loss += loss.detach()
#             total_seg_loss += loss_seg.detach()
#             total_slope_loss += loss_slope.detach()
#             train_abs_err = depth_error(pred2, z)
#             total_train_abs_err += train_abs_err

#             pred = torch.argmax(pred, dim=1)
#             total += y.size().numel()
#             total_correct += (pred == y).sum().item()
#             cm += confmat(pred, y)

#             # Debugging: Print slope predictions for the first batch
#             if i == 0:
#                 print("Slope Predictions (first batch):")
#                 print(pred2[0].cpu().detach().numpy())

#         avg_seg_loss = total_seg_loss / train_steps
#         avg_slope_loss = total_slope_loss / train_steps
#         train_iou = intersection_over_union(cm.cpu().detach().numpy())
#         filtered_train_iou = train_iou[train_iou != 0]
#         mean_train_iou = np.mean(filtered_train_iou)
#         mean_abs_slope_error = total_train_abs_err / len(train_loader)
#         print('Train Class IOU', filtered_train_iou)
#         print('Train Mean Class IOU:{}'.format(mean_train_iou))
#         print('Train Mean Abs Slope Error: {:.4f}'.format(mean_abs_slope_error))
#         print('Train Avg Segmentation Loss: {:.4f}'.format(avg_seg_loss))
#         print('Train Avg Slope Loss: {:.4f}'.format(avg_slope_loss))
#         print('AWL Params:', awl.params.detach().cpu().numpy())

#         with torch.no_grad():
#             confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes).to(DEVICE)
#             cm = torch.zeros((num_classes, num_classes)).to(DEVICE)
#             unet.eval()
#             total_val_loss = 0
#             total_val_abs_err = 0
#             total_val_correct = 0
#             total_val = 0
#             total_val_seg_loss = 0
#             total_val_slope_loss = 0
#             for (x, y, z) in val_loader:
#                 y *= 255
#                 (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
#                 pred, pred2 = unet(x)

#                 y = y.long().squeeze()
#                 loss_seg = loss_func(pred, y)
#                 loss_slope = smooth_l1_loss_with_center_penalty(pred2, z)
#                 loss = awl(loss_seg, loss_slope * 1.0)  # Adjust the influence of slope loss
#                 total_val_loss += loss.item()
#                 total_val_seg_loss += loss_seg.item()
#                 total_val_slope_loss += loss_slope.item()

#                 val_abs_err = depth_error(pred2, z)
#                 total_val_abs_err += val_abs_err

#                 pred = torch.argmax(pred, dim=1)
#                 total_val += y.size().numel()
#                 total_val_correct += (pred == y).sum().item()

#                 cm += confmat(pred, y)

#             avg_val_seg_loss = total_val_seg_loss / val_steps
#             avg_val_slope_loss = total_val_slope_loss / val_steps
#             val_iou = intersection_over_union(cm.cpu().detach().numpy())
#             filtered_val_iou = val_iou[val_iou != 0]
#             mean_val_iou = np.mean(filtered_val_iou)
#             mean_abs_slope_error = total_val_abs_err / len(val_loader)
#             print('Val Class IOU', filtered_val_iou)
#             print('Val Mean Class IOU:{}'.format(mean_val_iou))
#             print('Val Mean Abs Slope Error: {:.4f}'.format(mean_abs_slope_error))
#             print('Val Avg Segmentation Loss: {:.4f}'.format(avg_val_seg_loss))
#             print('Val Avg Slope Loss: {:.4f}'.format(avg_val_slope_loss))
#             print('AWL Params:', awl.params.detach().cpu().numpy())

#             if mean_val_iou > best_val_score:
#                 print('Better Model Found: {} > {}'.format(mean_val_iou, best_val_score))
#                 best_val_score = mean_val_iou
#                 torch.save(unet, model_output_path / model_name)

#         avg_train_loss = total_train_loss / train_steps
#         avg_val_loss = total_val_loss / val_steps
#         avg_train_acc = total_correct / total
#         avg_val_acc = total_val_correct / total_val

#         H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
#         H["val_loss"].append(avg_val_loss)
#         H["train_acc"].append(avg_train_acc)
#         H["val_acc"].append(avg_val_acc)
#         print("[INFO] EPOCH: {}/{}".format(e + 1, n_epochs))
#         print("Train loss: {:.6f}, Val loss: {:.4f} Train Acc: {:.6f}, Val Acc: {:.4f}".format(
#             avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc))
#         scheduler_seg.step(avg_val_loss)
#         scheduler_slope.step(avg_val_loss)

#     end_time = time.time()
#     print("[INFO] total time taken to train the model: {:.2f}s".format(
#         end_time - start_time))

#     torch.save(unet, model_output_path / 'model_last.pth')
#     writer.flush()
#     print('Best Val Score:{}'.format(best_val_score))


import pathlib
import cv2
import numpy as np
import torch
import torchvision
import torchmetrics
import os
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.optim
import torch.nn.functional as F
import torch.nn as nn

from passion.segmentation import model_slopes

cv2.setNumThreads(4)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, label_paths, slope_paths, transforms):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.slope_paths = slope_paths
        print(f"Number of images: {len(self.image_paths)}")
        print(f"Number of labels: {len(self.label_paths)}")
        print(f"Number of slopes: {len(self.slope_paths)}")
        
        assert len(self.image_paths) == len(self.label_paths) == len(self.slope_paths), \
            "Mismatch in the number of images, labels, and slopes"
    
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        imagePath = self.image_paths[idx]
        image = cv2.imread(str(imagePath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.label_paths[idx], 0)
        slope = cv2.imread(self.slope_paths[idx], 0)
        # print(len(image))
        # print(len(slope))

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
            slope = self.transforms(slope) 

        return (image, mask, slope)

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())

def intersection_over_union(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    
    epsilon = 1e-9
    iou = intersection / (union + epsilon)
    iou = np.nan_to_num(iou, nan=0.0)
    return iou

def train_model(train_data_path: pathlib.Path,
                val_data_path: pathlib.Path,
                model_output_path: pathlib.Path,
                model_name: str,
                num_classes_orient: int = 18,
                num_classes_slope: int = 8,
                batch_size: int = 16,
                learning_rate_seg: float = 0.0001,
                learning_rate_slope: float = 0.0001,
                n_epochs: int = 20):
    print(f"Training Configuration:")
    print(f"Model Name: {model_name}")
    print(f"Number of Classes (Orient): {num_classes_orient}")
    print(f"Number of Classes (Slope): {num_classes_slope}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate (Segmentation): {learning_rate_seg}")
    print(f"Learning Rate (Slope): {learning_rate_slope}")
    print(f"Number of Epochs: {n_epochs}")
    print("-" * 50)

    model_output_path.mkdir(parents=True, exist_ok=True)
    
    train_image_paths = sorted(list((train_data_path / 'image').glob('*.png')))
    train_image_paths = [str(path) for path in train_image_paths]
    train_label_paths = sorted(list((train_data_path / 'label').glob('*.png')))
    train_label_paths = [str(path) for path in train_label_paths]
    train_slope_paths = sorted(list((train_data_path / 'slope').glob('*.png')))
    train_slope_paths = [str(path) for path in train_slope_paths]
    val_image_paths = sorted(list((val_data_path / 'image').glob('*.png')))
    val_image_paths = [str(path) for path in val_image_paths]
    val_label_paths = sorted(list((val_data_path / 'label').glob('*.png')))
    val_label_paths = [str(path) for path in val_label_paths]
    val_slope_paths = sorted(list((val_data_path / 'slope').glob('*.png')))
    val_slope_paths = [str(path) for path in val_slope_paths]
    
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    
    train_dataset = SegmentationDataset(image_paths=train_image_paths,
                                        label_paths=train_label_paths,
                                        slope_paths=train_slope_paths,
                                        transforms=trans)
    
    val_dataset = SegmentationDataset(image_paths=val_image_paths,
                                      label_paths=val_label_paths,
                                      slope_paths=val_slope_paths,
                                      transforms=trans)

    print(f"[INFO] found {len(train_dataset)}, examples in the training set...")
    print(f"[INFO] {len(train_image_paths)},{len(train_label_paths)}")
    print(f"[INFO] found {len(val_dataset)}, examples in the validation set...")
    print(f"[INFO] {len(val_image_paths)},{len(val_label_paths)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
        batch_size=batch_size, pin_memory=PIN_MEMORY,
        num_workers=os.cpu_count())
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
        batch_size=batch_size, pin_memory=PIN_MEMORY,
        num_workers=os.cpu_count())

    focal_loss = torch.hub.load(
        '/scratch/clear/aboccala/PASSION/passion/segmentation/pytorch-multi-class-focal-loss',
        model='focal_loss',
        source='local',
        alpha=None,
        gamma=2,
        reduction='mean',
        device=DEVICE,
        dtype=torch.float32,
        force_reload=False
    )
    loss_func = focal_loss


    print(f'Initializing ResNetUNet with {num_classes_orient} orientation classes and {num_classes_slope} slope classes...')
    unet = model_slopes.ResNetUNet(n_class_orient=num_classes_orient, n_class_slope=num_classes_slope).to(DEVICE)
    unet.to(DEVICE)

    # Separate optimizers for segmentation and slope
    optimizer_seg = torch.optim.Adam(unet.parameters(), lr=learning_rate_seg, weight_decay=1e-5)
    optimizer_slope = torch.optim.Adam(unet.parameters(), lr=learning_rate_slope, weight_decay=1e-5)

    scheduler_seg = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_seg, 'min', factor=0.1, patience=5, verbose=True)
    scheduler_slope = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_slope, 'min', factor=0.1, patience=5, verbose=True)

    awl = AutomaticWeightedLoss(2).to(DEVICE)

    train_steps = len(train_dataset) // batch_size
    val_steps = len(val_dataset) // batch_size
    H = {"train_loss": [], "val_loss": [], "train_acc_orient":[], "val_acc_orient":[], "train_acc_slope":[], "val_acc_slope":[]}

    best_val_score = 0.0

    tensorboard_path = pathlib.Path(model_output_path / (model_name.split('.')[0]))
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_path))

    print("[INFO] training the network...")
    start_time = time.time()
    for e in tqdm(range(n_epochs)):
        confmat_orient = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes_orient).to(DEVICE)
        confmat_slope = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes_slope).to(DEVICE)
        cm_orient = torch.zeros((num_classes_orient, num_classes_orient)).to(DEVICE)
        cm_slope = torch.zeros((num_classes_slope, num_classes_slope)).to(DEVICE)
        unet.train()
        total_train_loss = 0
        total_correct_orient = 0
        total_orient = 0
        total_correct_slope = 0
        total_slope = 0
        total_seg_loss = 0
        total_slope_loss = 0
        total_val_correct_orient = 0
        total_val_correct_slope = 0
        total_val_orient = 0
        total_val_slope = 0
        total_val_seg_loss = 0
        total_val_slope_loss = 0

        for (i, (x, y, z)) in enumerate(train_loader):
            y *= 255
            z *= 255
            (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
            pred_orient, pred_slope = unet(x)

            y = y.long().squeeze()
            z = z.long().squeeze()
            loss_seg = loss_func(pred_orient, y)
            loss_slope = loss_func(pred_slope, z)

            # loss = awl(loss_seg, loss_slope * 1.0)  # Adjust the influence of slope loss
            loss = loss_seg+loss_slope

            optimizer_seg.zero_grad()
            optimizer_slope.zero_grad()
            loss.backward()
            optimizer_seg.step()
            optimizer_slope.step()

            total_train_loss += loss.detach()
            total_seg_loss += loss_seg.detach()
            total_slope_loss += loss_slope.detach()

            pred_orient = torch.argmax(pred_orient, dim=1)
            pred_slope = torch.argmax(pred_slope, dim=1)
            total_orient += y.size().numel()
            total_slope += z.size().numel()
            total_correct_orient += (pred_orient == y).sum().item()
            total_correct_slope += (pred_slope == z).sum().item()
            cm_orient += confmat_orient(pred_orient, y)
            cm_slope += confmat_slope(pred_slope, z)

            # # Debugging: Print slope predictions for the first batch
            # if i == 0:
            #     print("Slope Predictions (first batch):")
            #     print(pred_slope[0].cpu().detach().numpy())

        avg_seg_loss = total_seg_loss / train_steps
        avg_slope_loss = total_slope_loss / train_steps
        train_iou_orient = intersection_over_union(cm_orient.cpu().detach().numpy())
        train_iou_slope = intersection_over_union(cm_slope.cpu().detach().numpy())
        mean_train_iou_orient = np.mean(train_iou_orient[train_iou_orient != 0])
        mean_train_iou_slope = np.mean(train_iou_slope[train_iou_slope != 0])
        print('Train Orient Class IOU', train_iou_orient)
        print('Train Slope Class IOU', train_iou_slope)
        print('Train Mean Orient Class IOU:{}'.format(mean_train_iou_orient))
        print('Train Mean Slope Class IOU:{}'.format(mean_train_iou_slope))
        print('Train Avg Segmentation Loss: {:.4f}'.format(avg_seg_loss))
        print('Train Avg Slope Loss: {:.4f}'.format(avg_slope_loss))
        # print('AWL Params:', awl.params.detach().cpu().numpy())

        with torch.no_grad():
            cm_orient = torch.zeros((num_classes_orient, num_classes_orient)).to(DEVICE)
            cm_slope = torch.zeros((num_classes_slope, num_classes_slope)).to(DEVICE)
            unet.eval()
            total_val_loss = 0
            for (x, y, z) in val_loader:
                y *= 255
                z *= 255
                (x, y, z) = (x.to(DEVICE), y.to(DEVICE), z.to(DEVICE))
                pred_orient, pred_slope = unet(x)

                y = y.long().squeeze()
                z = z.long().squeeze()
                loss_seg = loss_func(pred_orient, y)
                loss_slope = loss_func(pred_slope, z)
                # loss = awl(loss_seg, loss_slope * 1.0)  # Adjust the influence of slope loss
                loss = loss_seg+loss_slope
                total_val_loss += loss.item()
                total_val_seg_loss += loss_seg.item()
                total_val_slope_loss += loss_slope.item()

                pred_orient = torch.argmax(pred_orient, dim=1)
                pred_slope = torch.argmax(pred_slope, dim=1)
                total_val_orient += y.size().numel()
                total_val_slope += z.size().numel()
                total_val_correct_orient += (pred_orient == y).sum().item()
                total_val_correct_slope += (pred_slope == z).sum().item()
                cm_orient += confmat_orient(pred_orient, y)
                cm_slope += confmat_slope(pred_slope, z)

            avg_val_seg_loss = total_val_seg_loss / val_steps
            avg_val_slope_loss = total_val_slope_loss / val_steps
            val_iou_orient = intersection_over_union(cm_orient.cpu().detach().numpy())
            val_iou_slope = intersection_over_union(cm_slope.cpu().detach().numpy())
            mean_val_iou_orient = np.mean(val_iou_orient[val_iou_orient != 0])
            mean_val_iou_slope = np.mean(val_iou_slope[val_iou_slope != 0])
            print('Val Orient Class IOU', val_iou_orient)
            print('Val Slope Class IOU', val_iou_slope)
            print('Val Mean Orient Class IOU:{}'.format(mean_val_iou_orient))
            print('Val Mean Slope Class IOU:{}'.format(mean_val_iou_slope))
            print('Val Avg Segmentation Loss: {:.4f}'.format(avg_val_seg_loss))
            print('Val Avg Slope Loss: {:.4f}'.format(avg_val_slope_loss))
            # print('AWL Params:', awl.params.detach().cpu().numpy())

            combined_val_score = mean_val_iou_orient + mean_val_iou_slope
            if combined_val_score > best_val_score:
                print('Better Model Found: {} > {}'.format(combined_val_score, best_val_score))
                best_val_score = combined_val_score
                torch.save(unet, model_output_path / model_name)

        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps
        avg_train_acc_orient = total_correct_orient / total_orient
        avg_train_acc_slope = total_correct_slope / total_slope
        avg_val_acc_orient = total_val_correct_orient / total_val_orient
        avg_val_acc_slope = total_val_correct_slope / total_val_slope

        H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        H["val_loss"].append(avg_val_loss)
        H["train_acc_orient"].append(avg_train_acc_orient)
        H["val_acc_orient"].append(avg_val_acc_orient)
        H["train_acc_slope"].append(avg_train_acc_slope)
        H["val_acc_slope"].append(avg_val_acc_slope)
        print("[INFO] EPOCH: {}/{}".format(e + 1, n_epochs))
        print("Train loss: {:.6f}, Val loss: {:.4f} Train Acc Orient: {:.6f}, Val Acc Orient: {:.4f}, Train Acc Slope: {:.6f}, Val Acc Slope: {:.4f}".format(
            avg_train_loss, avg_val_loss, avg_train_acc_orient, avg_val_acc_orient, avg_train_acc_slope, avg_val_acc_slope))
        scheduler_seg.step(avg_val_loss)
        scheduler_slope.step(avg_val_loss)

    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        end_time - start_time))

    torch.save(unet, model_output_path / 'model_last.pth')
    writer.flush()
    print('Best Val Score:{}'.format(best_val_score))




