import torch
import pathlib
import cv2
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from passion.segmentation import prediction

device = ('cuda' if torch.cuda.is_available() else 'cpu')

models_folder_path = pathlib.Path('/scratch/clear/aboccala/PASSION/workflow/output/model')
section_models_folder_path = models_folder_path / 'section-segmentation'

# Sections models
model_test = torch.load(str(section_models_folder_path / 'new_152_outnoint_all_attention_drop_resnet152_dp03.pth'), map_location=torch.device(device))

rid_test_folder = pathlib.Path('/scratch/clear/aboccala/training/RID/output_noint_all/masks_segments/test')
rid_test_folder_image = rid_test_folder / 'image'
rid_test_folder_label = rid_test_folder / 'label'
rid_test_folder_slope = rid_test_folder / 'slope'

# rid_test_folder = pathlib.Path('/scratch/clear/aboccala/training/dataset_vaud')
# rid_test_folder_image = rid_test_folder / 'images'
# rid_test_folder_label = rid_test_folder / 'labels'

# import torch
# import numpy as np
# import cv2
# import pathlib
# from torchvision import transforms
# from PIL import Image

# # Funzione per preparare l'immagine
# def prepare_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),  # Assumi che il modello richieda 512x512
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return transform(image).unsqueeze(0)  # Aggiunge una dimensione batch

# # Funzione per prevedere la slope
# def predict_slope(model, image_tensor):
#     model.eval()
#     with torch.no_grad():
#         _, pred_slope = model(image_tensor)  # Assumi che la slope sia il secondo output
#         pred_slope = pred_slope.squeeze().detach().cpu().numpy()
#     return pred_slope

# # Funzione per calcolare MAE e RMSE
# def calculate_errors(pred_slope, true_slope):
#     mask = true_slope != -1  # Assumi -1 come valore di ignorare
#     pred_slope = pred_slope[mask]
#     true_slope = true_slope[mask]
#     mae = np.mean(np.abs(pred_slope - true_slope))
#     rmse = np.sqrt(np.mean((pred_slope - true_slope) ** 2))
#     return mae, rmse

# # Funzione principale per valutare le slopes
# def evaluate_slopes(model, test_folder_path):
#     test_folder = pathlib.Path(test_folder_path)
#     image_paths = list(test_folder.glob('*.png'))
#     slope_paths = [test_folder / 'slope' / p.name.replace('image', 'slope') for p in image_paths]

#     total_mae = []
#     total_rmse = []

#     for img_path, slope_path in zip(image_paths, slope_paths):
#         image_tensor = prepare_image(img_path)
#         true_slope = cv2.imread(str(slope_path), cv2.IMREAD_GRAYSCALE)

#         pred_slope = predict_slope(model, image_tensor)
#         mae, rmse = calculate_errors(pred_slope, true_slope)

#         total_mae.append(mae)
#         total_rmse.append(rmse)
#         print(f'Processed {img_path.name}: MAE={mae:.2f}, RMSE={rmse:.2f}')

#     average_mae = np.mean(total_mae)
#     average_rmse = np.mean(total_rmse)
#     print(f'Average MAE: {average_mae:.2f}, Average RMSE: {average_rmse:.2f}')

#     return average_mae, average_rmse

# # Carica il modello
# model = torch.load(section_models_folder_path / 'new_outnoint_all_slopes152_drop0303.pth', map_location='cpu')

# # Valuta le slopes
# evaluate_slopes(model,'/scratch/clear/aboccala/training/RID/output_noint_all/masks_segments/test')

def intersect_and_union(pred_label, label, num_classes, ignore_index):
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label

def angle_difference(angle_1, angle_2):
    a = angle_1 - angle_2
    a = (a + 180) % 360 - 180
    return a

def mean_angle_difference(pred_label, label, num_classes, angles, background_class, ignore_index):
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]
    
    combined_pred_label = (pred_label != background_class).astype('uint8')
    combined_label = (label != background_class).astype('uint8')
    
    # Take those pixels where both images predict a class different than background
    pred_label_angles = pred_label[combined_pred_label == combined_label]
    pred_label_angles = pred_label_angles[pred_label_angles < len(angles)]
    label_angles = label[combined_pred_label == combined_label]
    label_angles = label_angles[label_angles < len(angles)]

    total_diff = 0
    for angle_1, angle_2 in zip(np.nditer(pred_label_angles, flags=['zerosize_ok']), np.nditer(label_angles, flags=['zerosize_ok'])):
        try:
            angle_1, angle_2 = angles[angle_1], angles[angle_2]
        except:
            print(angle_1, angle_2)
        diff = angle_difference(angle_1, angle_2)
        total_diff = total_diff + diff
    
    if pred_label_angles.size == 0:
        return None
    
    mean_diff = total_diff/pred_label_angles.size
    
    return mean_diff

def calculate_iou_per_class(area_intersect, area_union, num_classes):
    iou_per_class = np.zeros(num_classes)
    for cls in range(num_classes):
        if area_union[cls] > 0:
            iou_per_class[cls] = area_intersect[cls] / area_union[cls]
        else:
            iou_per_class[cls] = np.nan  # Usa NaN per classi senza intersezione/unione
    return iou_per_class

def adjust_predictions(pred):
    # Mappa le classi di orientamento e flat roof
    pred_adj = np.where((pred >= 1) & (pred <= 16), pred - 1, pred)
    # Mappa la classe di background
    pred_adj = np.where(pred == 0, 17, pred_adj)
    # Mappa la classe di flat roof
    pred_adj = np.where(pred == 17, 16, pred_adj)
    return pred_adj

def test_model_sections(model, test_folder, num_classes, background_class, ignore_index, num_angles, output=True):
    start = time.time()
    
    test_folder_image = test_folder / 'image'
    test_folder_label = test_folder / 'label'
    # test_folder_slope = test_folder / 'slope'
    
    angles = [i * (360/num_angles) for i in range(num_angles)]
    
    total_angle_difference = 0

    total_valid_iou = []  # Accumulatore per gli IoU validi di tutto il dataset
    total_angle_difference = []
    
    for i, filename in enumerate(test_folder_image.glob('*.png')):
        image = cv2.imread(str(filename))
        label = cv2.imread(str(test_folder_label / filename.name), cv2.IMREAD_GRAYSCALE)
        # slope = cv2.imread(str(test_folder_slope / filename.name), cv2.IMREAD_GRAYSCALE)
        pred = prediction.segment_img(image, model, tile_size=256, stride=256, background_class=background_class)
        corrected_pred = pred.copy()
        # corrected_pred = adjust_predictions(pred)
        # print(label[0])
        # print(corrected_pred [0])
        # exit()
        # Calcola IoU utilizzando la funzione intersect_and_union modificata
        area_intersect, area_union, _, _ = intersect_and_union(corrected_pred, label, num_classes, ignore_index)
        iou_per_class = calculate_iou_per_class(area_intersect, area_union, num_classes)
        valid_iou = iou_per_class[~np.isnan(iou_per_class)]  # Filtra NaN
        if valid_iou.size > 0:
            total_valid_iou.extend(valid_iou)
        mean_iou = np.nanmean(iou_per_class)
        # area_intersect, area_union, _, _ = intersect_and_union(corrected_pred, label, num_classes, ignore_index)
        
        # iou = area_intersect / (area_union + 1e-10)
        # valid_iou = iou[area_union > 0]
        # if valid_iou is not None:
        #     total_valid_iou.extend(valid_iou)  # Aggiungi i valori IoU validi alla lista per il dataset
        
        # mean_iou = np.mean(valid_iou) if valid_iou.size > 0 else 0
        
        mad = mean_angle_difference(corrected_pred, label, num_classes, angles, background_class, ignore_index)
        if mad is not None:
            total_angle_difference.append(mad)


        if (i % 50==0) and output:
            f, axarr = plt.subplots(1,3, figsize=(10, 10))
            axarr[0].imshow(image)
            axarr[1].imshow(label)
            axarr[2].imshow(corrected_pred)
            axarr[0].title.set_text('Image')
            axarr[1].title.set_text('Ground truth')
            axarr[2].title.set_text(f'mean IoU: {mean_iou:.4f}')

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            for ax in axarr:
                ax.axis('off')
            plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/output_{filename.stem}_1.png', bbox_inches='tight')
            plt.close(f)  # Chiude la figura per liberare memoria
        if output:
            print(f'Processed image {filename.stem}, mean IoU: {mean_iou:.4f}, mean angle difference: {mad if mad is not None else "N/A"}')
        for cls in range(num_classes):
            if area_union[cls] > 0:  # Controlla che la classe sia presente nell'immagine
                iou_message = f'NaN' if np.isnan(iou_per_class[cls]) else f'{iou_per_class[cls]:.4f}'
                print(f'    Class {cls}: IoU: {iou_message}')
    

    # Calcola la media degli IoU validi per tutto il dataset
    dataset_mean_iou = np.mean(total_valid_iou) if total_valid_iou else 0
    dataset_mean_angle_difference = np.mean(total_angle_difference) if total_angle_difference else 0

    end = time.time()
    elapsed_time = end - start
    
    if output:
        # Stampa i risultati aggregati per tutto il dataset
        print('\nTotal processing time:', elapsed_time, 'seconds')
        print(f'Dataset Mean IoU: {dataset_mean_iou:.4f}')
        print(f'Dataset Mean Angle Difference: {dataset_mean_angle_difference:.4f}')

    return dataset_mean_iou, dataset_mean_angle_difference, elapsed_time

# test_folder = pathlib.Path('/scratch/clear/aboccala/training/dataset_vaud')

# def test_model_sections(model, test_folder, num_classes, background_class, ignore_index, num_angles, output=True):
#     start = time.time()
    
#     test_folder_image = test_folder / 'images'
#     test_folder_label = test_folder / 'labels'

#     angles = [i * (360/num_angles) for i in range(num_angles)]

#     total_area_intersect = np.zeros((num_classes, ), dtype=float)
#     total_area_union = np.zeros((num_classes, ), dtype=float)
#     total_area_pred_label = np.zeros((num_classes, ), dtype=float)
#     total_area_label = np.zeros((num_classes, ), dtype=float)
#     total_angle_difference = 0

#     file_counter = 0  # Contatore per tracciare il numero di file processati

#     for filename in test_folder_image.glob('*.tif'):
#         image = cv2.imread(str(filename))
#         label = cv2.imread(str(test_folder_label / filename.name), cv2.IMREAD_GRAYSCALE)

#         pred = prediction.segment_img(image, model, tile_size=256, stride=256, background_class=background_class)
#         corrected_pred = pred.copy()

#         area_intersect, area_union, area_pred_label, area_label = \
#                             intersect_and_union(corrected_pred, label, num_classes, ignore_index)
#         total_area_intersect += area_intersect
#         total_area_union += area_union
#         total_area_pred_label += area_pred_label
#         total_area_label += area_label

#         mad = mean_angle_difference(corrected_pred, label, num_classes, angles, background_class, ignore_index)
#         if mad is not None:
#             total_angle_difference += mad
        
#         # Calcola l'IoU per l'immagine corrente
#         iou_per_class = area_intersect / (area_union + 1e-10)  # Aggiungi un piccolo epsilon per evitare la divisione per zero
#         valid_iou = iou_per_class[area_union > 0]  # Filtra per evitare la divisione per zero
#         mean_iou = np.nanmean(valid_iou)  # Calcola la media dell'IoU per le classi valide

#         iou = total_area_intersect / total_area_union
#         # Salva l'immagine solo se l'indice corrente è un multiplo di 50
#         if file_counter % 100 == 0 and output:
#         # if output:
#             f, axarr = plt.subplots(1,3, figsize=(10, 10))
#             axarr[0].imshow(image)
#             axarr[1].imshow(label)
#             axarr[2].imshow(corrected_pred)
#             axarr[2].title.set_text(f'mean IoU: {mean_iou:.4f}')

#             plt.subplots_adjust(wspace=0.1, hspace=0.1)
#             for ax in axarr:
#                 ax.axis('off')
#             plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/output_{filename.stem}.png', bbox_inches='tight')
#             plt.close(f) 

#         if output:
#             print(f'Processed image {filename.stem}, partial mean iou: {np.mean(iou[:-1][~np.isnan(iou[:-1])])}, with background class: {np.mean(iou[~np.isnan(iou)])}, mean angle difference: {mad}')

#         file_counter += 1  # Incrementa il contatore di file processati

#     all_acc = total_area_intersect.sum() / total_area_label.sum()
#     acc = total_area_intersect / total_area_label
#     iou = total_area_intersect / total_area_union
#     mad = total_angle_difference / file_counter

#     end = time.time()
#     elapsed_time = (end - start)

#     if output:
#         print('\n')
#         print(f'Elapsed time: {elapsed_time} seconds')
#         print(f'Final IoU per class: {iou}')
#         print(f'Final mean IoU with background class: {np.mean(iou[~np.isnan(iou)])}')
#         print(f'Final mean IoU without background class: {np.mean(iou[:-1][~np.isnan(iou[:-1])])}')
#         print(f'Final mean angle difference: {mad}')

#     return iou, np.mean(iou[~np.isnan(iou)]), mad, elapsed_time

# Rimuovi la limitazione sul numero di campioni nel chiamante della funzione

num_classes = 18
background_class = 0
ignore_index = -1
num_angles = 16

print(f'Testing model <model_test17>...')

# Chiamata alla funzione test_model_sections con i parametri aggiornati
dataset_mean_iou, dataset_mean_angle_difference, elapsed_time = test_model_sections(
    model_test, 
    rid_test_folder, 
    num_classes, 
    background_class, 
    ignore_index, 
    num_angles, 
    output=True
)

# Stampa i risultati finali per tutto il dataset
print(f'Dataset Mean IoU: {dataset_mean_iou:.4f}')
print(f'Dataset Mean Angle Difference: {dataset_mean_angle_difference:.4f} degrees')
print(f'Elapsed Time: {elapsed_time:.2f} seconds')