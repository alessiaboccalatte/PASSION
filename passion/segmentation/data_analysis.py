import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import glob

def calculate_co_occurrence_matrix(label_paths, num_classes):
    co_occurrence_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for label_path in label_paths:
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        unique_classes = np.unique(mask)
        for cls1 in unique_classes:
            for cls2 in unique_classes:
                co_occurrence_matrix[cls1, cls2] += 1
    return co_occurrence_matrix

def plot_co_occurrence_matrix(co_occurrence_matrix, class_names, save_path="/scratch/clear/aboccala/PASSION/passion/segmentation/data_analysis_res"):
    try:
        plt.figure(figsize=(12, 10))
        plt.imshow(co_occurrence_matrix, cmap='Blues')
        plt.colorbar()
        plt.xlabel('Class ID')
        plt.ylabel('Class ID')
        plt.title('Co-Occurrence Matrix')
        plt.xticks(range(num_classes), class_names, rotation=45)
        plt.yticks(range(num_classes), class_names)
        plt.savefig(f'{save_path}/co_occurrence_matrix.png', bbox_inches='tight')
        plt.close()
        print(f"Co-occurrence matrix saved to {save_path}/co_occurrence_matrix.png")
    except Exception as e:
        print(f"An error occurred while plotting/saving the co-occurrence matrix: {e}")

def calculate_class_frequencies(label_paths):
    class_frequencies = Counter()
    object_sizes = []
    objects_per_image = []
    
    for label_path in label_paths:
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        unique, counts = np.unique(mask, return_counts=True)
        class_frequencies.update(unique.tolist())
        
        # Escludere il background
        mask_objects = mask[mask != 0]
        unique_objects, counts_objects = np.unique(mask_objects, return_counts=True)
        objects_per_image.append(len(unique_objects))
        object_sizes.extend(counts_objects.tolist())
    
    # Rimuovere il background dalle frequenze
    if 0 in class_frequencies:
        del class_frequencies[0]
        
    return class_frequencies, object_sizes, objects_per_image

def plot_class_frequencies(class_frequencies):
    classes = sorted(class_frequencies.keys())
    frequencies = [class_frequencies[cls] for cls in classes]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(classes)), frequencies, color='skyblue')
    plt.xlabel('Class ID')
    plt.ylabel('Frequency')
    plt.title('Class Frequencies in Dataset')
    plt.xticks(range(len(classes)), classes)
    plt.savefig(f'/scratch/clear/aboccala/PASSION/passion/segmentation/data_analysis_res/0.png', bbox_inches='tight')
    plt.close()




def plot_objects_size_distribution(object_sizes):
    plt.figure(figsize=(10, 6))
    plt.hist(object_sizes, bins=20, color='skyblue')
    plt.xlabel('Object Size (pixels)')
    plt.ylabel('Frequency')
    plt.title('Object Size Distribution')
    plt.savefig(f'/scratch/clear/aboccala/PASSION/passion/segmentation/data_analysis_res/00.png', bbox_inches='tight')
    plt.close()

def plot_objects_per_image_distribution(objects_per_image):
    plt.figure(figsize=(10, 6))
    plt.hist(objects_per_image, bins=max(objects_per_image), color='skyblue')
    plt.xlabel('Objects per Image')
    plt.ylabel('Frequency')
    plt.title('Distribution of Objects per Image')
    plt.savefig(f'/scratch/clear/aboccala/PASSION/passion/segmentation/data_analysis_res/000.png', bbox_inches='tight')
    plt.close()

# Percorso alla cartella delle maschere di training
label_dir = "/scratch/clear/aboccala/training/RID/output_noint/masks_segments/test/label"
label_paths = glob.glob(label_dir + '/*.png')

# Calcola la distribuzione delle classi
class_frequencies, object_sizes, objects_per_image = calculate_class_frequencies(label_paths)

# Assumi che il numero di classi sia il massimo ID di classe trovato + 1 (se includi 0 come background)
num_classes = max(class_frequencies.keys()) + 1
class_names = [str(i) for i in range(num_classes)]  # o una lista dei nomi delle classi se disponibile

# Calcola e visualizza la matrice di co-occorrenza
co_occurrence_matrix = calculate_co_occurrence_matrix(label_paths, num_classes)
plot_co_occurrence_matrix(co_occurrence_matrix, class_names)

# Visualizza le altre statistiche calcolate
print("Frequenze delle classi (escluso il background):", class_frequencies)
plot_class_frequencies(class_frequencies)
plot_objects_size_distribution(object_sizes)
plot_objects_per_image_distribution(objects_per_image)



    


