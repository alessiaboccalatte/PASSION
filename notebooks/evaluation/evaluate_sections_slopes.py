# import torch
# import pathlib
# import cv2
# import PIL
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import time
# from tqdm import tqdm
# import pvlib
# from pathlib import Path
# import matplotlib.colors as mcolors
# import geopandas as gpd
# import numpy as np

# from passion.segmentation import prediction_slopes

# device = ('cuda' if torch.cuda.is_available() else 'cpu')

# models_folder_path = pathlib.Path('/scratch/clear/aboccala/PASSION/workflow/output/model')
# section_models_folder_path = models_folder_path / 'section-segmentation'

# # Sections models
# model_test = torch.load(str(section_models_folder_path / 'test_seg_ge_va.pth'), map_location=torch.device(device))

# rid_test_folder = pathlib.Path('/scratch/clear/aboccala/training/RID/output_mix_ge_va/test_va')
# rid_test_folder_image = rid_test_folder / 'image'
# rid_test_folder_label = rid_test_folder / 'label'
# rid_test_folder_slope = rid_test_folder / 'slope'

# def intersect_and_union(pred_label, label, num_classes, ignore_index):
#     mask = (label != ignore_index)
#     pred_label = pred_label[mask]
#     label = label[mask]

#     intersect = pred_label[pred_label == label]
#     area_intersect, _ = np.histogram(
#         intersect, bins=np.arange(num_classes + 1))
#     area_pred_label, _ = np.histogram(
#         pred_label, bins=np.arange(num_classes + 1))
#     area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
#     area_union = area_pred_label + area_label - area_intersect

#     return area_intersect, area_union, area_pred_label, area_label

# def angle_difference(angle_1, angle_2):
#     a = angle_1 - angle_2
#     a = (a + 180) % 360 - 180
#     return a

# def mean_angle_difference(pred_label, label, num_classes, angles, background_class, ignore_index):
#     mask = (label != ignore_index)
#     pred_label = pred_label[mask]
#     label = label[mask]
    
#     combined_pred_label = (pred_label != background_class).astype('uint8')
#     combined_label = (label != background_class).astype('uint8')
    
#     # Take those pixels where both images predict a class different than background
#     pred_label_angles = pred_label[combined_pred_label == combined_label]
#     pred_label_angles = pred_label_angles[pred_label_angles < len(angles)]
#     label_angles = label[combined_pred_label == combined_label]
#     label_angles = label_angles[label_angles < len(angles)]

#     total_diff = 0
#     for angle_1, angle_2 in zip(np.nditer(pred_label_angles, flags=['zerosize_ok']), np.nditer(label_angles, flags=['zerosize_ok'])):
#         try:
#             angle_1, angle_2 = angles[angle_1], angles[angle_2]
#         except:
#             print(angle_1, angle_2)
#         diff = angle_difference(angle_1, angle_2)
#         total_diff = total_diff + diff
    
#     if pred_label_angles.size == 0:
#         return None
    
#     mean_diff = total_diff/pred_label_angles.size
    
#     return mean_diff

# def calculate_iou_per_class(area_intersect, area_union, num_classes):
#     iou_per_class = np.zeros(num_classes)
#     for cls in range(num_classes):
#         if area_union[cls] > 0:
#             iou_per_class[cls] = area_intersect[cls] / area_union[cls]
#         else:
#             iou_per_class[cls] = np.nan  # Usa NaN per classi senza intersezione/unione
#     return iou_per_class

# def replace_slope_values_with_median(segmentation, slope, background_class=0, method='median'):
#     """
#     Replace slope predictions with the chosen statistic (median, mean, mode) from the predicted regions, excluding background.
#     """
#     from scipy import stats
#     modified_slope_image = np.zeros_like(slope, dtype=float)
#     unique_classes = np.unique(segmentation)
    
#     for cls in unique_classes:
#         if cls == background_class:
#             mask = segmentation == cls
#             modified_slope_image[mask] = np.nan  # Explicitly set background slope to zero
#         elif cls != background_class:
#             mask = segmentation == cls
#             slopes = slope[mask]
#             if slopes.size > 0:
#                 if method == 'median':
#                     slope_value = np.median(slopes)
#                 elif method == 'mean':
#                     slope_value = np.mean(slopes)
#                 elif method == 'mode':
#                     slope_value = stats.mode(slopes, nan_policy='omit').mode[0]
#                 modified_slope_image[mask] = slope_value
#             else:
#                 modified_slope_image[mask] = np.nan  # Mark areas with no data explicitly

#     return modified_slope_image

# def calculate_mae_rmse(pred, true):
#     """ Calculate Mean Absolute Error and Root Mean Square Error. """
#     pred = pred.flatten()
#     true = true.flatten()
#     mae = np.mean(np.abs(pred - true))
#     rmse = np.sqrt(np.mean((pred - true) ** 2))
#     return mae, rmse

# import requests

# def get_pvgis_tmy(latitude, longitude, outputformat='json', usehorizon=True,
#                   userhorizon=None, startyear=None, endyear=None,
#                   raddatabase='PVGIS-SARAH2', meteodb='ERA-Interim',
#                   map_variables=True, url='https://re.jrc.ec.europa.eu/api/v5_2/', timeout=120):
#     params = {
#         'lat': latitude,
#         'lon': longitude,
#         'outputformat': outputformat,
#         'usehorizon': int(usehorizon),
#         'userhorizon': ','.join(map(str, userhorizon)) if userhorizon else None,
#         'startyear': startyear,
#         'endyear': endyear,
#         'raddatabase': raddatabase,
#         'meteodb': meteodb
#     }
#     response = requests.get(url + 'tmy', params=params, timeout=timeout)
#     if response.status_code != 200:
#         print("Response Text:", response.text)  # Stampa il testo della risposta in caso di errore
#         response.raise_for_status()

#     if outputformat == 'json':
#         data = response.json()
#         data_frame = pd.DataFrame(data['outputs']['tmy_hourly'])
#         if map_variables:
#             data_frame.rename(columns={
#                 'G(h)': 'ghi',
#                 'Gb(n)': 'dni',
#                 'Gd(h)': 'dhi'
#             }, inplace=True)
#         return data_frame, data['inputs'], data['meta']
#     else:
#         raise ValueError("Unsupported format")
    
# def slope_class_to_value(slope_class):
#     slope_mapping = {
#         1: 0,    # flat
#         2: 5,    # 0 < slope < 10
#         3: 15,   # 10 <= slope < 20
#         4: 25,   # 20 <= slope < 30
#         5: 35,   # 30 <= slope < 40
#         6: 45,   # 40 <= slope < 50
#         7: 55    # slope >= 50
#     }
#     return slope_mapping.get(slope_class, 0)  # Default to 0 if class not found

# def calculate_irradiance(segmentation, slope, angles, tmy_data, location, num_classes, background_class):
#     irradiance_map = np.zeros_like(segmentation, dtype=np.float32)

#     for cls in range(1, num_classes):
#         mask = segmentation == cls
#         if not np.any(mask):
#             continue

#         # Convert the slope class to a real value
#         slope_values = np.vectorize(slope_class_to_value)(slope[mask])
#         tilt = np.median(slope_values)

#         azimuth = 180 if cls == 17 else angles[cls % len(angles)]

#         solar_position = location.get_solarposition(times=tmy_data.index)
#         irradiance_components = pvlib.irradiance.get_total_irradiance(
#             surface_tilt=tilt,
#             surface_azimuth=azimuth,
#             dni=tmy_data['dni'],
#             ghi=tmy_data['ghi'],
#             dhi=tmy_data['dhi'],
#             dni_extra=pvlib.irradiance.get_extra_radiation(tmy_data.index),
#             solar_zenith=solar_position['apparent_zenith'],
#             solar_azimuth=solar_position['azimuth'],
#             model='perez', albedo=0.20
#         )
#         irradiance_map[mask] = irradiance_components['poa_global'].sum() / 1000

#     return irradiance_map


# import numpy as np
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np

# import matplotlib.pyplot as plt
# import numpy as np

# def visualize_comparisons(image, label, pred, slope, pred2, filename):
#     """
#     Visualize the comparison of segmentation and slope predictions against their ground truths.
    
#     Parameters:
#     - image: The original image.
#     - label: Ground truth for segmentation.
#     - pred: Predicted segmentation.
#     - slope: Ground truth for slope.
#     - pred2: Predicted slope after corrections.
#     - filename: Name of the file for saving the output images.
#     """
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12), gridspec_kw={'width_ratios': [1, 1, 1.2]})  # Adjust the width ratio for the last column to fit colorbar

#     # Original Image
#     axes[0, 0].imshow(image)
#     axes[0, 0].set_title('Original Image')
#     axes[0, 0].axis('off')

#     # Ground Truth Segmentation
#     im1 = axes[0, 1].imshow(label, cmap='viridis')
#     axes[0, 1].set_title('Ground Truth Segmentation')
#     axes[0, 1].axis('off')
#     fig.colorbar(im1, ax=axes[0, 1], orientation='vertical', fraction=0.046, pad=0.04)

#     # Predicted Segmentation
#     im2 = axes[0, 2].imshow(pred, cmap='viridis')
#     axes[0, 2].set_title('Predicted Segmentation')
#     axes[0, 2].axis('off')
#     fig.colorbar(im2, ax=axes[0, 2], orientation='vertical', fraction=0.046, pad=0.04)

#     # Ground Truth Slope
#     # norm = plt.Normalize(vmin=0, vmax=100)  # Normalize both actual and predicted slope between 0 and 100
#     im3 = axes[1, 0].imshow(slope, cmap='inferno')
#     axes[1, 0].set_title('Ground Truth Slope')
#     axes[1, 0].axis('off')
#     fig.colorbar(im3, ax=axes[1, 0], orientation='vertical', fraction=0.046, pad=0.04)

#     # Predicted Slope
#     im4 = axes[1, 1].imshow(pred2, cmap='inferno')
#     axes[1, 1].set_title('Predicted Slope')
#     axes[1, 1].axis('off')
#     fig.colorbar(im4, ax=axes[1, 1], orientation='vertical', fraction=0.046, pad=0.04)

#     # Difference in Slope
#     diff = np.abs(slope.astype(float) - pred2.astype(float))
#     im5 = axes[1, 2].imshow(diff, cmap='coolwarm', vmin=0, vmax=50)  # Assuming difference ranges reasonably within 0 to 50
#     axes[1, 2].set_title('Difference in Slope')
#     axes[1, 2].axis('off')
#     fig.colorbar(im5, ax=axes[1, 2], orientation='vertical', fraction=0.046, pad=0.04)

#     plt.tight_layout()
#     plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/{filename}.png', bbox_inches='tight')
#     plt.close(fig)  # Close the plot to free up memory


# def compare_irradiance(b_id, image, pred_irradiance, filename, gdf, img_size_m, results_list):
#     plt.figure(figsize=(20, 6))

#     ax0 = plt.subplot(1, 3, 1)
#     plt.imshow(image)
#     plt.title('Original Image')
#     ax0.set_xticks([])
#     ax0.set_yticks([])

#     pixel_area = 0.04  # 0.2 meters per pixel, squared for area

#     ax1 = plt.subplot(1, 3, 2)
#     norm = mcolors.Normalize(vmin=0, vmax=1700)
#     im1=plt.imshow(pred_irradiance, cmap='YlOrRd', norm=norm)
#     plt.colorbar(ax=ax1)
#     total_pred_weighted_irradiance = np.sum(pred_irradiance[pred_irradiance > 0] * pixel_area)
#     total_pred_area = np.count_nonzero(pred_irradiance > 0) * pixel_area
#     mean_pred_irradiance = total_pred_weighted_irradiance / total_pred_area if total_pred_area > 0 else 0
#     plt.title(f'Predicted Irradiance - Mean: {mean_pred_irradiance:.2f}\nTotal Predicted Area: {total_pred_area:.2f} sq m')
#     ax1.set_xticks([])
#     ax1.set_yticks([])

#     building = gdf[gdf['b_id'] == int(b_id)]
#     if not building.empty:
#         centroid = building.geometry.centroid.iloc[0]
#         buffer = centroid.buffer(img_size_m / 2, cap_style=3)

#         sg_data = gdf[gdf.geometry.intersects(buffer)]
#         sg_data = sg_data.copy()
#         sg_data['intersected_geometry'] = sg_data.geometry.intersection(buffer)
#         sg_data['area'] = sg_data['intersected_geometry'].area

#         sg_data['weighted_irr'] = sg_data['MSTRAHLUNG'] * sg_data['area']
#         total_weighted_irradiance = sg_data['weighted_irr'].sum()
#         total_area = sg_data['area'].sum()
#         mean_real_irradiance = total_weighted_irradiance / total_area if total_area > 0 else 0

#         ax2 = plt.subplot(1, 3, 3)
#         ax2.set_xlim([buffer.bounds[0], buffer.bounds[2]])
#         ax2.set_ylim([buffer.bounds[1], buffer.bounds[3]])
#         norm = mcolors.Normalize(vmin=0, vmax=1700)
#         im2=sg_data.plot(ax=ax2, column='MSTRAHLUNG', cmap='YlOrRd', legend=True, norm=norm)
#         # plt.colorbar(ax=ax2)  
#         plt.title(f'Real Irradiance - Mean: {mean_real_irradiance:.2f}\nTotal Real Area: {total_area:.2f} sq m')
#         ax2.set_xticks([])
#         ax2.set_yticks([])

#     percent_difference = ((mean_real_irradiance - mean_pred_irradiance) / mean_pred_irradiance) * 100 if mean_pred_irradiance > 0 else float('inf')
#     results_list.append(percent_difference)

#     plt.suptitle(f'Irradiance Difference: {percent_difference:.2f}%')
#     plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/{filename.stem}_1.png', bbox_inches='tight')
#     plt.close()

# import dask_geopandas as dg

# def test_model_sections(model, test_folder, num_classes, background_class, ignore_index, num_angles, output=True):
#     test_folder_image = test_folder / 'image'
#     test_folder_label = test_folder / 'label'
#     test_folder_slope = test_folder / 'slope'

#     shapefile_path = r'/scratch/clear/aboccala/PASSION/notebooks/evaluation/vaud_final.shp'
#     test_b_ids = {f.stem for f in Path(test_folder_image).glob('*.png')}

#     img_size_px = 256
#     img_res_m_per_px = 0.2
#     img_size_m = img_size_px * img_res_m_per_px

#     gdf = dg.read_file(shapefile_path, npartitions=10)
#     gdf_total = gdf.compute()
    
#     try:
#         test_b_ids_int = {int(id) for id in test_b_ids}
#         print("Conversione riuscita. ID convertiti in interi.")
#     except ValueError:
#         print("Errore nella conversione. Verifica che tutti gli ID siano numerici.")

#     if 'test_b_ids_int' in locals():
#         gdf_filtered = gdf[gdf['b_id'].isin(test_b_ids_int)]
#         gdf = gdf_filtered.compute()
#         if gdf.empty:
#             print("Nessun dato corrispondente trovato. Verifica i valori degli ID.")
#         else:
#             print("Dati filtrati con successo.")

#     angles = np.array([i * (360 / num_angles) for i in range(num_angles)])

#     total_valid_iou = []
#     total_angle_difference = []
#     total_slope_mae = []
#     total_slope_rmse = []
#     results_list = []

#     for filename in tqdm(list(test_folder_image.glob('*.png'))):
#         image = cv2.imread(str(filename))
#         label = cv2.imread(str(test_folder_label / filename.name), cv2.IMREAD_GRAYSCALE)
#         slope = cv2.imread(str(test_folder_slope / filename.name), cv2.IMREAD_GRAYSCALE)

#         pred, pred2 = prediction_slopes.segment_img(image, model, tile_size=256, stride=256, background_class=background_class)
#         pred2 = replace_slope_values_with_median(pred, pred2, background_class)

#         b_id = filename.stem
#         building = gdf[gdf['b_id'] == int(b_id)]
#         building_projected = building.to_crs('EPSG:2056') 
#         if not building_projected.empty:
#             centroid_projected = building_projected.geometry.centroid.iloc[0]
#             # Convert the centroid back to geographic CRS by transforming the entire GeoSeries for simplicity
#             # Create a new GeoDataFrame for the single centroid for proper CRS transformation
#             centroid_geo = gpd.GeoDataFrame(geometry=gpd.GeoSeries(centroid_projected), crs=building_projected.crs)
#             centroid_geo = centroid_geo.to_crs(epsg=4326)
#             # Extract the first (and only) centroid point
#             centroid = centroid_geo.geometry.iloc[0]

#             latitude = centroid.y
#             longitude = centroid.x


#             location = pvlib.location.Location(latitude, longitude)
#             tmy_data, _, _ = get_pvgis_tmy(latitude, longitude)
#             tmy_data['time(UTC)'] = pd.to_datetime(tmy_data['time(UTC)'], format='%Y%m%d:%H%M', utc=True)
#             tmy_data.set_index('time(UTC)', inplace=True)

#         # Calcola IoU utilizzando la funzione intersect_and_union modificata
#         area_intersect, area_union, _, _ = intersect_and_union(pred, label, num_classes, ignore_index)
#         iou_per_class = calculate_iou_per_class(area_intersect, area_union, num_classes)
#         valid_iou = iou_per_class[~np.isnan(iou_per_class)]  # Filtra NaN
#         if valid_iou.size > 0:
#             total_valid_iou.extend(valid_iou)
#         mean_iou = np.nanmean(iou_per_class)

#         mad = mean_angle_difference(pred, label, num_classes, angles, background_class, ignore_index)
#         if mad is not None:
#             total_angle_difference.append(mad)

#         mae, rmse = calculate_mae_rmse(pred2, slope)
#         total_slope_mae.append(mae)
#         total_slope_rmse.append(rmse)

#         # Qui chiamiamo la funzione calculate_irradiance
#         irradiance_map = calculate_irradiance(pred, pred2, angles, tmy_data, location, num_classes, background_class)
#         irradiance_map[pred == 0] = np.nan
#         # Confronto delle mappe di irraggiamento
#         visualize_comparisons(image, label, pred, slope, pred2, filename.stem)
#         compare_irradiance(filename.stem, image, irradiance_map, Path(filename), gdf_total, img_size_m, results_list)
#         # Visualizza la heatmap dell'irraggiamento predetto
#         # plt.figure(figsize=(10, 10))
#         # plt.imshow(image, cmap='gray')
#         # plt.imshow(irradiance_map, cmap='Reds', alpha=0.5)
#         # plt.colorbar(label='Annual Irradiance (kWh/m²)')
#         # plt.title('Solar Irradiance Heatmap on Segmented Image')
#         # plt.axis('off')
#         # plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/Irr2_{filename.stem}_hp.png', bbox_inches='tight')
#         # plt.close()
#         if output:
#             print(f'Processed image {filename.stem}, mean IoU: {mean_iou:.4f}, mean angle difference: {mad if mad is not None else "N/A"}, mae: {mae:.4f}, rmse: {rmse:.4f}')
#         for cls in range(num_classes):
#             if area_union[cls] > 0:  # Controlla che la classe sia presente nell'immagine
#                 iou_message = f'NaN' if np.isnan(iou_per_class[cls]) else f'{iou_per_class[cls]:.4f}'
#                 print(f'    Class {cls}: IoU: {iou_message}')
        
#     # Calcola la media degli IoU validi per tutto il dataset
#     dataset_mean_iou = np.mean(total_valid_iou) if total_valid_iou else 0
#     dataset_mean_angle_difference = np.mean(total_angle_difference) if total_angle_difference else 0
#     dataset_mean_mae = np.mean(total_slope_mae) 
#     dataset_mean_rmse = np.mean(total_slope_rmse) 
#     mean_percent_difference = np.mean([pd for pd in results_list if pd != float('inf')])
    
#     return dataset_mean_iou, dataset_mean_angle_difference,dataset_mean_mae, dataset_mean_rmse, mean_percent_difference


                                
# # def test_model_sections(model, test_folder, num_classes, background_class, ignore_index, num_angles, output=True):
# #     start = time.time()
    
# #     test_folder_image = test_folder / 'image'
# #     test_folder_label = test_folder / 'label'
# #     test_folder_slope = test_folder / 'slope'
    
# #     angles = [i * (360/num_angles) for i in range(num_angles)]
    
# #           

# #     total_valid_iou = []  # Accumulatore per gli IoU validi di tutto il dataset
# #     total_angle_difference = []
# #     total_slope_mae = []
# #     total_slope_rmse = []
    
# #     for i, filename in enumerate(test_folder_image.glob('*.png')):
# #         image = cv2.imread(str(filename))
# #         label = cv2.imread(str(test_folder_label / filename.name), cv2.IMREAD_GRAYSCALE)
# #         slope = cv2.imread(str(test_folder_slope / filename.name), cv2.IMREAD_GRAYSCALE)
# #         pred, pred2 = prediction_slopes.segment_img(image, model, tile_size=256, stride=256, background_class=background_class)

# #         corrected_pred = pred.copy()

# #         pred2 = np.where(pred2 > 80, 0, pred2)
# #         slope = np.where(slope==99, 0, slope)

# #         corrected_pred2 = replace_slope_values_with_median(pred, pred2, background_class)

# #         max_value = max(slope.max(), corrected_pred2.max())
# #         min_value = min(slope.min(), corrected_pred2.min())

# #         # Calcola IoU utilizzando la funzione intersect_and_union modificata
# #         area_intersect, area_union, _, _ = intersect_and_union(corrected_pred, label, num_classes, ignore_index)
# #         iou_per_class = calculate_iou_per_class(area_intersect, area_union, num_classes)
# #         valid_iou = iou_per_class[~np.isnan(iou_per_class)]  # Filtra NaN
# #         if valid_iou.size > 0:
# #             total_valid_iou.extend(valid_iou)
# #         mean_iou = np.nanmean(iou_per_class)
        
# #         mad = mean_angle_difference(corrected_pred, label, num_classes, angles, background_class, ignore_index)
# #         if mad is not None:
# #             total_angle_difference.append(mad)

# #         # Calculate error metrics for slope
# #         mae, rmse = calculate_mae_rmse(corrected_pred2, slope)
# #         total_slope_mae.append(mae)
# #         total_slope_rmse.append(rmse)

# #         from matplotlib import gridspec
# #         if (i % 200==0 ) and output:
# #         # if (i % 200 == 0) and output:
# #             plt.figure(figsize=(25, 10))
# #             gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 0.05, 1, 0.05])  # L'ultimo elemento è per la colorbar
            
# #             ax0 = plt.subplot(gs[0])
# #             im0 = ax0.imshow(image)
# #             ax0.set_title('Image')
# #             ax0.axis('off')  # Nascondi gli assi per una visualizzazione più pulita

# #             ax1 = plt.subplot(gs[1])
# #             norm = plt.Normalize(vmin=0, vmax=100)
# #             im1 = ax1.imshow(slope, norm=norm, cmap='tab20')
# #             ax1.set_title('Ground truth')
# #             ax1.axis('off')

# #             ax2 = plt.subplot(gs[2])
# #             norm2 = plt.Normalize(vmin=0, vmax=100)
# #             im2 = ax2.imshow(corrected_pred2, norm=norm2, cmap='tab20')
# #             ax2.set_title(f'Pred')
# #             ax2.axis('off')

# #             # Aggiungi la colorbar
# #             cax = plt.subplot(gs[3])  # Asse per la colorbar
# #             plt.colorbar(im2, cax=cax, orientation='vertical')
# #             cax.set_title('Scale')

# #             ax3 = plt.subplot(gs[4])
# #             norm3 = plt.Normalize(vmin=-50, vmax=50)  # Assicurati che l'intervallo sia adeguato per mostrare le differenze
# #             im3 = ax3.imshow(slope - corrected_pred2, cmap='tab10', norm=norm3)
# #             ax3.set_title('Differences')
# #             ax3.axis('off')

# #             # Aggiungi la colorbar
# #             cax = plt.subplot(gs[5])  # Asse per la colorbar
# #             plt.colorbar(im3, cax=cax, orientation='vertical')
# #             cax.set_title('Scale')

# #             plt.subplots_adjust(wspace=0.2, hspace=0.1)  # Regola lo spazio tra i subplot se necessario
# #             plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/output_{filename.stem}_sm3.png', bbox_inches='tight')
# #             plt.close()  # Chiude la figura per liberare memoria

# #         if (i % 200==0) and output:
# #         # if (i % 200==0) and output:
# #             f, axarr = plt.subplots(1,3, figsize=(10, 10))
# #             axarr[0].imshow(image)
# #             axarr[1].imshow(label)
# #             axarr[2].imshow(corrected_pred)
# #             axarr[0].title.set_text('Image')
# #             axarr[1].title.set_text('Ground truth')
# #             axarr[2].title.set_text(f'mean IoU: {mean_iou:.4f}')

# #             plt.subplots_adjust(wspace=0.1, hspace=0.1)
# #             for ax in axarr:
# #                 ax.axis('off')
# #             plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/output_{filename.stem}_1.png', bbox_inches='tight')
# #             plt.close(f)  # Chiude la figura per liberare memoria
# #         if output:
# #             print(f'Processed image {filename.stem}, mean IoU: {mean_iou:.4f}, mean angle difference: {mad if mad is not None else "N/A"}')
# #         for cls in range(num_classes):
# #             if area_union[cls] > 0:  # Controlla che la classe sia presente nell'immagine
# #                 iou_message = f'NaN' if np.isnan(iou_per_class[cls]) else f'{iou_per_class[cls]:.4f}'
# #                 print(f'    Class {cls}: IoU: {iou_message}')
    

#     # # Calcola la media degli IoU validi per tutto il dataset
#     # dataset_mean_iou = np.mean(total_valid_iou) if total_valid_iou else 0
#     # dataset_mean_angle_difference = np.mean(total_angle_difference) if total_angle_difference else 0
#     # dataset_mean_mae = np.mean(total_slope_mae) 
#     # dataset_mean_rmse = np.mean(total_slope_rmse) 

#     # end = time.time()
#     # elapsed_time = end - start
    
#     # if output:
#     #     # Stampa i risultati aggregati per tutto il dataset
#     #     print('\nTotal processing time:', elapsed_time, 'seconds')
#     #     print(f'Dataset Mean IoU: {dataset_mean_iou:.4f}')
#     #     print(f'Dataset Mean Angle Difference: {dataset_mean_angle_difference:.4f}')
#     #     print(f'Dataset Mean MAE: {dataset_mean_mae:.4f}')
#     #     print(f'Dataset Mean RMSE: {dataset_mean_rmse:.4f}')

#     # return dataset_mean_iou, dataset_mean_angle_difference,dataset_mean_mae, dataset_mean_rmse, elapsed_time

# num_classes = 18
# background_class = 0
# ignore_index = -1
# num_angles = 16

# print(f'Testing model <model_test17>...')

# # # Chiamata alla funzione test_model_sections con i parametri aggiornati
# # mean_percent_difference = test_model_sections(
# #     model_test, 
# #     rid_test_folder, 
# #     num_classes, 
# #     background_class, 
# #     ignore_index, 
# #     num_angles, 
# #     output=True
# # )
# # print(f'Dataset Mean % Difference: {mean_percent_difference:.4f}')

# # Chiamata alla funzione test_model_sections con i parametri aggiornati
# dataset_mean_iou, dataset_mean_angle_difference,dataset_mean_mae, dataset_mean_rmse, mean_percent_difference = test_model_sections(
#     model_test, 
#     rid_test_folder, 
#     num_classes, 
#     background_class, 
#     ignore_index, 
#     num_angles, 
#     output=True
# )

# # # Stampa i risultati finali per tutto il dataset
# print(f'Dataset Mean IoU: {dataset_mean_iou:.4f}')
# print(f'Dataset Mean Angle Difference: {dataset_mean_angle_difference:.4f}')
# print(f'Dataset Mean MAE: {dataset_mean_mae:.4f}')
# print(f'Dataset Mean RMSE: {dataset_mean_rmse:.4f}')
# print(f'Dataset Mean % Difference: {mean_percent_difference:.4f}')
# # print(f'Elapsed Time: {elapsed_time:.2f} seconds')
import matplotlib.patches as mpatches
import torch
import pathlib
import cv2
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pvlib
from pathlib import Path
import matplotlib.colors as mcolors
import geopandas as gpd
from passion.segmentation import prediction_slopes
import logging
import requests
from scipy import stats
import dask_geopandas as dg
import matplotlib.patches as mpatches

# Imposta il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = ('cuda' if torch.cuda.is_available() else 'cpu')

models_folder_path = pathlib.Path('/scratch/clear/aboccala/PASSION/workflow/output/model')
section_models_folder_path = models_folder_path / 'section-segmentation'

# Sections models
model_test = torch.load(str(section_models_folder_path / 'test_seg_ge_va.pth'), map_location=torch.device(device))

rid_test_folder = pathlib.Path('/scratch/clear/aboccala/training/RID/output_mix_ge_va/test_va')
rid_test_folder_image = rid_test_folder / 'image'
rid_test_folder_label = rid_test_folder / 'label'
rid_test_folder_slope = rid_test_folder / 'slope'

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
    
    pred_label_angles = pred_label[combined_pred_label == combined_label]
    pred_label_angles = pred_label_angles[pred_label_angles < len(angles)]
    label_angles = label[combined_pred_label == combined_label]
    label_angles = label_angles[label_angles < len(angles)]

    total_diff = 0
    for angle_1, angle_2 in zip(np.nditer(pred_label_angles, flags=['zerosize_ok']), np.nditer(label_angles, flags=['zerosize_ok'])):
        try:
            angle_1, angle_2 = angles[angle_1], angles[angle_2]
        except Exception as e:
            logging.error(f"Error in angle conversion: {angle_1}, {angle_2}, {str(e)}")
        diff = angle_difference(angle_1, angle_2)
        total_diff += diff
    
    if pred_label_angles.size == 0:
        return None
    
    mean_diff = total_diff / pred_label_angles.size
    
    return mean_diff

def visualize_ground_truth(image, label, slope, filename):
    orientation_labels = [
        "Background", "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "Flat"
    ]
    slope_labels = [
        "Background", "Flat Roof", ">0° & <10°", ">=10° & <20°", ">=20° & <30°", 
        ">=30° & <40°", ">=40° & <50°", ">=50°"
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 24))  # Disposizione in colonna

    # Original Image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=20)
    axes[0].axis('off')

    # Orientation Ground Truth
    cmap_orient = plt.get_cmap('viridis', 18)
    im1 = axes[1].imshow(label, cmap=cmap_orient, vmin=0, vmax=17)
    
    # Create custom legend for orientations
    orientation_patches = [mpatches.Patch(color=cmap_orient(i), label=orientation_labels[i]) for i in range(18)]
    legend1 = axes[1].legend(handles=orientation_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=16)
    axes[1].add_artist(legend1)
    
    axes[1].set_title('Ground Truth Orientations', fontsize=20)
    axes[1].axis('off')

    # Slope Ground Truth
    cmap_slope = plt.get_cmap('plasma', 8)
    im2 = axes[2].imshow(slope, cmap=cmap_slope, vmin=0, vmax=7)
    
    # Create custom legend for slopes
    slope_patches = [mpatches.Patch(color=cmap_slope(i), label=slope_labels[i]) for i in range(8)]
    legend2 = axes[2].legend(handles=slope_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=16)
    axes[2].add_artist(legend2)
    
    axes[2].set_title('Ground Truth Slopes', fontsize=20)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/{filename}_ground_truth.png', bbox_inches='tight')
    plt.close(fig)

def calculate_iou_per_class(area_intersect, area_union, num_classes):
    iou_per_class = np.zeros(num_classes)
    for cls in range(num_classes):
        if area_union[cls] > 0:
            iou_per_class[cls] = area_intersect[cls] / area_union[cls]
        else:
            iou_per_class[cls] = np.nan
    return iou_per_class

def slope_class_to_value(slope_class):
    slope_mapping = {
        1: 0,    # flat
        2: 5,    # 0 < slope < 10
        3: 15,   # 10 <= slope < 20
        4: 25,   # 20 <= slope < 30
        5: 35,   # 30 <= slope < 40
        6: 45,   # 40 <= slope < 50
        7: 55    # slope >= 50
    }
    return slope_mapping.get(slope_class, 0)  # Default to 0 if class not found

def replace_slope_values_with_median(segmentation, slope, background_class=0, method='median'):
    modified_slope_image = np.zeros_like(slope, dtype=float)
    unique_classes = np.unique(segmentation)
    
    for cls in unique_classes:
        if cls == background_class:
            mask = segmentation == cls
            modified_slope_image[mask] = np.nan
        elif cls != background_class:
            mask = segmentation == cls
            slopes = slope[mask]
            if slopes.size > 0:
                if method == 'median':
                    slope_value = np.median(slopes)
                elif method == 'mean':
                    slope_value = np.mean(slopes)
                elif method == 'mode':
                    slope_value = stats.mode(slopes, nan_policy='omit').mode[0]
                modified_slope_image[mask] = slope_value
            else:
                modified_slope_image[mask] = np.nan

    return modified_slope_image

def calculate_mae_rmse(pred, true):
    pred = pred.flatten()
    true = true.flatten()
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    return mae, rmse

def get_pvgis_tmy(latitude, longitude, outputformat='json', usehorizon=True,
                  userhorizon=None, startyear=None, endyear=None,
                  raddatabase='PVGIS-SARAH2', meteodb='ERA-Interim',
                  map_variables=True, url='https://re.jrc.ec.europa.eu/api/v5_2/', timeout=120):
    params = {
        'lat': latitude,
        'lon': longitude,
        'outputformat': outputformat,
        'usehorizon': int(usehorizon),
        'userhorizon': ','.join(map(str, userhorizon)) if userhorizon else None,
        'startyear': startyear,
        'endyear': endyear,
        'raddatabase': raddatabase,
        'meteodb': meteodb
    }
    response = requests.get(url + 'tmy', params=params, timeout=timeout)
    if response.status_code != 200:
        logging.error(f"Error fetching PVGIS TMY data: {response.text}")
        response.raise_for_status()

    if outputformat == 'json':
        data = response.json()
        data_frame = pd.DataFrame(data['outputs']['tmy_hourly'])
        if map_variables:
            data_frame.rename(columns={
                'G(h)': 'ghi',
                'Gb(n)': 'dni',
                'Gd(h)': 'dhi'
            }, inplace=True)
        return data_frame, data['inputs'], data['meta']
    else:
        raise ValueError("Unsupported format")

def calculate_irradiance(segmentation, slope, angles, tmy_data, location, num_classes, background_class):
    irradiance_map = np.zeros_like(segmentation, dtype=np.float32)

    for cls in range(1, num_classes):
        mask = segmentation == cls
        if not np.any(mask):
            continue

        # Convert the slope class to a real value
        slope_values = np.vectorize(slope_class_to_value)(slope[mask])
        tilt = np.median(slope_values)

        azimuth = 180 if cls == 17 else angles[cls % len(angles)]

        solar_position = location.get_solarposition(times=tmy_data.index)
        irradiance_components = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            dni=tmy_data['dni'],
            ghi=tmy_data['ghi'],
            dhi=tmy_data['dhi'],
            dni_extra=pvlib.irradiance.get_extra_radiation(tmy_data.index),
            solar_zenith=solar_position['apparent_zenith'],
            solar_azimuth=solar_position['azimuth'],
            model='perez', albedo=0.20
        )
        irradiance_map[mask] = irradiance_components['poa_global'].sum() / 1000

    return irradiance_map

def visualize_comparisons(image, label, pred, slope, pred2, filename):
    orientation_labels = [
        "Background", "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "Flat"
    ]
    slope_labels = [
        "Background", "Flat Roof", ">0° & <10°", ">=10° & <20°", ">=20° & <30°", 
        ">=30° & <40°", ">=40° & <50°", ">=50°"
    ]

    fig, axes = plt.subplots(5, 1, figsize=(10, 30))  # Disposizione in colonna

    # Original Image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=16)
    axes[0].axis('off')

    # Ground Truth Segmentation
    cmap_orient = plt.get_cmap('viridis', 18)
    im1 = axes[1].imshow(label, cmap=cmap_orient, vmin=0, vmax=17)
    
    # Create custom legend for orientations
    orientation_patches = [mpatches.Patch(color=cmap_orient(i), label=orientation_labels[i]) for i in range(18)]
    legend1 = axes[1].legend(handles=orientation_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
    axes[1].add_artist(legend1)
    
    axes[1].set_title('Ground Truth Orientations', fontsize=16)
    axes[1].axis('off')

    # Predicted Segmentation
    im2 = axes[2].imshow(pred, cmap=cmap_orient, vmin=0, vmax=17)
    
    # Add legend to predicted segmentation
    legend2 = axes[2].legend(handles=orientation_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
    axes[2].add_artist(legend2)
    
    axes[2].set_title('Predicted Orientations', fontsize=16)
    axes[2].axis('off')

    # Ground Truth Slope
    cmap_slope = plt.get_cmap('plasma', 8)
    im3 = axes[3].imshow(slope, cmap=cmap_slope, vmin=0, vmax=7)
    
    # Create custom legend for slopes
    slope_patches = [mpatches.Patch(color=cmap_slope(i), label=slope_labels[i]) for i in range(8)]
    legend3 = axes[3].legend(handles=slope_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
    axes[3].add_artist(legend3)
    
    axes[3].set_title('Ground Truth Slopes', fontsize=16)
    axes[3].axis('off')

    # Predicted Slope
    im4 = axes[4].imshow(pred2, cmap=cmap_slope, vmin=0, vmax=7)
    
    # Add legend to predicted slopes
    legend4 = axes[4].legend(handles=slope_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
    axes[4].add_artist(legend4)
    
    axes[4].set_title('Predicted Slopes', fontsize=16)
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/{filename}.png', bbox_inches='tight')
    plt.close(fig)

def compare_irradiance(b_id, image, pred_irradiance, filename, gdf, img_size_m, results_list):
    plt.figure(figsize=(20, 6))

    ax0 = plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    ax0.set_xticks([])
    ax0.set_yticks([])

    pixel_area = 0.04  # 0.2 meters per pixel, squared for area

    ax1 = plt.subplot(1, 3, 2)
    norm = mcolors.Normalize(vmin=0, vmax=1700)
    im1 = plt.imshow(pred_irradiance, cmap='YlOrRd', norm=norm)
    plt.colorbar(ax=ax1)
    total_pred_weighted_irradiance = np.sum(pred_irradiance[pred_irradiance > 0] * pixel_area)
    total_pred_area = np.count_nonzero(pred_irradiance > 0) * pixel_area
    mean_pred_irradiance = total_pred_weighted_irradiance / total_pred_area if total_pred_area > 0 else 0
    plt.title(f'Predicted Irradiance - Weighted Mean: {mean_pred_irradiance:.2f} kWh/m2 year')
    ax1.set_xticks([])
    ax1.set_yticks([])

    building = gdf[gdf['b_id'] == int(b_id)]
    if not building.empty:
        centroid = building.geometry.centroid.iloc[0]
        buffer = centroid.buffer(img_size_m / 2, cap_style=3)

        sg_data = gdf[gdf.geometry.intersects(buffer)]
        sg_data = sg_data.copy()
        sg_data['intersected_geometry'] = sg_data.geometry.intersection(buffer)
        sg_data['area'] = sg_data['intersected_geometry'].area

        sg_data['weighted_irr'] = sg_data['MSTRAHLUNG'] * sg_data['area']
        total_weighted_irradiance = sg_data['weighted_irr'].sum()
        total_area = sg_data['area'].sum()
        mean_real_irradiance = total_weighted_irradiance / total_area if total_area > 0 else 0

        ax2 = plt.subplot(1, 3, 3)
        ax2.set_xlim([buffer.bounds[0], buffer.bounds[2]])
        ax2.set_ylim([buffer.bounds[1], buffer.bounds[3]])
        norm = mcolors.Normalize(vmin=0, vmax=1700)
        sg_data.plot(ax=ax2, column='MSTRAHLUNG', cmap='YlOrRd', legend=True, norm=norm)
        plt.title(f'Real Irradiance - Weighted Mean: {mean_real_irradiance:.2f} kWh/m2 year')
        ax2.set_xticks([])
        ax2.set_yticks([])

    percent_difference = ((mean_real_irradiance - mean_pred_irradiance) / mean_pred_irradiance) * 100 if mean_pred_irradiance > 0 else float('inf')
    results_list.append(percent_difference)

    plt.suptitle(f'% Difference: {percent_difference:.2f}%')
    plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/{filename.stem}_1.png', bbox_inches='tight')
    plt.close()

def test_model_sections(model, test_folder, num_classes, background_class, ignore_index, num_angles, output=True):
    test_folder_image = test_folder / 'image'
    test_folder_label = test_folder / 'label'
    test_folder_slope = test_folder / 'slope'

    shapefile_path = r'/scratch/clear/aboccala/PASSION/notebooks/evaluation/vaud_final.shp'
    test_b_ids = {f.stem for f in Path(test_folder_image).glob('*.png')}

    img_size_px = 256
    img_res_m_per_px = 0.2
    img_size_m = img_size_px * img_res_m_per_px

    gdf = dg.read_file(shapefile_path, npartitions=10)
    gdf_total = gdf.compute()
    
    try:
        test_b_ids_int = {int(id) for id in test_b_ids}
        logging.info("Conversione riuscita. ID convertiti in interi.")
    except ValueError:
        logging.error("Errore nella conversione. Verifica che tutti gli ID siano numerici.")

    if 'test_b_ids_int' in locals():
        gdf_filtered = gdf[gdf['b_id'].isin(test_b_ids_int)]
        gdf = gdf_filtered.compute()
        if gdf.empty:
            logging.warning("Nessun dato corrispondente trovato. Verifica i valori degli ID.")
        else:
            logging.info("Dati filtrati con successo.")

    angles = np.array([i * (360 / num_angles) for i in range(num_angles)])

    total_valid_iou = []
    total_valid_slope_iou = []  # Per slope IoU
    total_angle_difference = []
    total_slope_mae = []
    total_slope_rmse = []
    results_list = []

    iou_per_image = []
    slope_iou_per_image = []

    for filename in tqdm(list(test_folder_image.glob('*.png'))):
        try:
            image = cv2.imread(str(filename))
            label = cv2.imread(str(test_folder_label / filename.name), cv2.IMREAD_GRAYSCALE)
            slope = cv2.imread(str(test_folder_slope / filename.name), cv2.IMREAD_GRAYSCALE)

            visualize_ground_truth(image, label, slope, filename.stem)

            pred, pred2 = prediction_slopes.segment_img(image, model, tile_size=256, stride=256, background_class=background_class)
            pred2 = replace_slope_values_with_median(pred, pred2, background_class)

            b_id = filename.stem
            building = gdf[gdf['b_id'] == int(b_id)]
            building_projected = building.to_crs('EPSG:2056') 
            if not building_projected.empty:
                centroid_projected = building_projected.geometry.centroid.iloc[0]
                centroid_geo = gpd.GeoDataFrame(geometry=gpd.GeoSeries(centroid_projected), crs=building_projected.crs)
                centroid_geo = centroid_geo.to_crs(epsg=4326)
                centroid = centroid_geo.geometry.iloc[0]

                latitude = centroid.y
                longitude = centroid.x

                location = pvlib.location.Location(latitude, longitude)
                tmy_data, _, _ = get_pvgis_tmy(latitude, longitude)
                tmy_data['time(UTC)'] = pd.to_datetime(tmy_data['time(UTC)'], format='%Y%m%d:%H%M', utc=True)
                tmy_data.set_index('time(UTC)', inplace=True)

            area_intersect, area_union, _, _ = intersect_and_union(pred, label, num_classes, ignore_index)
            iou_per_class = calculate_iou_per_class(area_intersect, area_union, num_classes)
            valid_iou = iou_per_class[~np.isnan(iou_per_class)]
            if valid_iou.size > 0:
                total_valid_iou.extend(valid_iou)
                iou_per_image.append(np.nanmean(iou_per_class))

            area_intersect_slope, area_union_slope, _, _ = intersect_and_union(pred2, slope, 8, ignore_index)
            iou_per_class_slope = calculate_iou_per_class(area_intersect_slope, area_union_slope, 8)
            valid_slope_iou = iou_per_class_slope[~np.isnan(iou_per_class_slope)]
            if valid_slope_iou.size > 0:
                total_valid_slope_iou.extend(valid_slope_iou)
                slope_iou_per_image.append(np.nanmean(iou_per_class_slope))

            mad = mean_angle_difference(pred, label, num_classes, angles, background_class, ignore_index)
            if mad is not None:
                total_angle_difference.append(mad)

            mae, rmse = calculate_mae_rmse(pred2, slope)
            total_slope_mae.append(mae)
            total_slope_rmse.append(rmse)

            irradiance_map = calculate_irradiance(pred, pred2, angles, tmy_data, location, num_classes, background_class)
            irradiance_map[pred == 0] = np.nan

            visualize_comparisons(image, label, pred, slope, pred2, filename.stem)
            compare_irradiance(filename.stem, image, irradiance_map, Path(filename), gdf_total, img_size_m, results_list)

            if output:
                logging.info(f'Processed image {filename.stem}, mean IoU: {np.nanmean(iou_per_class):.4f}, mean angle difference: {mad if mad is not None else "N/A"}, mae: {mae:.4f}, rmse: {rmse:.4f}')
            for cls in range(num_classes):
                if area_union[cls] > 0:
                    iou_message = f'NaN' if np.isnan(iou_per_class[cls]) else f'{iou_per_class[cls]:.4f}'
                    logging.info(f'    Class {cls}: IoU: {iou_message}')
        except Exception as e:
            logging.error(f"Error processing file {filename.stem}: {str(e)}")
        
    dataset_mean_iou = np.mean(total_valid_iou) if total_valid_iou else 0
    dataset_mean_slope_iou = np.mean(total_valid_slope_iou) if total_valid_slope_iou else 0
    dataset_mean_angle_difference = np.mean(total_angle_difference) if total_angle_difference else 0
    dataset_mean_mae = np.mean(total_slope_mae)
    dataset_mean_rmse = np.mean(total_slope_rmse)
    mean_percent_difference = np.mean([pd for pd in results_list if pd != float('inf')])

    # Plot IoU per image for orientations and slopes
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    ax[0].plot(iou_per_image, marker='o', linestyle='-', color='b', label='Orientation IoU')
    ax[0].set_title('Mean IoU per Image - Orientation')
    ax[0].set_xlabel('Image Index')
    ax[0].set_ylabel('Mean IoU')
    ax[0].legend()

    ax[1].plot(slope_iou_per_image, marker='o', linestyle='-', color='r', label='Slope IoU')
    ax[1].set_title('Mean IoU per Image - Slope')
    ax[1].set_xlabel('Image Index')
    ax[1].set_ylabel('Mean IoU')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f'/scratch/clear/aboccala/PASSION/notebooks/evaluation/images/mean_iou_per_image.png', bbox_inches='tight')
    plt.close()

    return dataset_mean_iou, dataset_mean_slope_iou, dataset_mean_angle_difference, dataset_mean_mae, dataset_mean_rmse, mean_percent_difference

num_classes = 18
background_class = 0
ignore_index = -1
num_angles = 16

print(f'Testing model <model_test17>...')

# Chiamata alla funzione test_model_sections con i parametri aggiornati
dataset_mean_iou, dataset_mean_slope_iou, dataset_mean_angle_difference, dataset_mean_mae, dataset_mean_rmse, mean_percent_difference = test_model_sections(
    model_test, 
    rid_test_folder, 
    num_classes, 
    background_class, 
    ignore_index, 
    num_angles, 
    output=True
)

# # Stampa i risultati finali per tutto il dataset
print(f'Dataset Mean IoU (Orientation): {dataset_mean_iou:.4f}')
print(f'Dataset Mean IoU (Slope): {dataset_mean_slope_iou:.4f}')
print(f'Dataset Mean Angle Difference: {dataset_mean_angle_difference:.4f}')
print(f'Dataset Mean MAE: {dataset_mean_mae:.4f}')
print(f'Dataset Mean RMSE: {dataset_mean_rmse:.4f}')
print(f'Dataset Mean % Difference: {mean_percent_difference:.4f}')
