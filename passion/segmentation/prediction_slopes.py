import pathlib
import numpy as np
import cv2
import PIL
from enum import Enum
import tqdm
import shapely.geometry
import PIL

import torch
import torchvision

from typing import List

import passion.util

def segment_dataset(input_path: pathlib.Path,
                    model: torch.nn.Module,
                    output_path: pathlib.Path,
                    background_class: int,
                    tile_size: int = 256,
                    stride: int = 256,
                    save_masks: bool = True,
                    save_filtered: bool = True,
                    opening_closing_kernel: int = 9,
                    erosion_kernel: int = 9
):
  '''Segments a full dataset generated by generate_dataset(),
  saving the rooftop segmented masks in the specified folder.
  Segmented images are saved with the same name as the original
  images appending '_MASK' in the end.

  A stride can be specified smaller than the tile_size
  in order to make predictions in the borders.

  Tile size must match model's input size.

  ---
  
  input_path              -- Path, path of the input images.
  model                   -- torch.nn.Module, path of the training data.
  output_path             -- Path, output path of the segmented data.
  background_class        -- int, integer defined to be the background class in the input dataset.
  tile_size               -- int, input size of the model.
  stride                  -- int, image separation for each new prediction.
  save_masks              -- bool, if true, saves the resulting masks into disk.
  save_filtered           -- bool, if true, saves the filtered images into disk.
  opening_closing_kernel  -- int, size of the kernel for opening and closing in post processing.
  erosion_kernel          -- int, size of the kernel for erosion in post processing.
  '''
  output_path.mkdir(parents=True, exist_ok=True)
  
  paths = list(input_path.glob('*.tif'))
  pbar = tqdm.tqdm(paths)
  for img_path in pbar:
    src = passion.util.io.read_geotiff(img_path)
    image = src.ReadAsArray()
    # Change channels first to channels last
    image = np.moveaxis(image, 0, -1)

    image = preprocess_input(image)

    seg_image = segment_img(image, model, tile_size, stride, background_class)

    seg_image = postprocess_output(seg_image, opening_closing_kernel, erosion_kernel, background_class)
    
    seg_image = seg_image[np.newaxis, ...]
    channels, height, width = seg_image.shape

    # Change channels first to channels last
    seg_image = np.moveaxis(seg_image, 0, -1)
    passion.util.io.write_geotiff(str(output_path / (img_path.stem + '_MASK.tif')),
                                  seg_image,
                                  src.GetGeoTransform(),
                                  src.GetProjection(),
                                  src.GetMetadata())

  return

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

def segment_img(image: np.ndarray, model: torch.nn.Module, tile_size: int, stride: int, background_class: int):
    """Segments a single image in numpy format with a given model. Tile size has to be specified and must match model's input size.
    This function now returns both segmentation and slope predictions."""
    if type(image) != np.ndarray:
        print('Image type: {0} not a np.array'.format(type(image)))
        return None
    if len(image.shape) != 3 or image.shape[-1] != 3:
        print('Image shape: {0} not in format MxNx3'.format(image.shape))
        return None

    tiles = divide_img_tiles(image, tile_size, stride)
    seg_tiles = []
    slope_tiles = []

    for tile in tiles:
        pred_segmentation, pred_slope = segment_tile(tile, model, tile_size, background_class)
        if pred_segmentation is None or pred_slope is None:
            print('Error processing tile, returning...')
            return None, None
        seg_tiles.append(pred_segmentation)
        slope_tiles.append(pred_slope)

    seg_image = compose_tiles_pil(np.array(seg_tiles), image.shape[:2], stride)
    slope_image = compose_tiles_pil(np.array(slope_tiles), image.shape[:2], stride)

    return seg_image, slope_image

# def segment_tile(tile: np.ndarray, model: torch.nn.Module, tile_size: int, background_class: int):
#     """Segments a single tile of model's input size. This function returns both segmentation and slope predictions."""
#     if type(tile) != np.ndarray:
#         print('Tile type: {0} not a np.array'.format(type(tile)))
#         return None, None
#     if tile.shape != (tile_size, tile_size, 3):
#         print('Tile shape: {0} not in format {1}x{1}x3'.format(tile.shape, tile_size))
#         return None, None

#     trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#     tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
#     tile = trans(tile).reshape(1, 3, tile_size, tile_size)
#     tile_tensor = tile.to("cuda" if torch.cuda.is_available() else "cpu")

#     pred_segmentation, pred_slope = None, None  # Initialize variables to handle edge cases

#     with torch.no_grad():
#         try:
#             outputs = model(tile_tensor)
#             if outputs and len(outputs) == 2:
#                 pred_segmentation = torch.argmax(outputs[0], dim=1).squeeze(0).cpu().numpy()
#                 # Applica la moltiplicazione per 99 direttamente sul tensore
#                 pred_slope = outputs[1].squeeze(0) * 255  # Moltiplica per 99 per scalare in [0, 99]
#                 pred_slope = pred_slope.cpu().numpy()  # Converti in numpy array
                
#             else:
#                 print('Model did not return expected outputs')
#                 return None, None
#         except Exception as e:
#             print(f'Failed to process tile with error: {str(e)}')
#             return None, None

#     # Adjust classes if background_class is not 0
#     if background_class != 0 and pred_segmentation is not None:
#         pred_segmentation = np.where(pred_segmentation == background_class, 0, pred_segmentation)
#         pred_segmentation = np.where(pred_segmentation != 0, pred_segmentation + 1, pred_segmentation)

#     return pred_segmentation, pred_slope

def segment_tile(tile: np.ndarray, model: torch.nn.Module, tile_size: int, background_class: int):
    """Segments a single tile of model's input size. This function returns both segmentation and slope predictions."""
    if type(tile) != np.ndarray:
        print('Tile type: {0} not a np.array'.format(type(tile)))
        return None, None
    if tile.shape != (tile_size, tile_size, 3):
        print('Tile shape: {0} not in format {1}x{1}x3'.format(tile.shape, tile_size))
        return None, None

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    tile = trans(tile).reshape(1, 3, tile_size, tile_size)
    tile_tensor = tile.to("cuda" if torch.cuda.is_available() else "cpu")

    pred_segmentation, pred_slope = None, None  # Initialize variables to handle edge cases

    with torch.no_grad():
        outputs = model(tile_tensor)
        if outputs and len(outputs) == 2:
            pred_segmentation = torch.argmax(outputs[0], dim=1).squeeze(0).cpu().numpy()
            # pred_slope = outputs[1].squeeze(0) * 255  # Moltiplica per 99 per scalare in [0, 99]
            pred_slope = torch.argmax(outputs[1], dim=1).squeeze(0).cpu().numpy()  # Moltiplica per 99 per scalare in [0, 99]



            # Applica la maschera della segmentazione alla predizione della pendenza
            mask = pred_segmentation != background_class
            pred_slope_masked = np.where(mask, pred_slope, np.nan)  # Usa np.nan o un altro valore per indicare 'nessuna pendenza'

        else:
            print('Model did not return expected outputs')
            return None, None

    return pred_segmentation, pred_slope_masked

def divide_img_tiles(image: np.ndarray, tile_size: int, stride: int):
  '''Divides an image into a list of tiles of specified size.'''
  img_size_y, img_size_x, img_channels = image.shape

  tiles = []
  for h in range(0,img_size_y,stride):
    for w in range(0,img_size_x,stride):
      i = image[h:h+tile_size,w:w+tile_size]
      i = np.pad(i, ((0, tile_size - i.shape[0]), (0, tile_size - i.shape[1]), (0, 0)),
              mode='constant', constant_values=0)
      tiles.append(i)
  
  return np.array(tiles)

'''
class MERGE_MODE(Enum):
  \'''Custom enum to register the different merging policies.\'''
  OR, AND, VOTE = range(3)
'''

def compose_tiles(tiles: list,
                  img_shape: tuple,
                  stride: int
                  #merge_mode: MERGE_MODE=MERGE_MODE.AND
):
  '''Composes back a list of tiles into a single image.

  If a stride smaller than the tile size is specified,
  a merging policy can be specified from the following:
  
  - MERGE_MODE.AND:   For multiple predictions in a single
  pixel, its value will only be 255 if all of the predictions
  for that pixel are 255.
  - MERGE_MODE.OR:    For multiple predictions in a single
  pixel, its value will only be 255 if any of the predictions
  for that pixel are 255.
  - MERGE_MODE.VOTE:  For multiple predictions in a single
  pixel, its value will only be 255 if the majority of the
  predictions for that pixel are 255.
  '''
  if type(tiles) != np.ndarray:
    print('Error: input tiles type {0} not np.ndarray'.format(type(tiles)))
    return None
  if len(img_shape) != 2:
    print('Image shape {0} not two dimensional'.format(img_shape))
    return None
  
  num_images = len(tiles)
  tile_size_y, tile_size_x = tiles[0].shape
  img_size_y, img_size_x = img_shape

  final_pred = PIL.Image.new('L', (img_size_x, img_size_y), color=1)
  
  '''
  if merge_mode == MERGE_MODE.AND:
    sliding_window_mask = np.array(PIL.Image.new('L', (img_size_x, img_size_y), color=1))
  else:
    sliding_window_mask = np.array(PIL.Image.new('L', (img_size_x, img_size_y), color=0))
  '''
  current_x = 0
  current_y = 0

  for i, tile in enumerate(tiles):
    p = PIL.Image.fromarray(np.uint8(tile))
    '''
    if tile_size_x != stride:
      former = sliding_window_mask[current_y:current_y+tile_size_y,current_x:current_x+tile_size_x]
      new = np.asarray(p)

      new = np.pad(new, ((0, tile_size_x - new.shape[0]), (0, tile_size_y - new.shape[1])),
                  mode='constant', constant_values=0)
      
      if merge_mode == MERGE_MODE.OR:
        former = np.pad(former, ((0, tile_size_x - former.shape[0]), (0, tile_size_y - former.shape[1])),
                mode='constant', constant_values=0)
        merge = np.logical_or(former, new).astype(np.uint8)
      if merge_mode == MERGE_MODE.AND:
        former = np.pad(former, ((0, tile_size_x - former.shape[0]), (0, tile_size_y - former.shape[1])),
                mode='constant', constant_values=1)
        merge = np.logical_and(former, new).astype(np.uint8)
      if merge_mode == MERGE_MODE.VOTE:
        former = np.pad(former, ((0, tile_size_x - former.shape[0]), (0, tile_size_y - former.shape[1])),
                mode='constant', constant_values=0)
        merge = former + new

      p = PIL.Image.fromarray(merge)

      target_shape = sliding_window_mask[current_y:current_y+tile_size_y,current_x:current_x+tile_size_x].shape
      sliding_window_mask[current_y:current_y+tile_size_y,current_x:current_x+tile_size_x] = merge[0:target_shape[0], 0:target_shape[1]]
    '''
    final_pred.paste(p, box=(current_x,current_y,current_x+tile_size_x,current_y+tile_size_y))
    
    
    current_x += stride
        
    if current_x >= img_size_x:
      current_x = 0
      current_y += stride

  final_pred_arr = np.asarray(final_pred)

  #threshold = np.amax(final_pred_arr) // 2
  #final_pred_arr = (final_pred_arr > threshold).astype(np.uint8) * 255
  
  return final_pred_arr

import numpy as np
import PIL.Image

def compose_tiles_s(tiles, img_shape, stride):
    """Composes back a list of tiles into a single image."""
    if tiles.ndim not in [3, 4]:  # Checking if tiles are in the expected number of dimensions
        print(f'Error: input tiles dimension {tiles.ndim} not expected.')
        return None

    if len(img_shape) != 2:
        print(f'Image shape {img_shape} not two dimensional')
        return None

    img_size_y, img_size_x = img_shape
    num_tiles_y = (img_size_y + stride - 1) // stride
    num_tiles_x = (img_size_x + stride - 1) // stride

    if tiles.ndim == 4:
        tile_size_y, tile_size_x, _ = tiles[0].shape
    else:
        tile_size_y, tile_size_x = tiles[0].shape

    final_image = np.zeros((num_tiles_y * stride, num_tiles_x * stride), dtype=tiles.dtype)

    tile_index = 0
    for y in range(0, num_tiles_y):
        for x in range(0, num_tiles_x):
            if tile_index < len(tiles):
                tile = tiles[tile_index]
                # Calculate the correct slice dimensions
                target_y_start = y * stride
                target_x_start = x * stride
                target_y_end = min(target_y_start + tile_size_y, img_size_y)
                target_x_end = min(target_x_start + tile_size_x, img_size_x)

                final_image[target_y_start:target_y_end, target_x_start:target_x_end] = tile[:(target_y_end - target_y_start), :(target_x_end - target_x_start)]
            tile_index += 1

    # Crop the canvas to the original image size if necessary
    final_image = final_image[:img_size_y, :img_size_x]
    final_pred_arr = np.asarray(final_image)

    return final_pred_arr

from PIL import Image

def compose_tiles_pil(tiles, img_shape, stride):
    """Composes back a list of tiles into a single image using PIL, ensuring correct data types and dimensions,
    and returns the result as a numpy array."""
    img_size_y, img_size_x = img_shape
    num_tiles_y = (img_size_y + stride - 1) // stride
    num_tiles_x = (img_size_x + stride - 1) // stride

    final_image = Image.new('L', (img_size_x, img_size_y), color=1)  # Create a new PIL image for grayscale output

    tile_index = 0
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            if tile_index < len(tiles):
                tile = tiles[tile_index]
                # Ensure the tile is correctly shaped as (height, width)
                if tile.ndim == 3 and tile.shape[0] == 1:  # Handling single-channel 3D arrays shaped (1, height, width)
                    tile = tile.squeeze(0)  # Remove the single channel dimension to make it 2D
                elif tile.ndim != 2:
                    raise ValueError(f"Unexpected tile dimensions: {tile.shape}")
                tile_image = Image.fromarray(tile.astype(np.uint8))
                x_pos = x * stride
                y_pos = y * stride
                # Ensure we don't paste outside the boundaries of the final image
                if x_pos + tile_image.width > img_size_x:
                    tile_image = tile_image.crop((0, 0, img_size_x - x_pos, tile_image.height))
                if y_pos + tile_image.height > img_size_y:
                    tile_image = tile_image.crop((0, 0, tile_image.width, img_size_y - y_pos))
                final_image.paste(tile_image, (x_pos, y_pos))
            tile_index += 1

    # Convert the final PIL image back to a numpy array
    final_array = np.array(final_image)

    return final_array

def preprocess_input(image: np.ndarray):
  '''Preprocessing made to the numpy image before performing segmentation.
  Can be redefined with a custom function as:
  prediction.preprocess_input = custom_preprocess_function
  '''
  return image


def postprocess_output(image: np.ndarray, opening_closing_kernel: int = 7, erosion_kernel: int = 1, background_class: int = 0):
  '''Postprocessing made to the numpy image after performing segmentation.
  Can be redefined with a custom function as:
  prediction.postprocess_output = custom_postprocess_function
  '''
  out_image = np.full(image.shape, background_class)

  opening_closing_kernel = np.ones((opening_closing_kernel, opening_closing_kernel), np.uint8)
  erosion_kernel = np.ones((erosion_kernel, erosion_kernel), np.uint8)

  poly_list = []
  class_list = []
  seg_classes = np.unique(image)[np.unique(image) != background_class]
  for seg_class in seg_classes:
    #image_class = image.copy()
    image_class = (image == seg_class).astype(np.uint8)

    # opening + closing
    image_class = cv2.morphologyEx(image_class, cv2.MORPH_OPEN, opening_closing_kernel)
    image_class = cv2.morphologyEx(image_class, cv2.MORPH_CLOSE, opening_closing_kernel)
    # erosion
    image_class = cv2.erode(image_class, erosion_kernel)

    # From binary to class
    out_image[image_class == 1] = seg_class

  return out_image