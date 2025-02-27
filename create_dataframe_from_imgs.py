import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import convolve

def split_into_patches(image, patch_size=64):
    """Split an image into non-overlapping patches."""
    patches = []
    height, width = image.shape[:2]
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches

def compute_indices_for_pixels(images, eps=1e-6):

    # images argument
    # images = {
    # 'R': red_band_array,        # 2D NumPy array for Red band
    # 'G': green_band_array,      # 2D NumPy array for Green band
    # 'B': blue_band_array,       # 2D NumPy array for Blue band
    # 'N': nir_band_array,        # 2D NumPy array for Near-Infrared band
    # 'E': red_edge_band_array    # 2D NumPy array for Red Edge band
    # }

    """Compute vegetation indices for each pixel from multispectral TIF images."""
    indices = {}
    red = images['R'].astype(np.float32)
    green = images['G'].astype(np.float32)
    blue = images['B'].astype(np.float32)
    nir = images['N'].astype(np.float32)
    red_edge = images['E'].astype(np.float32)
    
    indices['NDVI'] = (nir - red) / (nir + red + eps)
    indices['NGRDI'] = (green - red) / (green + red + eps)
    indices['VARI'] = (green - red) / (green + red - blue + eps)
    indices['GLI'] = (2 * green - red - blue) / (2 * green + red + blue + eps)
    indices['vNDVI'] = (nir - red_edge) / (nir + red_edge + eps)
    indices['RGBVI'] = (green**2 - red * blue) / (green**2 + red * blue + eps)
    indices['MGRVI'] = (green**2 - red**2) / (green**2 + red**2 + eps)

    return indices

def compute_indices_for_pixels_from_jpg(images, eps=1e-6):
    """
    Compute vegetation indices for each pixel using channels from the RGB JPG image.
    For NDVI and vNDVI, use NIR from the TIF and red extracted from the JPG.
    """
    indices = {}
    rgb_img = images['RGB'].astype(np.float32)
    # OpenCV loads images in BGR order
    blue_jpg = rgb_img[..., 0]
    green_jpg = rgb_img[..., 1]
    red_jpg = rgb_img[..., 2]
    
    nir = images['N'].astype(np.float32)
    
    indices['NDVI'] = (nir - red_jpg) / (nir + red_jpg + eps)
    indices['NGRDI'] = (green_jpg - red_jpg) / (green_jpg + red_jpg + eps)
    indices['VARI'] = (green_jpg - red_jpg) / (green_jpg + red_jpg - blue_jpg + eps)
    indices['GLI'] = (2 * green_jpg - red_jpg - blue_jpg) / (2 * green_jpg + red_jpg + blue_jpg + eps)
    # For vNDVI, substitute red edge (not available in JPG) with the red channel from JPG.
    #indices['vNDVI'] = (nir - red_jpg) / (nir + red_jpg + eps)
    indices['RGBVI'] = (green_jpg**2 - red_jpg * blue_jpg) / (green_jpg**2 + red_jpg * blue_jpg + eps)
    indices['MGRVI'] = (green_jpg**2 - red_jpg**2) / (green_jpg**2 + red_jpg**2 + eps)

    return indices

def compute_indices_for_patches(patches_dict, eps=1e-6):
    """Compute vegetation indices for patches from multispectral TIF images."""
    indices_list = []
    num_patches = len(patches_dict['R'])  # Assuming all bands have the same number of patches
    for i in range(num_patches):
        indices = {}
        red = patches_dict['R'][i].astype(np.float32)
        green = patches_dict['G'][i].astype(np.float32)
        blue = patches_dict['B'][i].astype(np.float32)
        nir = patches_dict['N'][i].astype(np.float32)
        red_edge = patches_dict['E'][i].astype(np.float32)
    
        indices['NDVI'] = np.nanmean((nir - red) / (nir + red + eps))
        indices['NGRDI'] = np.nanmean((green - red) / (green + red + eps))
        indices['VARI'] = np.nanmean((green - red) / (green + red - blue + eps))
        indices['GLI'] = np.nanmean((2 * green - red - blue) / (2 * green + red + blue + eps))
        indices['vNDVI'] = np.nanmean((nir - red_edge) / (nir + red_edge + eps))
        indices['RGBVI'] = np.nanmean((green**2 - red * blue) / (green**2 + red * blue + eps))
        indices['MGRVI'] = np.nanmean((green**2 - red**2) / (green**2 + red**2 + eps))

        indices_list.append(indices)
    return indices_list

def compute_indices_for_patches_from_jpg(patches_dict, eps=1e-6):
    """
    Compute vegetation indices for patches using channels from the RGB JPG image.
    For NDVI and vNDVI, use NIR from the TIF and red extracted from the JPG.
    """
    indices_list = []
    num_patches = len(patches_dict['RGB'])
    for i in range(num_patches):
        indices = {}
        rgb_patch = patches_dict['RGB'][i].astype(np.float32)
        blue_jpg = rgb_patch[..., 0]
        green_jpg = rgb_patch[..., 1]
        red_jpg = rgb_patch[..., 2]
        
        nir_patch = patches_dict['N'][i].astype(np.float32)
    
        indices['NDVI'] = np.nanmean((nir_patch - red_jpg) / (nir_patch + red_jpg + eps))
        indices['NGRDI'] = np.nanmean((green_jpg - red_jpg) / (green_jpg + red_jpg + eps))
        indices['VARI'] = np.nanmean((green_jpg - red_jpg) / (green_jpg + red_jpg - blue_jpg + eps))
        indices['GLI'] = np.nanmean((2 * green_jpg - red_jpg - blue_jpg) / (2 * green_jpg + red_jpg + blue_jpg + eps))
      #  indices['vNDVI'] = np.nanmean((nir_patch - red_jpg) / (nir_patch + red_jpg + eps))
        indices['RGBVI'] = np.nanmean((green_jpg**2 - red_jpg * blue_jpg) / (green_jpg**2 + red_jpg * blue_jpg + eps))
        indices['MGRVI'] = np.nanmean((green_jpg**2 - red_jpg**2) / (green_jpg**2 + red_jpg**2 + eps))

        indices_list.append(indices)
    return indices_list

def load_images_from_scene(scene_files, target_size=None):
    """Load multispectral images from a scene and ensure they are the same size."""
    images = {}
    # print("Loading images for scene:")
    for file in scene_files:
        #print(f" Loaded image:  {file}")
        if file.endswith(".JPG"):
            img = cv2.imread(file)
            if img is None:
                print(f"Failed to load: {file}")
            else:
                #print(" Load Rgb image to dic: ", file)
                images['RGB'] = img
        elif file.endswith(".TIF"):
            
            # This line extracts a character from the file name. The notation file[-5] means
            #  "take the fifth character from the end of the filename string."
            # Purpose: It assumes that this character represents the band number.
            # Example: If the file is named "image3.TIF", the character at position -5 might be '3'.

            band_index = file[-5]
            band_mapping = {'1': 'B', '2': 'G', '3': 'R', '4': 'E', '5': 'N'}
            if band_index in band_mapping:
                band_name = band_mapping[band_index]

                # Here, the code looks up the corresponding band identifier from the dictionary.
                # Example: If band_index is '3', then band_name becomes 'R' (Red band).

                band_name = band_mapping[band_index]
                img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"Failed to load: {file}")
                else:
                    # print(" \n Load TIF image to dic: ", file)
                    images[band_name] = img

    if len(images) != 6:
        print(f"Warning: Incomplete image set: {scene_files}")
        return None

    if target_size is None:
        target_size = list(images.values())[0].shape[:2]

    for key in images:
        if images[key].shape[:2] != target_size:
            images[key] = cv2.resize(images[key], (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

    # Images is a dictoniary of the 6 iamges as cv2 files. 
    # Each band and the rgb has an index 
    return images

def normalize_images(images, method='min-max'):
    """Normalize images using min-max scaling or z-score."""
    normalized = {}
    for key, img in images.items():
        img = img.astype(np.float32)
        if method == 'min-max':
            img_min, img_max = img.min(), img.max()
            normalized[key] = (img - img_min) / (img_max - img_min + 1e-6)
        elif method == 'z-score':
            img_mean, img_std = img.mean(), img.std()
            normalized[key] = (img - img_mean) / (img_std + 1e-6)
    return normalized

def downsample_images(images, factor=2):
    """Downsample images by a given factor."""
    downsampled = {}
    for key, img in images.items():
        new_size = (img.shape[1] // factor, img.shape[0] // factor)
        downsampled[key] = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return downsampled

def apply_mean_convolution(images, kernel_size=5):
    """Apply mean convolution with given kernel size and stride."""
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    convolved = {}
    for key, img in images.items():
        if img.ndim == 3:  # For RGB images
            convolved_img = np.zeros_like(img)
            for i in range(3):
                convolved_img[..., i] = convolve(img[..., i], kernel, mode='reflect')
            convolved[key] = convolved_img
        else:
            convolved[key] = convolve(img, kernel, mode='reflect')
    return convolved

def process_all_subdirectories(base_directory):
    """Process all subdirectories and save results to CSVs."""
    all_pixel_results = []
    all_patch_results = []

    for root, dirs, files in os.walk(base_directory):
        print(dirs)
        if not files:
            continue

        # Collect all relevant image files in a clear way.
        all_files = []
        for f in files:
            if f.endswith('.JPG') or f.endswith('.TIF'):
                all_files.append(os.path.join(root, f))
        all_files = sorted(all_files)

       # print("All files ", all_files)
       # print("\n \n")
        
        if len(all_files) < 6:
            print(f"Skipping {root} due to insufficient images.")
            continue

        # Group files into scenes of 6 files each.
        all_scenes = []    # List of scenes - scene is one group of 6 pictures, starting with the JPG
        for i in range(0, len(all_files), 6):
            scene_of_six_files = all_files[i:i+6]   # group of 6 pictures, starting with the JPG
            #print("\n\n Scene files : ", scene_of_six_files)
            if len(scene_of_six_files) != 6:
                continue
            if not scene_of_six_files[0].endswith('.JPG'):
                continue
            all_scenes.append(scene_of_six_files)
            #print("\n\n\n Scenes \n : ", all_scenes)
        print(f"Total valid scenes in {root}: {len(all_scenes)}")

        target_size = None
        # Process all scenes in groups of 6
        for scene_of_six_files in all_scenes:
            
            images = load_images_from_scene(scene_of_six_files, target_size)  # Images is a dictoniary of the 6 iamges as cv2 files. 
                                                                                # Each band and the rgb has an index 
           # print(" \n Images : ",len(images) )
           # print("\n \n")
            if images:
                # Preprocess (e.g., normalize, downsample)
                images = normalize_images(images)

                # Check if the folder is 100FPLAN or 101FPLAN (pixel-based analysis)
                folder_name = os.path.basename(root)
                if folder_name in ['100FPLAN', '101FPLAN']:
                    continue
                    images = downsample_images(images, factor=8)

                    pixel_indices = compute_indices_for_pixels(images)
                    jpg_pixel_indices = compute_indices_for_pixels_from_jpg(images)

                    for y in range(pixel_indices['NDVI'].shape[0]):
                        for x in range(pixel_indices['NDVI'].shape[1]):
                            pixel_data = {
                                'Image': os.path.basename(scene_of_six_files[0]),
                                'X': x,
                                'Y': y,
                                'NDVI': pixel_indices['NDVI'][y, x],
                                'NGRDI': pixel_indices['NGRDI'][y, x],
                                'VARI': pixel_indices['VARI'][y, x],
                                'GLI': pixel_indices['GLI'][y, x],
                                'vNDVI': pixel_indices['vNDVI'][y, x],
                                'RGBVI': pixel_indices['RGBVI'][y, x],
                                'MGRVI': jpg_pixel_indices['MGRVI'][y,x],
                                'jpg_NDVI': jpg_pixel_indices['NDVI'][y, x],
                                'jpg_NGRDI': jpg_pixel_indices['NGRDI'][y, x],
                                'jpg_VARI': jpg_pixel_indices['VARI'][y, x],
                                'jpg_GLI': jpg_pixel_indices['GLI'][y, x],
                                #'jpg_vNDVI': jpg_pixel_indices['vNDVI'][y, x],
                                'jpg_RGBVI': jpg_pixel_indices['RGBVI'][y, x],
                                'jpg_MGRVI': jpg_pixel_indices['MGRVI'][y,x]

                            }
                            all_pixel_results.append(pixel_data)
                else:
                    # For patch-based analysis.
                    images = downsample_images(images, factor=8)

                    # Apply convolution with stride = kernel size.
                    kernel_size = 5
                    images = apply_mean_convolution(images, kernel_size=kernel_size)

                    patch_size = 64
                    patches = {}
                    for band, img in images.items():
                        patches[band] = split_into_patches(img, patch_size)

                    # Compute multispectral indices for each patch.
                    patch_indices = compute_indices_for_patches(patches)
                    # Compute JPG-based indices for each patch.
                    jpg_patch_indices = compute_indices_for_patches_from_jpg(patches)

                    for idx, indices in enumerate(patch_indices):
                        indices['Scene'] = os.path.basename(scene_of_six_files[0])
                        indices['Patch'] = idx
                        indices['jpg_NDVI'] = jpg_patch_indices[idx]['NDVI']
                        indices['jpg_NGRDI'] = jpg_patch_indices[idx]['NGRDI']
                        indices['jpg_VARI'] = jpg_patch_indices[idx]['VARI']
                        indices['jpg_GLI'] = jpg_patch_indices[idx]['GLI']
                        #indices['jpg_vNDVI'] = jpg_patch_indices[idx]['vNDVI']
                        indices['jpg_RGBVI'] = jpg_patch_indices[idx]['RGBVI']
                        all_patch_results.append(indices)
                    break
    # Save pixel results to CSV.
    df_pixel = pd.DataFrame(all_pixel_results)
    df_pixel.to_csv(os.path.join(base_directory, 'veg_indices_perpixel_kazachstan.csv'), index=False)
    print("Pixel results saved to veg_indices_perpixel_kazachstan.csv")

    # Save patch results to CSV.
    df_patch = pd.DataFrame(all_patch_results)
    df_patch.to_csv(os.path.join(base_directory, 'vegetation_indices_patched_kazachstan.csv'), index=False)
    print("Patch results saved to vegetation_indices_patched_kazachstan.csv")

    return df_pixel, df_patch


# ---------------------
# New function for the new file structure
# ---------------------
def process_dataset_new_structure(base_directory, subset="Train_Images"):
    """
    Process the new dataset structure:
      base_directory/
         RGB_Images/{subset}/...
         Spectral_Images/Green_Channel/{subset}/...
         Spectral_Images/Near_Infrared_Channel/{subset}/...
         Spectral_Images/Red_Channel/{subset}/...
         Spectral_Images/Red_Edge_Chanel/{subset}/...
    A scene is defined by images with the same filename.
    """
    all_pixel_results = []
    
    # Define folder paths.
    rgb_folder = os.path.join(base_directory, "RGB_Images", subset)
    red_folder = os.path.join(base_directory, "Spectral_Images", "Red_Channel", subset)
    green_folder = os.path.join(base_directory, "Spectral_Images", "Green_Channel", subset)
    red_edge_folder = os.path.join(base_directory, "Spectral_Images", "Red_Edge_Channel", subset)
    nir_folder = os.path.join(base_directory, "Spectral_Images", "Near_Infrared_Channel", subset)
    
    if not os.path.isdir(rgb_folder):
        print(f"RGB folder not found: {rgb_folder}")
        return None

    # List all RGB images (assumed to be the reference for a scene).
    rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.lower().endswith(('.jpg', '.png'))])
    print(f"Found {len(rgb_files)} RGB images in {rgb_folder} for subset {subset}.")
    
    for file_name in rgb_files:
        # Build full paths for each channel based on the same filename.
        rgb_path = os.path.join(rgb_folder, file_name)
        red_path = os.path.join(red_folder, file_name)
        green_path = os.path.join(green_folder, file_name)
        red_edge_path = os.path.join(red_edge_folder, file_name)
        nir_path = os.path.join(nir_folder, file_name)
       # print(rgb_path + "\n" + red_path + "\n" + green_path + "\n" + red_edge_path + "\n" + nir_path)
        
        # Check that all files exist.
        if not (os.path.exists(rgb_path) and os.path.exists(red_path) and 
                os.path.exists(green_path) and os.path.exists(red_edge_path) and 
                os.path.exists(nir_path)):
            print(f"Skipping {file_name}: not all channels are present.")
            continue
        
        # Load the images.
        rgb_img = cv2.imread(rgb_path)  # Will be in BGR order.
        red_img = cv2.imread(red_path, cv2.IMREAD_UNCHANGED)
        green_img = cv2.imread(green_path, cv2.IMREAD_UNCHANGED)
        red_edge_img = cv2.imread(red_edge_path, cv2.IMREAD_UNCHANGED)
        nir_img = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)
        
        if rgb_img is None or red_img is None or green_img is None or red_edge_img is None or nir_img is None:
            print(f"Skipping {file_name}: failed to load one or more images.")
            continue
        
        # For the Blue channel (not provided spectrally), extract from the RGB image.
        blue_img = rgb_img[..., 0]  # Since cv2.imread returns BGR, channel 0 is Blue.
        
        # Assemble the images dictionary.
        images = {
            'RGB': rgb_img,
            'R': red_img,
            'G': green_img,
            'E': red_edge_img,
            'N': nir_img,
            'B': blue_img
        }
        
        # Validate and equalize image sizes (using the RGB image as the target).
        target_size = rgb_img.shape[:2]
        for key in images:
            if images[key].shape[:2] != target_size:
                images[key] = cv2.resize(images[key], (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Normalize and optionally downsample (here we use a downsample factor of 8, as before).
        images = normalize_images(images)
        images = downsample_images(images, factor=4)
        
        # Compute vegetation indices.
        pixel_indices = compute_indices_for_pixels(images)
        jpg_pixel_indices = compute_indices_for_pixels_from_jpg(images)
        
        h, w = pixel_indices['NDVI'].shape
        for y in range(h):
            for x in range(w):
                pixel_data = {
                    'Image': file_name,
                    'X': x,
                    'Y': y,
                    'NDVI': pixel_indices['NDVI'][y, x],
                    'NGRDI': pixel_indices['NGRDI'][y, x],
                    'VARI': pixel_indices['VARI'][y, x],
                    'GLI': pixel_indices['GLI'][y, x],
                    'vNDVI': pixel_indices['vNDVI'][y, x],
                    'RGBVI': pixel_indices['RGBVI'][y, x],
                    'MGRVI': pixel_indices['MGRVI'][y,x],
                    'jpg_NDVI': jpg_pixel_indices['NDVI'][y, x],
                    'jpg_NGRDI': jpg_pixel_indices['NGRDI'][y, x],
                    'jpg_VARI': jpg_pixel_indices['VARI'][y, x],
                    'jpg_GLI': jpg_pixel_indices['GLI'][y, x],
                    #'jpg_vNDVI': jpg_pixel_indices['vNDVI'][y, x],
                    'jpg_RGBVI': jpg_pixel_indices['RGBVI'][y, x],
                    'jpg_MGRVI': jpg_pixel_indices['MGRVI'][y,x]
                }
                all_pixel_results.append(pixel_data)
    
    # Save pixel-level results to CSV.
    output_csv = os.path.join(base_directory, f'veg_indices_perpixel_new_structure_{subset}.csv')
    df_pixel = pd.DataFrame(all_pixel_results)
    df_pixel.to_csv(output_csv, index=False)
    print(f"Pixel results saved to {output_csv}")
    

    return df_pixel


if __name__ == "__main__":
    # Original function call remains available:
    base_dir = r'D:/MRc/FIIT/DP_Model/Datasets/kazachstan_multispectral_UAV/filght_session_02/2022-06-09'
    df_pixel = process_all_subdirectories(base_dir)
    
    # New file structure base directory:
    new_base_dir = r'D:/MRc/FIIT/DP_Model/Datasets/Multispectral_images_dataset'
    new_base_dir = new_base_dir.replace('\\', '/')
    
    # # Process Train images:
    # df_pixel_train = process_dataset_new_structure(new_base_dir, subset="Train_images")
    # # Process Test images:
    # df_pixel_test = process_dataset_new_structure(new_base_dir, subset="Test_Images")