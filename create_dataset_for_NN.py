import os
import shutil
import cv2


def downsample_images(images, target_size=(256, 256)):
    """Downsample images to a fixed target size (256x256 by default)."""
    downsampled = {}
    for key, img in images.items():
        downsampled[key] = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    return downsampled


def process_all_subdirectories(base_directory):
    # Define the target directory
    target_directory = os.path.join(base_directory, 'NNdataset')
    
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_directory):
        # Sort files to ensure correct order
        files.sort()
        
        # Iterate through the files
        i = 0
        while i < len(files):
            if files[i].lower().endswith('.jpg'):
                jpg_file = files[i]
                jpg_path = os.path.join(root, jpg_file)
                
                # Check if there are at least 5 TIF files after the JPG
                if i + 5 < len(files) and all(files[i + j + 1].lower().endswith('.tif') for j in range(5)):
                    # The 5th TIF file is the NIR file
                    nir_file = files[i + 5]
                    nir_path = os.path.join(root, nir_file)
                    
                    # Get the root folder name to avoid naming duplicates
                    root_folder_name = os.path.basename(root)
                    
                    # Create new names for the JPG and NIR files
                    new_jpg_name = f"{root_folder_name}_{jpg_file}"
                    new_nir_name = f"{root_folder_name}_{jpg_file.replace('JPG','TIF')}"
                    
                    # Define the new paths in the target directory
                    new_jpg_path = os.path.join(target_directory, new_jpg_name)
                    new_nir_path = os.path.join(target_directory, new_nir_name)
                    
                    # Read the JPG and NIR images
                    jpg_image = cv2.imread(jpg_path)
                    nir_image = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)  # Read TIF as is
                    
                    # Downsample the images
                    downsampled_images = downsample_images({
                        'jpg': jpg_image,
                        'nir': nir_image
                    })
                    
                    # Save the downsampled images
                    cv2.imwrite(new_jpg_path, downsampled_images['jpg'])
                    cv2.imwrite(new_nir_path, downsampled_images['nir'])
                    
                    # print(f"Downsampled and saved {new_jpg_name} and {new_nir_name} to {target_directory}")
                
                # Move to the next JPG file
                i += 6
            else:
                i += 1


# Define the flight session directory
flight_session = r'D:/MRc/FIIT/DP_Model/Datasets/kazachstan_multispectral_UAV/filght_session_02/2022-06-09'

# Process all subdirectories
process_all_subdirectories(flight_session)