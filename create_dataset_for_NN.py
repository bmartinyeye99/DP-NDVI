import os
import cv2
from PIL import Image
import os
import numpy as np

# Get the number of existing scenes in the output directories
def count_existing_scenes(output_dir):
    # List all files in the output directory
    files = os.listdir(output_dir)
    
    # Extract unique scene numbers from filenames
    scene_numbers = set()
    for filename in files:
        if  filename.endswith(".jpg"):
            # Extract the number from the filename
            number = filename.split("_")[0]
            scene_numbers.add(number)
        elif filename.endswith(".TIF"):
            # Extract the number from the filename
            number = filename.split("_")[0]
            scene_numbers.add(number)
    
    # Return the number of unique scenes
    return len(scene_numbers)



def remove_black_background(image):
    # Convert the image to RGBA if it's not already
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    numpydata = np.array(image)
    gray = cv2.cvtColor(numpydata, cv2.COLOR_RGB2GRAY)

    # Threshold the image to create a binary mask
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # Adjust threshold value (10)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return the original image
    if not contours:
        return image

    # Get the bounding box of the largest contour
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Crop the image to the bounding box
    cropped_img = numpydata[y:y + h, x:x + w]

    return Image.fromarray(cropped_img)


def split_and_save_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Get the number of existing images in the output directories
    num_scenes_in_dir = count_existing_scenes(output_dir)

    # Iterate through all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            width, height = img.size

            # Split the image into left and right halves
            left_half = img.crop((0, 0, width // 2, height))
            right_half = img.crop((width // 2, 0, width, height))

            left_half = remove_black_background(left_half)
            right_half = remove_black_background(right_half)

            # # Downsample the images
            # downsampled_images = downsample_images({
            #     'jpg': left_half,
            #     'nir': right_half
            # })

            # Convert RGBA to RGB before saving as JPEG
            rgb_image = left_half.convert('RGB')
            nir_image = right_half.convert('RGB')

            # Save the left half as RGB image
            rgb_filename = f"{num_scenes_in_dir}_Scene_RGB.jpg"
            rgb_path = os.path.join(output_dir, rgb_filename)
            rgb_image.save(rgb_path)

            # Save the right half as NIR image
            nir_filename = f"{num_scenes_in_dir}_Scene_NIR.TIF"
            nir_path = os.path.join(output_dir, nir_filename)
            nir_image.save(nir_path)

            # Increment the counter
            num_scenes_in_dir += 1


def downsample_images(images, target_size=(256, 256)):
    """Downsample images to a fixed target size (256x256 by default)."""
    downsampled = {}
    for key, img in images.items():
        # Check if the input is a PIL Image
        if isinstance(img, Image.Image):
            # Convert PIL Image to NumPy array
            img_np = np.array(img)
        # Check if the input is already a NumPy array
        elif isinstance(img, np.ndarray):
            img_np = img
        else:
            raise TypeError(f"Unsupported image type: {type(img)}. Expected PIL.Image or numpy.ndarray.")

        # Resize using OpenCV
        img_resized = cv2.resize(img_np, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        # Convert back to PIL Image
        downsampled[key] = Image.fromarray(img_resized)
    return downsampled



def copy_to_target_dir():
    target_directory = 'D:\MRc\FIIT\DP_Model\pix2pixWin\dataset'
    base_directory = r'D:\MRc\FIIT\DP_Model\Datasets\Multispectral_images_dataset'
    num_scenes_in_dir = count_existing_scenes(target_directory)

    rgb_folder = os.path.join(base_directory, "RGB_Images", "Train_Images")
    nir_folder = os.path.join(base_directory, "Spectral_Images", "Near_Infrared_Channel", "Train_Images")
    
    if not os.path.isdir(rgb_folder):
        print(f"RGB folder not found: {rgb_folder}")
        return None

    # List all RGB images (assumed to be the reference for a scene).
    rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.lower().endswith(('.jpg', '.png'))])
    print(f"Found {len(rgb_files)} RGB images in {rgb_folder} ")
    
    for file_name in rgb_files:
        rgb_path = os.path.join(rgb_folder, file_name)
        nir_path = os.path.join(nir_folder, file_name)
        
        # Check that all files exist.
        if not (os.path.exists(rgb_path) and 
                os.path.exists(nir_path)):
            print(f"Skipping {file_name}: not all channels are present.")
            continue
        
        # Load the images.
        rgb_img = cv2.imread(rgb_path)  # Will be in BGR order
        nir_img = cv2.imread(nir_path)
        
        if rgb_img is None or nir_img is None:
            print(f"Skipping {file_name}: failed to load one or more images.")
            continue
        
        # Downsample the images
        downsampled_images = downsample_images({
            'jpg': rgb_img,
            'nir': nir_img
        })
        
        new_jpg_path = os.path.join(target_directory, f'{num_scenes_in_dir}_Scene_RGB.jpg')
        new_nir_path = os.path.join(target_directory, f'{num_scenes_in_dir}_Scene_NIR.TIF')

       # Save the downsampled images
        cv2.imwrite(new_jpg_path, np.array(downsampled_images['jpg']))
        cv2.imwrite(new_nir_path, np.array(downsampled_images['nir']))

        num_scenes_in_dir += 1



def process_all_subdirectories_kazachstan(base_directory):
    # Define the target directory
    target_directory = 'D:\MRc\FIIT\DP_Model\pix2pixWin\dataset'
    
    num_scenes_in_dir = count_existing_scenes(target_directory)

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
                    
                    # Create new names for the JPG and NIR files
                    new_jpg_name = f"{num_scenes_in_dir}_Scene_RGB.jpg"
                    new_nir_name = f"{num_scenes_in_dir}_Scene_NIR.TIF"
                    
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
                    cv2.imwrite(new_jpg_path, np.array(downsampled_images['jpg']))
                    cv2.imwrite(new_nir_path, np.array(downsampled_images['nir']))

                    num_scenes_in_dir += 1
                    
                    # print(f"Downsampled and saved {new_jpg_name} and {new_nir_name} to {target_directory}")
                
                # Move to the next JPG file
                i += 6
            else:
                i += 1


# Define the flight session directory
flight_session = r'D:/MRc/FIIT/DP_Model/Datasets/kazachstan_multispectral_UAV/filght_session_02/2022-06-09'
rgb_nir_to_split = r'D:\MRc\FIIT\DP_Model\Datasets\RGB-NIR-dataset\nirscene0\jpg1'

output_directory = r'D:\MRc\FIIT\DP_Model\pix2pixWin\dataset'
#output_directory = r'D:\MRc\FIIT\DP_Model\Datasets\RGB-NIR-dataset\nirscene0\Splitted'


# Process all subdirectories
process_all_subdirectories_kazachstan(flight_session)
split_and_save_images(rgb_nir_to_split, output_directory)
copy_to_target_dir()