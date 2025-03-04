import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import color, exposure

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def resize_to_match(image, reference):
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC)

def save_image(image, folder, filename):
    output_path = os.path.join(folder, filename)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")

def normalize_image(image):
    """Normalize pixel values to the range [0, 1]."""
    return image.astype(np.float32) / 255.0

def convert_to_xyz(image):
    return (color.rgb2xyz(image) * 255).astype(np.uint8)

def histogram_matching(source, reference):
    return exposure.match_histograms(source, reference, channel_axis=-1)

def calculate_indices(rgb, nir):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    nir = nir[:, :, 0] if len(nir.shape) == 3 else nir
    
    ndvi = (nir - r) / (nir + r + 1e-6)
    ngrdi = (g - r) / (g + r + 1e-6)
    vari = (g - r) / (g + r - b + 1e-6)
    gli = (2 * g - r - b) / (2 * g + r + b + 1e-6)
    vndvi = (g - r) / (g + r + 1e-6)
    rgbvi = (g**2 - b * r) / (g**2 + b * r + 1e-6)
    mgrvi = (g**2 - r**2) / (g**2 + r**2 + 1e-6)
    
    return {"NDVI": ndvi, "NGRDI": ngrdi, "VARI": vari, "GLI": gli, "vNDVI": vndvi, "RGBVI": rgbvi, "MGRVI": mgrvi}

def save_colormap(index, output_folder, filename):
    plt.imshow(index, cmap='RdYlGn')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
    plt.close()

def process_images(image_pairs, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # reference_rgb = load_image(image_pairs[0]["rgb"])
    # reference_rgb = normalize_image(reference_rgb)  # Normalize reference image
    results = []
    
    for idx, pair in enumerate(image_pairs):
        print(f"Processing pair {idx + 1}...")
        rgb = load_image(pair["rgb"])
        nir = load_image(pair["nir"])
        
        # Normalize RGB and NIR images
        rgb = normalize_image(rgb)
        nir = normalize_image(nir)
        
        # Resize NIR to match RGB dimensions
        nir = resize_to_match(nir, rgb)
        
        # Calculate indices for original RGB-NIR pair
        original_indices = calculate_indices(rgb, nir)
        
        # Convert RGB to XYZ color space
        xyz = convert_to_xyz(rgb)
        xyz = normalize_image(xyz)  # Normalize XYZ image
        xyz_indices = calculate_indices(xyz, nir)
        
        # Perform histogram matching
       # matched_rgb = histogram_matching(rgb, reference_rgb)
        # matched_indices = calculate_indices(matched_rgb, nir)
        
        # Save colormaps for each index
        for key in original_indices.keys():
            save_colormap(original_indices[key], output_folder, f"{key}_original_{idx+1}.png")
            save_colormap(xyz_indices[key], output_folder, f"{key}_xyz_{idx+1}.png")
            #save_colormap(matched_indices[key], output_folder, f"{key}_matched_{idx+1}.png")
            
            # Save mean values of indices
            results.append({
                "Image": idx + 1,
                "Index": key,
                "Original": np.mean(original_indices[key]),
                "XYZ": np.mean(xyz_indices[key]),
                #"Matched": np.mean(matched_indices[key])
            })
    
    # Save results to CSV
    # pd.DataFrame(results).to_csv(os.path.join(output_folder, "vegetation_indices.csv"), index=False)

# Define image pairs
image_pairs = [
    {"rgb": "D:\\MRc\\FIIT\\DP_Model\\Datasets\\kazachstan_multispectral_UAV\\filght_session_02\\2022-06-09\\NNdataset\\100FPLAN_DJI_0010.JPG", 
     "nir": "D:\\MRc\\FIIT\\DP_Model\\Datasets\\kazachstan_multispectral_UAV\\filght_session_02\\2022-06-09\\NNdataset\\100FPLAN_DJI_0010.TIF"},
    # {"rgb": "D:/MRc/FIIT/DP_Model/Datasets/kazachstan_multispectral_UAV/filght_session_02/2022-06-08/101FPLAN/DJI_0010.JPG", 
    #  "nir": "D:/MRc/FIIT/DP_Model/Datasets/kazachstan_multispectral_UAV/filght_session_02/2022-06-08/101FPLAN/DJI_0015.TIF"},
    # {"rgb": "D:/MRc/FIIT/DP_Model/Datasets/Multispectral_images_dataset/RGB_Images/Train_Images/Image_027.jpg", 
    #  "nir": "D:/MRc/FIIT/DP_Model/Datasets/Multispectral_images_dataset/Spectral_Images/Near_Infrared_Channel/Train_Images/Image_027.jpg"}
]

# Output folder
output_folder = "D:/MRc/FIIT/DP_Model/corrected_images"
process_images(image_pairs, output_folder)