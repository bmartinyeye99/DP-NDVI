
# Helper functions
def compute_rgb_indices(rgb, eps=1e-6):
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ngrdi = (G - R) / (G + R + eps)
    vari  = (G - R) / (G + R - B + eps)
    gli   = (2 * G - R - B) / (2 * G + R + B + eps)
    return {'NGRDI': ngrdi, 'VARI': vari, 'GLI': gli}

def compute_ndvi(nir, red, eps=1e-6):
    return (nir - red) / (nir + red + eps)

