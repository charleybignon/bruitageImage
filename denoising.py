import numpy as np

def denoise_image_median(image, filter_size):
    rows, cols = image.shape
    offset = filter_size // 2
    padded_image = np.pad(image, pad_width=offset, mode='reflect')
    denoised_image = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i + filter_size, j:j + filter_size]
            denoised_image[i, j] = np.median(region)
    return denoised_image

def convolve_image(image, kernel):
    rows, cols = image.shape
    krows, kcols = kernel.shape
    if krows % 2 == 0 or kcols % 2 == 0:
        raise ValueError("Les dimensions du noyau doivent être impaires.")
    
    pad_rows = krows // 2
    pad_cols = kcols // 2
    padded_image = np.pad(image, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='reflect')
    convolved_image = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i + krows, j:j + kcols]
            convolved_image[i, j] = np.sum(region * kernel)
    return convolved_image
