import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.feature import canny
from scipy.ndimage import median_filter

def detect_contours(image, method="sobel"):
    """
    Détecte les contours d'une image avec Sobel ou Canny.
    
    :param image: ndarray, image en niveaux de gris normalisée [0, 1].
    :param method: str, "sobel" ou "canny" pour choisir la méthode.
    :return: ndarray, image des contours.
    """
    if method == "sobel":
        return sobel(image)
    elif method == "canny":
        return canny(image, sigma=1)  # Paramètre sigma ajustable
    else:
        raise ValueError("Méthode inconnue. Utilisez 'sobel' ou 'canny'.")

def filtrage_adaptatif(image, contours):
    """
    Applique un filtrage adaptatif en utilisant une carte de contours.
    
    :param image: ndarray, image originale en niveaux de gris.
    :param contours: ndarray, image des contours.
    :return: ndarray, image filtrée.
    """
    # Appliquer un filtre médian sur les régions avec contours
    filtered_image = image.copy()
    filtered_image[contours > 0] = median_filter(image, size=3)[contours > 0]
    return filtered_image

def detecter_contours_et_filtrer():
    """
    Charge une image, détecte les contours et applique un filtrage adaptatif.
    
    :param image_path: str, chemin de l'image à traiter.
    """
    from utils import load_image  # Fonction pour charger l'image

    image_path = "data/image_reference1.png" 
    image = load_image(image_path, as_gray=True)

    # Détection des contours
    contours_sobel = detect_contours(image, method="sobel")
    contours_canny = detect_contours(image, method="canny")

    # Afficher les résultats
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Image originale")
    ax[1].imshow(contours_sobel, cmap="gray")
    ax[1].set_title("Contours Sobel")
    ax[2].imshow(contours_canny, cmap="gray")
    ax[2].set_title("Contours Canny")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.show()
