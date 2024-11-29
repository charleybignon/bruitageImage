import matplotlib.pyplot as plt
from snr import calculate_snr, calculate_snr_gain
from noise import add_noise_additive, add_noise_salt_pepper, add_noise_multiplicative
from denoising import denoise_image_median, convolve_image
from utils import load_image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Charger l'image de référence
    image_path = "data/image_reference1.png" 
    image_reference = load_image(image_path, as_gray=True)

    # Charger plusieurs images bruitées et calculer leurs SNR
    images_bruitees = [
        {"filename": "data/image1_bruitee_snr_41.8939.png", "expected_snr": 41.8939},
        {"filename": "data/image1_bruitee_snr_36.1414.png", "expected_snr": 36.1414},
        {"filename": "data/image1_bruitee_snr_32.6777.png", "expected_snr": 32.6777},
        {"filename": "data/image1_bruitee_snr_28.2378.png", "expected_snr": 28.2378},
        {"filename": "data/image1_bruitee_snr_22.2912.png", "expected_snr": 22.2912},
        {"filename": "data/image1_bruitee_snr_16.4138.png", "expected_snr": 16.4138},
        {"filename": "data/image1_bruitee_snr_13.0913.png", "expected_snr": 13.0913},
        {"filename": "data/image1_bruitee_snr_10.8656.png", "expected_snr": 10.8656},
        {"filename": "data/image1_bruitee_snr_9.2885.png", "expected_snr": 9.2885},
    ]

    results = []
    for image in images_bruitees:
        noisy_image = load_image(image["filename"], as_gray=True)
        snr_calculated = calculate_snr(image_reference, noisy_image)
        snr_difference = abs(snr_calculated - image["expected_snr"])
        
        # Vérifier si la différence de SNR est proche de zéro
        if snr_difference < 0.001:
            message = "Calcul du SNR validé"
        else:
            message = f"Différence de SNR: {snr_difference:.4f}"
        
        results.append({
            "Nom du fichier": image["filename"],
            "SNR attendu": image["expected_snr"],
            "SNR calculé": snr_calculated,
            "Différence": snr_difference,
            "Message": message,
        })

    df_comparison = pd.DataFrame(results)
    print(df_comparison)

    # Ajouter du bruit et calculer le SNR
    noisy_gaussian = add_noise_additive(image_reference, var=0.10)
    snr_gaussian = calculate_snr(image_reference, noisy_gaussian)

    noisy_salt_pepper = add_noise_salt_pepper(image_reference, amount=0.15)
    snr_salt_pepper = calculate_snr(image_reference, noisy_salt_pepper)

    noisy_multiplicative = add_noise_multiplicative(image_reference,var=0.10)
    snr_multiplicative = calculate_snr(image_reference, noisy_multiplicative)

    # Afficher les images bruitées
    fig, ax = plt.subplots(1, 4, figsize=(18, 6))
    ax[0].imshow(image_reference, cmap='gray')
    ax[0].set_title("Image de référence")
    ax[0].axis('off')

    ax[1].imshow(noisy_gaussian, cmap='gray')
    ax[1].set_title(f"Image bruitée (Gaussien, SNR={snr_gaussian:.2f})")
    ax[1].axis('off')

    ax[2].imshow(noisy_salt_pepper, cmap='gray')
    ax[2].set_title(f"Image bruitée (Sel et Poivre, SNR={snr_salt_pepper:.2f})")
    ax[2].axis('off')

    ax[3].imshow(noisy_multiplicative, cmap='gray')
    ax[3].set_title(f"Image bruitée (multiplicatif, SNR={snr_multiplicative:.2f})")
    ax[3].axis('off')

    plt.tight_layout()
    plt.show()

    # Charger la nouvelle image de référence
    image_path = "data/lenaNB.tiff" 
    image_reference = load_image(image_path, as_gray=True)

    # Ajouter du bruit avec une variance élevée
    noisy_image_high_noise = add_noise_salt_pepper(image_reference, amount=0.05)

    # Calculer le SNR de l'image bruitée
    snr_high_noise = calculate_snr(image_reference, noisy_image_high_noise)

    # Débruiter avec le filtre médian
    denoised_median_high_noise = denoise_image_median(noisy_image_high_noise, filter_size=3)
    snr_denoised_median_high_noise = calculate_snr(image_reference, denoised_median_high_noise)

    # Débruiter avec la convolution
    kernel = np.ones((5, 5)) / 25
    denoised_convolution_high_noise = convolve_image(noisy_image_high_noise, kernel)
    snr_denoised_convolution_high_noise = calculate_snr(image_reference, denoised_convolution_high_noise)

    # Afficher les images côte à côte
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    # Image de référence
    ax[0].imshow(image_reference, cmap='gray')
    ax[0].set_title("Image de référence")
    ax[0].axis('off')

    # Image bruitée
    ax[1].imshow(noisy_image_high_noise, cmap='gray')
    ax[1].set_title(f"Bruitée (s&p) (SNR={snr_high_noise:.2f})")
    ax[1].axis('off')

    # Image débruitée (Médian)
    ax[2].imshow(denoised_median_high_noise, cmap='gray')
    ax[2].set_title(f"Débruitée Médian (SNR={snr_denoised_median_high_noise:.2f})")
    ax[2].axis('off')

    # Image débruitée (Convolution)
    ax[3].imshow(denoised_convolution_high_noise, cmap='gray')
    ax[3].set_title(f"Débruitée Convolution (SNR={snr_denoised_convolution_high_noise:.2f})")
    ax[3].axis('off')

    # Afficher les résultats
    plt.tight_layout()
    plt.show()

    # Ajouter du bruit additif avec une variance élevée
    noisy_image_high_noise = add_noise_additive(image_reference, var=0.01)

    # Calculer le SNR de l'image bruitée
    snr_high_noise = calculate_snr(image_reference, noisy_image_high_noise)

    # Débruiter avec le filtre médian
    denoised_median_high_noise = denoise_image_median(noisy_image_high_noise, filter_size=3)
    snr_denoised_median_high_noise = calculate_snr(image_reference, denoised_median_high_noise)

    # Débruiter avec la convolution
    kernel = np.ones((5, 5)) / 25
    denoised_convolution_high_noise = convolve_image(noisy_image_high_noise, kernel)
    snr_denoised_convolution_high_noise = calculate_snr(image_reference, denoised_convolution_high_noise)

    # Afficher les images côte à côte
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    # Image de référence
    ax[0].imshow(image_reference, cmap='gray')
    ax[0].set_title("Image de référence")
    ax[0].axis('off')

    # Image bruitée
    ax[1].imshow(noisy_image_high_noise, cmap='gray')
    ax[1].set_title(f"Bruitée (additif) (SNR={snr_high_noise:.2f})")
    ax[1].axis('off')

    # Image débruitée (Médian)
    ax[2].imshow(denoised_median_high_noise, cmap='gray')
    ax[2].set_title(f"Débruitée Médian (SNR={snr_denoised_median_high_noise:.2f})")
    ax[2].axis('off')

    # Image débruitée (Convolution)
    ax[3].imshow(denoised_convolution_high_noise, cmap='gray')
    ax[3].set_title(f"Débruitée Convolution (SNR={snr_denoised_convolution_high_noise:.2f})")
    ax[3].axis('off')

    # Afficher les résultats
    plt.tight_layout()
    plt.show()

    # Ajouter du bruit multiplicatif avec une variance élevée
    noisy_image_multiplicative = add_noise_multiplicative(image_reference, var=0.05)

    # Calculer le SNR de l'image bruitée
    snr_multiplicative = calculate_snr(image_reference, noisy_image_multiplicative)

    # Débruiter avec le filtre médian
    denoised_median_multiplicative = denoise_image_median(noisy_image_multiplicative, filter_size=3)
    snr_denoised_median_multiplicative = calculate_snr(image_reference, denoised_median_multiplicative)

    # Débruiter avec la convolution
    kernel = np.ones((5, 5)) / 25
    denoised_convolution_multiplicative = convolve_image(noisy_image_multiplicative, kernel)
    snr_denoised_convolution_multiplicative = calculate_snr(image_reference, denoised_convolution_multiplicative)

    # Afficher les images côte à côte
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    # Image de référence
    ax[0].imshow(image_reference, cmap='gray')
    ax[0].set_title("Image de référence")
    ax[0].axis('off')

    # Image bruitée
    ax[1].imshow(noisy_image_multiplicative, cmap='gray')
    ax[1].set_title(f"Bruitée multiplicatif (SNR={snr_multiplicative:.2f})")
    ax[1].axis('off')

    # Image débruitée (Médian)
    ax[2].imshow(denoised_median_multiplicative, cmap='gray')
    ax[2].set_title(f"Débruitée Médian (SNR={snr_denoised_median_multiplicative:.2f})")
    ax[2].axis('off')

    # Image débruitée (Convolution)
    ax[3].imshow(denoised_convolution_multiplicative, cmap='gray')
    ax[3].set_title(f"Débruitée Convolution (SNR={snr_denoised_convolution_multiplicative:.2f})")
    ax[3].axis('off')

    # Afficher les résultats
    plt.tight_layout()
    plt.show()


   # Tester différents niveaux de bruit gaussien
    variances = [0.01, 0.05, 0.1]
    results = []
    for var in variances:
        noisy_image = add_noise_salt_pepper(image_reference, amount=var)
        snr_noisy = calculate_snr(image_reference, noisy_image)

        # Débruiter avec le filtre médian
        denoised_median = denoise_image_median(noisy_image, filter_size=3)
        snr_denoised_median = calculate_snr(image_reference, denoised_median)
        snr_gain_median = calculate_snr_gain(snr_noisy, snr_denoised_median)

        # Débruiter avec la convolution
        kernel = np.ones((5, 5)) / 25
        denoised_convolution = convolve_image(noisy_image, kernel)
        snr_denoised_convolution = calculate_snr(image_reference, denoised_convolution)
        snr_gain_convolution = calculate_snr_gain(snr_noisy, snr_denoised_convolution)

        # Ajouter les résultats pour chaque méthode
        results.append({
            "Variance": var,
            "SNR bruité (S&P)": snr_noisy,
            "SNR débruité (Médian)": snr_denoised_median,
            "Gain SNR (Médian)": snr_gain_median,
            "SNR débruité (Convolution)": snr_denoised_convolution,
            "Gain SNR (Convolution)": snr_gain_convolution,
        })

    # Créer un DataFrame pour exploiter les données
    df_results = pd.DataFrame(results)

    # Tracer les résultats
    plt.figure(figsize=(10, 6))

    # SNR après débruitage pour chaque méthode
    plt.plot(df_results["Variance"], df_results["SNR bruité (S&P)"], marker='o', label="SNR bruité (S&P)", color='blue')
    plt.plot(df_results["Variance"], df_results["SNR débruité (Médian)"], marker='o', label="SNR débruité (Médian)", color='green')
    plt.plot(df_results["Variance"], df_results["SNR débruité (Convolution)"], marker='o', label="SNR débruité (Convolution)", color='orange')

    # Configurer le graphique
    plt.title("SNR avant et après débruitage pour différents niveaux de bruit", fontsize=14)
    plt.xlabel("Variance (Niveau de bruit)", fontsize=12)
    plt.ylabel("SNR", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Afficher le graphique
    plt.show()

    # Tester différents niveaux de bruit gaussien
    variances = [0.01, 0.05, 0.1]
    results = []
    for var in variances:
        # Ajouter du bruit gaussien
        noisy_image = add_noise_additive(image_reference, var=var)
        snr_noisy = calculate_snr(image_reference, noisy_image)

        # Débruiter avec convolution (noyau 3x3)
        kernel_3x3 = np.ones((3, 3)) / 9
        denoised_3x3 = convolve_image(noisy_image, kernel_3x3)
        snr_denoised_3x3 = calculate_snr(image_reference, denoised_3x3)

        # Débruiter avec convolution (noyau 5x5)
        kernel_5x5 = np.ones((5, 5)) / 25
        denoised_5x5 = convolve_image(noisy_image, kernel_5x5)
        snr_denoised_5x5 = calculate_snr(image_reference, denoised_5x5)

        # Ajouter les résultats
        results.append({
            "Variance": var,
            "SNR bruité (Gaussien)": snr_noisy,
            "SNR débruité (Convolution 3x3)": snr_denoised_3x3,
            "SNR débruité (Convolution 5x5)": snr_denoised_5x5,
        })

    # Créer un DataFrame pour exploiter les données
    df_results = pd.DataFrame(results)

    # Tracer les résultats
    plt.figure(figsize=(10, 6))

    # SNR pour chaque méthode
    plt.plot(df_results["Variance"], df_results["SNR bruité (Gaussien)"], marker='o', label="SNR bruité (Gaussien)", color='blue')
    plt.plot(df_results["Variance"], df_results["SNR débruité (Convolution 3x3)"], marker='o', label="SNR débruité (Convolution 3x3)", color='green')
    plt.plot(df_results["Variance"], df_results["SNR débruité (Convolution 5x5)"], marker='o', label="SNR débruité (Convolution 5x5)", color='orange')

    # Configurer le graphique
    plt.title("SNR avant et après débruitage (Bruit Gaussien)", fontsize=14)
    plt.xlabel("Variance (Niveau de bruit)", fontsize=12)
    plt.ylabel("SNR", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Afficher le graphique
    plt.show()


if __name__ == "__main__":
    main()