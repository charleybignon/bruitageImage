import matplotlib.pyplot as plt
from snr import calculate_snr, calculate_snr_gain
from noise import add_noise_additive, add_noise_salt_pepper, add_noise_multiplicative
from denoising import denoise_image_median, convolve_image
from contours import detecter_contours_et_filtrer
from utils import load_image
import numpy as np
import pandas as pd

def menu_principal():
    while True:
        print("\n=== Menu Principal ===")
        print("1. Charger et comparer SNR des images bruitées")
        print("2. Ajouter du bruit à une image")
        print("3. Débruiter une image bruitée")
        print("4. Détecter les contours et filtrer l'image")
        print("5. Quitter")
        choix = input("Choisissez une option : ")
        
        if choix == "1":
            load_and_compare_snr()
        elif choix == "2":
            menu_ajouter_bruit()
        elif choix == "3":
            menu_debruiter_image()
        elif choix == "4":
            detecter_contours_et_filtrer()
        elif choix == "5":
            print("Fin du programme de traitement d'image !")
            break
        else:
            print("Choix invalide. Veuillez réessayer.")

def menu_ajouter_bruit():
    while True:
        print("\n=== Menu Ajouter du bruit ===")
        print("1 Choisir le type de bruit et la variance/niveau de bruit")
        print("2 Comparer différents bruitages avec un graphique")
        print("3 Retour au menu principal")
        choix = input("Choisissez une option : ")

        if choix == "1":
            choisir_bruit()
        elif choix == "2":
            compare_noises()
        elif choix == "3":
            break
        else:
            print("Choix invalide. Veuillez réessayer.")

def menu_debruiter_image():
    while True:
        print("\n=== Menu Débruiter une image ===")
        print("1 Visualiser le débruitage d'une image bruitée")
        print("2 Comparer les débruitages avec un graphique")
        print("3 Retour au menu principal")
        choix = input("Choisissez une option : ")

        if choix == "1":
            menu_visualiser_debruitage()
        elif choix == "2":
            menu_compare_denoises()
        elif choix == "3":
            break
        else:
            print("Choix invalide. Veuillez réessayer.")

def menu_visualiser_debruitage():
    while True:
        print("\n=== Visualiser le débruitage ===")
        print("1 d'une image bruitée par bruit additif")
        print("2 d'une image bruitée par bruit s&p")
        print("3 d'une image bruitée par bruit multiplicatif")
        print("4 Retour")
        choix = input("Choisissez une option : ")

        if choix == "1":
            denoise_image(type="additif")
        elif choix == "2":
            denoise_image(type="s&p")
        elif choix == "3":
            denoise_image(type="multiplicatif")
        elif choix == "4":
            break
        else:
            print("Choix invalide. Veuillez réessayer.")

def menu_compare_denoises():
    while True:
        print("\n=== Comparer les débruitages ===")
        print("1 d'une image bruitée par bruit s&p")
        print("2 par convolution 3x3 et 5x5")
        print("3 Retour")
        choix = input("Choisissez une option : ")

        if choix == "1":
            compare_denoise_sp()
        elif choix == "2":
            compare_denoise_gaussian()
        elif choix == "3":
            break
        else:
            print("Choix invalide. Veuillez réessayer.")

def choisir_bruit():
    print("\n=== Choisir le type de bruit ===")
    print("1. Bruit Gaussien (additif)")
    print("2. Bruit Sel et Poivre")
    print("3. Bruit Multiplicatif")
    print("4. Les 3 bruits")
    choix = input("Choisissez une option : ")

    if choix in ["1", "2", "3", "4"]:
        try:
            variance = float(input("Entrez la variance ou le niveau de bruit (ex: 0.1): "))
            if variance <= 0:
                raise ValueError("La variance doit être positive.")
        except ValueError as e:
            print(f"Entrée invalide : {e}")
            return
        image_reference_path = "data/image_reference1.png"
        image_reference = load_image(image_reference_path, as_gray=True)

        def afficher_comparaison(reference, bruitée, titre_bruit, snr):
            """Affiche l'image de référence à gauche et l'image bruitée à droite"""
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(reference, cmap='gray')
            axs[0].set_title("Image de Référence")
            axs[0].axis('off')
            
            axs[1].imshow(bruitée, cmap='gray')
            axs[1].set_title(f"{titre_bruit} (SNR: {snr:.2f})")
            axs[1].axis('off')
            
            plt.suptitle(f"Comparaison : Référence vs {titre_bruit}")
            plt.show()

        if choix == "1":
            noisy_image = add_noise_additive(image_reference, var=variance)
            bruit_type = "Bruit Gaussien"
            snr_noisy = calculate_snr(image_reference, noisy_image)
            afficher_comparaison(image_reference, noisy_image, bruit_type, snr_noisy)

        elif choix == "2":
            noisy_image = add_noise_salt_pepper(image_reference, amount=variance)
            bruit_type = "Bruit Sel et Poivre"
            snr_noisy = calculate_snr(image_reference, noisy_image)
            afficher_comparaison(image_reference, noisy_image, bruit_type, snr_noisy)

        elif choix == "3":
            noisy_image = add_noise_multiplicative(image_reference, var=variance)
            bruit_type = "Bruit Multiplicatif"
            snr_noisy = calculate_snr(image_reference, noisy_image)
            afficher_comparaison(image_reference, noisy_image, bruit_type, snr_noisy)

        elif choix == "4":
            # Bruit Gaussien
            noisy_gaussian = add_noise_additive(image_reference, var=variance)
            snr_gaussian = calculate_snr(image_reference, noisy_gaussian)
            
            # Bruit Sel et Poivre
            noisy_sp = add_noise_salt_pepper(image_reference, amount=variance)
            snr_sp = calculate_snr(image_reference, noisy_sp)
            
            # Bruit Multiplicatif
            noisy_multiplicative = add_noise_multiplicative(image_reference, var=variance)
            snr_multiplicative = calculate_snr(image_reference, noisy_multiplicative)
            
            # Afficher les résultats côte à côte
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            axs[0].imshow(image_reference, cmap='gray')
            axs[0].set_title("Référence")
            axs[0].axis('off')
            
            axs[1].imshow(noisy_gaussian, cmap='gray')
            axs[1].set_title(f"Bruit Gaussien\n(SNR: {snr_gaussian:.2f})")
            axs[1].axis('off')
            
            axs[2].imshow(noisy_sp, cmap='gray')
            axs[2].set_title(f"Sel et Poivre\n(SNR: {snr_sp:.2f})")
            axs[2].axis('off')
            
            axs[3].imshow(noisy_multiplicative, cmap='gray')
            axs[3].set_title(f"Multiplicatif\n(SNR: {snr_multiplicative:.2f})")
            axs[3].axis('off')
            
            plt.suptitle("Comparaison : Référence et Bruits")
            plt.show()

    else:
        print("Choix invalide. Retour au menu précédent.")

def compare_noises():
    # Charger l'image de référence
    image_reference_path = "data/image_reference1.png"
    image_reference = load_image(image_reference_path, as_gray=True)

    # Tester différents niveaux de bruit
    variances = [0.01, 0.02, 0.05, 0.08, 0.1]
    results_gaussian = []
    results_salt_pepper = []
    results_multiplicative = []

    # Itérer sur chaque variance
    for var in variances:
        # Bruit Gaussien
        noisy_gaussian = add_noise_additive(image_reference, var=var)
        snr_gaussian = calculate_snr(image_reference, noisy_gaussian)
        results_gaussian.append({"Variance": var, "SNR": snr_gaussian, "Type": "Gaussien"})

        # Bruit Sel et Poivre
        noisy_sp = add_noise_salt_pepper(image_reference, amount=var)
        snr_sp = calculate_snr(image_reference, noisy_sp)
        results_salt_pepper.append({"Variance": var, "SNR": snr_sp, "Type": "Sel et Poivre"})

        # Bruit Multiplicatif
        noisy_multiplicative = add_noise_multiplicative(image_reference, var=var)
        snr_multiplicative = calculate_snr(image_reference, noisy_multiplicative)
        results_multiplicative.append({"Variance": var, "SNR": snr_multiplicative, "Type": "Multiplicatif"})

    # Combiner les résultats
    all_results = results_gaussian + results_salt_pepper + results_multiplicative
    df_results = pd.DataFrame(all_results)

    # Tracer les résultats
    plt.figure(figsize=(10, 6))

    # Ajouter une courbe pour chaque type de bruit
    for bruit_type in df_results["Type"].unique():
        subset = df_results[df_results["Type"] == bruit_type]
        plt.plot(subset["Variance"], subset["SNR"], marker='o', label=f"SNR ({bruit_type})")

    # Configurer le graphique
    plt.title("SNR en fonction des Variances pour différents Bruitages", fontsize=14)
    plt.xlabel("Variance (Niveau de bruit)", fontsize=12)
    plt.ylabel("SNR", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Afficher le graphique
    plt.show()

def load_and_compare_snr():
    image_reference_path = "data/image_reference1.png"
    image_reference = load_image(image_reference_path, as_gray=True)

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

        message = "Calcul du SNR validé" if snr_difference < 0.001 else f"Différence de SNR: {snr_difference:.4f}"

        results.append({
            "Nom du fichier": image["filename"],
            "SNR attendu": image["expected_snr"],
            "SNR calculé": snr_calculated,
            "Différence": snr_difference,
            "Message": message,
        })

    df_comparison = pd.DataFrame(results)
    print(df_comparison)

def denoise_image(type):
    """
    Fonction pour ajouter un bruit spécifique à une image de référence, puis la débruiter 
    en utilisant deux méthodes (filtre médian et convolution).
    
    Args:
        type (str): Le type de bruit à ajouter ('additif' ou 's&p' ou 'multiplicatif').
    """
    # Charger l'image de référence
    image_reference_path = "data/lenaNB.tiff"
    image_reference = load_image(image_reference_path, as_gray=True)

    # Ajouter du bruit en fonction du type
    if type == "additif":
        noisy_image = add_noise_additive(image_reference, var=0.05)
        noise_title = "additif"
    elif type == "s&p":
        noisy_image = add_noise_salt_pepper(image_reference, amount=0.05)
        noise_title = "Sel & Poivre"
    elif type == "multiplicatif":
        noisy_image = add_noise_multiplicative(image_reference, var=0.05)
        noise_title = "multiplicatif"
    else:
        raise ValueError("Type de bruit non reconnu. Utilisez 'additif' ou 's&p' ou 'multiplicatif'.")

    # Calcul du SNR de l'image bruitée
    snr_noisy = calculate_snr(image_reference, noisy_image)

    # Débruitage avec un filtre médian
    denoised_median = denoise_image_median(noisy_image, filter_size=3)
    snr_denoised_median = calculate_snr(image_reference, denoised_median)

    # Débruitage avec convolution
    kernel = np.ones((5, 5)) / 25
    denoised_convolution = convolve_image(noisy_image, kernel)
    snr_denoised_convolution = calculate_snr(image_reference, denoised_convolution)

    # Affichage des résultats
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    # Image de référence
    ax[0].imshow(image_reference, cmap='gray')
    ax[0].set_title("Image de Référence")
    ax[0].axis('off')

    # Image bruitée
    ax[1].imshow(noisy_image, cmap='gray')
    ax[1].set_title(f"Bruitée ({noise_title})\nSNR = {snr_noisy:.2f}")
    ax[1].axis('off')

    # Image débruitée avec filtre médian
    ax[2].imshow(denoised_median, cmap='gray')
    ax[2].set_title(f"Débruitée (Filtre Médian)\nSNR = {snr_denoised_median:.2f}")
    ax[2].axis('off')

    # Image débruitée avec convolution
    ax[3].imshow(denoised_convolution, cmap='gray')
    ax[3].set_title(f"Débruitée (Convolution)\nSNR = {snr_denoised_convolution:.2f}")
    ax[3].axis('off')

    # Ajustement de la mise en page et affichage
    plt.tight_layout()
    plt.show()

def compare_denoise_sp():
    results = []

    # Charger l'image de référence
    image_reference_path = "data/lenaNB.tiff"
    image_reference = load_image(image_reference_path, as_gray=True)

    variances = [0.01, 0.05, 0.1]   
    for var in variances:
        # Ajouter du bruit sel et poivre
        noisy_image = add_noise_salt_pepper(image_reference, amount=var)
        snr_noisy = calculate_snr(image_reference, noisy_image)

        # Débruiter avec le filtre médian
        denoised_median = denoise_image_median(noisy_image, filter_size=3)
        snr_denoised_median = calculate_snr(image_reference, denoised_median)

        # Débruiter avec la convolution
        kernel = np.ones((5, 5)) / 25
        denoised_convolution = convolve_image(noisy_image, kernel)
        snr_denoised_convolution = calculate_snr(image_reference, denoised_convolution)

        results.append({
            "Variance": var,
            "SNR bruité (S&P)": snr_noisy,
            "SNR débruité (Médian)": snr_denoised_median,
            "SNR débruité (Convolution)": snr_denoised_convolution,
        })

    df_results = pd.DataFrame(results)

    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["Variance"], df_results["SNR bruité (S&P)"], marker='o', label="SNR bruité (S&P)", color='blue')
    plt.plot(df_results["Variance"], df_results["SNR débruité (Médian)"], marker='o', label="SNR débruité (Médian)", color='green')
    plt.plot(df_results["Variance"], df_results["SNR débruité (Convolution)"], marker='o', label="SNR débruité (Convolution)", color='orange')

    plt.title("SNR avant et après débruitage pour différents niveaux de bruit (S&P)", fontsize=14)
    plt.xlabel("Variance (Niveau de bruit)", fontsize=12)
    plt.ylabel("SNR", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_denoise_gaussian():
    results = []

    # Charger l'image de référence
    image_reference_path = "data/lenaNB.tiff"
    image_reference = load_image(image_reference_path, as_gray=True)

    variances = [0.01, 0.05, 0.1]
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

        results.append({
            "Variance": var,
            "SNR bruité (Gaussien)": snr_noisy,
            "SNR débruité (Convolution 3x3)": snr_denoised_3x3,
            "SNR débruité (Convolution 5x5)": snr_denoised_5x5,
        })

    df_results = pd.DataFrame(results)

    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["Variance"], df_results["SNR bruité (Gaussien)"], marker='o', label="SNR bruité (Gaussien)", color='blue')
    plt.plot(df_results["Variance"], df_results["SNR débruité (Convolution 3x3)"], marker='o', label="SNR débruité (Convolution 3x3)", color='green')
    plt.plot(df_results["Variance"], df_results["SNR débruité (Convolution 5x5)"], marker='o', label="SNR débruité (Convolution 5x5)", color='orange')

    plt.title("SNR avant et après débruitage (Bruit Gaussien)", fontsize=14)
    plt.xlabel("Variance (Niveau de bruit)", fontsize=12)
    plt.ylabel("SNR", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    menu_principal()
