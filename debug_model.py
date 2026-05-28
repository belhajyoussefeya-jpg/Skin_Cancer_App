"""
debug_model.py - Diagnostic du modele VGG16

Lance ce script pour tester automatiquement quel pretraitement marche.
Le script :
  1. Charge le modele
  2. Cherche des images de test (dans static/uploads/, dataset/, data/, etc.)
  3. Teste les 3 modes de pretraitement
  4. Te dit lequel utiliser dans app.py

Utilisation : python debug_model.py
"""

import os
import glob
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

MODEL_PATH = os.path.join('model', 'vgg16_skin_cancer.h5')


def find_test_images():
    """Cherche automatiquement des images de test dans les dossiers usuels."""
    candidates = []

    # Dossiers ou chercher des images
    search_dirs = [
        'static/uploads',
        'dataset/benign', 'dataset/malignant',
        'dataset/train/benign', 'dataset/train/malignant',
        'dataset/test/benign', 'dataset/test/malignant',
        'data/benign', 'data/malignant',
        'test_images',
    ]

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')

    for d in search_dirs:
        if os.path.isdir(d):
            for ext in extensions:
                candidates.extend(glob.glob(os.path.join(d, ext)))

    return candidates[:6]  # max 6 images


def test_image(model, path):
    """Teste les 3 methodes de pretraitement sur une image."""
    img = Image.open(path).convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    p_vgg = model.predict(preprocess_input(arr.copy()), verbose=0)
    p_rescale = model.predict(arr.copy() / 255.0, verbose=0)
    p_raw = model.predict(arr.copy(), verbose=0)

    return p_vgg, p_rescale, p_raw


def format_pred(pred, malignant_index=1):
    """Formatte la sortie pour l'affichage."""
    if pred.shape[-1] == 1:
        prob_mal = float(pred[0][0])
    else:
        prob_mal = float(pred[0][malignant_index])
    label = 'Malin' if prob_mal >= 0.5 else 'Benin'
    return f"{pred.flatten()} -> {label} ({prob_mal*100:.1f}%)"


def main():
    print("=" * 70)
    print(" DIAGNOSTIC MODELE SKIN_CANCER_APP")
    print("=" * 70)

    if not os.path.isfile(MODEL_PATH):
        print(f"\n[ERREUR] Modele introuvable : {MODEL_PATH}")
        print("Place le fichier vgg16_skin_cancer.h5 dans le dossier model/")
        return

    print(f"\n[1/3] Chargement du modele...")
    model = load_model(MODEL_PATH)
    print(f"      Output shape : {model.output_shape}")
    last_layer = model.layers[-1]
    print(f"      Derniere couche : {last_layer.__class__.__name__} "
          f"(activation = {last_layer.get_config().get('activation', '?')})")

    if model.output_shape[-1] == 1:
        print(f"      -> Modele BINAIRE (sigmoid). Index malin non applicable.")
    else:
        print(f"      -> Modele a {model.output_shape[-1]} classes (softmax).")
        print(f"         Index 0 et 1 a tester (un est Benin, l'autre Malin).")

    print(f"\n[2/3] Recherche d'images de test...")
    images = find_test_images()
    if not images:
        print("      [SKIP] Aucune image trouvee.")
        print("      Place quelques images .jpg dans static/uploads/ et relance.")
        print("      Ideal : 1 cas benin connu + 1 cas malin connu.")
        return
    print(f"      {len(images)} image(s) trouvee(s) :")
    for p in images:
        print(f"        - {p}")

    print(f"\n[3/3] Test des 3 methodes de pretraitement")
    print("=" * 70)

    results = {'vgg16': [], 'rescale': [], 'raw': []}

    for path in images:
        print(f"\n  Image : {path}")
        try:
            p_vgg, p_rescale, p_raw = test_image(model, path)
            print(f"    A) preprocess_input VGG16  : {format_pred(p_vgg)}")
            print(f"    B) rescale 1/255           : {format_pred(p_rescale)}")
            print(f"    C) pixels bruts 0-255      : {format_pred(p_raw)}")

            # Stocke pour analyser la diversite des predictions
            if p_vgg.shape[-1] == 1:
                results['vgg16'].append(float(p_vgg[0][0]))
                results['rescale'].append(float(p_rescale[0][0]))
                results['raw'].append(float(p_raw[0][0]))
            else:
                results['vgg16'].append(float(p_vgg[0][1]))
                results['rescale'].append(float(p_rescale[0][1]))
                results['raw'].append(float(p_raw[0][1]))
        except Exception as e:
            print(f"    [ERREUR] {e}")

    # Analyse : la BONNE methode est celle qui donne des predictions VARIEES
    # (la mauvaise methode donne toujours la meme valeur, ex: tout malin)
    print("\n" + "=" * 70)
    print(" ANALYSE")
    print("=" * 70)

    for method, vals in results.items():
        if not vals:
            continue
        spread = max(vals) - min(vals)
        avg = sum(vals) / len(vals)
        all_same_class = all(v >= 0.5 for v in vals) or all(v < 0.5 for v in vals)
        print(f"\n  Methode '{method}':")
        print(f"    Probabilites malin : {[f'{v:.3f}' for v in vals]}")
        print(f"    Ecart max-min      : {spread:.3f}")
        print(f"    Moyenne            : {avg:.3f}")
        if all_same_class and len(vals) > 1:
            print(f"    [!] Toutes les images sont classees pareil -> mauvais pretraitement")
        elif spread > 0.2:
            print(f"    [OK] Bonne diversite des predictions -> probablement le bon pretraitement")

    print("\n" + "=" * 70)
    print(" RECOMMANDATION")
    print("=" * 70)
    print("""
  Dans app.py, en haut du fichier, mets :

    PREPROCESSING_MODE = 'XXX'

  ou XXX est la methode qui donne le plus de DIVERSITE dans les predictions
  (la plus grand 'Ecart max-min' ci-dessus, en supposant que tu as melange
  des images benignes et malignes dans tes tests).

  Si toutes les methodes donnent toujours la meme classe :
    -> Tes images de test sont peut-etre toutes du meme type
    -> Ou le modele est mal entraine (a re-entrainer)
""")


if __name__ == '__main__':
    main()
