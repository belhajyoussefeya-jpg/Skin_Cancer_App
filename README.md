# SKIN.AI — Détection du Cancer de la Peau par IA

Application web Flask permettant de classifier des lésions cutanées comme **Bénignes** ou **Malignes** à partir d'images, en utilisant un modèle de Deep Learning basé sur **VGG16** (transfer learning).

> **TD 8 — Introduction à l'IA — ENSTAB 2025/2026**

---

## 🎯 Fonctionnalités

- 🔐 **Authentification** par nom d'utilisateur / mot de passe avec gestion de session
- 📊 **Tableau de bord** : statistiques globales (total patients, taux de malignité, moyenne d'âge) et série temporelle sur 7 jours
- 🖼️ **Analyse d'image** : upload d'une photo de lésion → prédiction du modèle VGG16 → résultat + probabilité
- 👥 **Gestion des patients** : liste, recherche, filtre (bénin / malin), édition, suppression
- 📄 **Génération de rapports PDF** avec image, diagnostic, confiance et recommandations médicales
- 🔌 **API REST JSON** (`/api/predict`) sécurisée par clé API pour intégration externe
- 🎨 **Interface Dark Neon AI** responsive

---

## 🧰 Technologies utilisées

### Backend & langage
| Technologie | Version | Rôle |
|---|---|---|
| **Python** | 3.10 / 3.11 | Langage principal |
| **Flask** | ≥ 2.3 | Micro-framework web (routage, templates, sessions) |
| **Werkzeug** | ≥ 2.3 | Utilitaires WSGI, `secure_filename` pour les uploads |
| **Jinja2** | (inclus avec Flask) | Moteur de templates HTML |

### Intelligence Artificielle / Deep Learning
| Technologie | Rôle |
|---|---|
| **TensorFlow / Keras** (≥ 2.12) | Framework de Deep Learning pour charger et exécuter le modèle |
| **VGG16** | Architecture CNN pré-entraînée (ImageNet) utilisée en transfer learning |
| **NumPy** (≥ 1.24) | Calcul numérique, manipulation des tenseurs d'images |
| **Pillow (PIL)** (≥ 10.0) | Lecture, redimensionnement et conversion RGB des images |

### Base de données
| Technologie | Rôle |
|---|---|
| **MySQL** (via XAMPP) | SGBD relationnel — stocke utilisateurs et patients |
| **mysql-connector-python** (≥ 8.0.33) | Driver Python officiel pour MySQL |
| **phpMyAdmin** | Interface web d'administration de la base (fournie par XAMPP) |

### Génération de documents
| Technologie | Rôle |
|---|---|
| **ReportLab** (≥ 4.0) | Génération dynamique des rapports PDF (Platypus, Tables, Styles) |

### Frontend
| Technologie | Rôle |
|---|---|
| **HTML5 / CSS3** | Structure et style des pages |
| **CSS custom (Dark Neon)** | Thème graphique défini dans `static/style.css` |
| **Jinja2** | Templates dynamiques héritant de `base.html` |

### Infrastructure / Outils
| Technologie | Rôle |
|---|---|
| **XAMPP** | Pile Apache + MySQL + PHP pour le développement local |
| **Git / GitHub** | Versioning et hébergement du code source |
| **venv** | Isolation de l'environnement Python |

---

## 🏗️ Architecture

```
┌────────────────────┐         ┌────────────────────┐
│   Navigateur Web   │ ──HTTP──│   Flask (app.py)   │
│  (HTML / CSS / JS) │         │   Port 5000        │
└────────────────────┘         └──────────┬─────────┘
                                          │
                          ┌───────────────┼───────────────┐
                          │               │               │
                          ▼               ▼               ▼
                  ┌─────────────┐  ┌────────────┐  ┌────────────┐
                  │   VGG16     │  │   MySQL    │  │  ReportLab │
                  │  (.h5 file) │  │ (XAMPP)    │  │   (PDF)    │
                  └─────────────┘  └────────────┘  └────────────┘
```

**Flux d'une prédiction :**
1. L'utilisateur upload une image via `/predict`
2. Werkzeug sécurise le nom de fichier, Pillow ouvre et redimensionne l'image en **224×224 RGB**
3. NumPy normalise les pixels (`/ 255.0`) et ajoute une dimension batch
4. TensorFlow exécute `model.predict()` → probabilité de malignité
5. Résultat (Bénin / Malin + confiance) stocké dans MySQL et affiché à l'utilisateur
6. PDF généré à la demande via ReportLab

---

## 📁 Structure du projet

```
SKIN_CANCER_APP/
├── app.py                       # Application Flask (routes, modèle, BDD, API, PDF)
├── debug_model.py               # Script de diagnostic du modèle
├── database.sql                 # Schéma MySQL + données initiales
├── requirements.txt             # Dépendances Python
├── README.md                    # Ce fichier
├── .gitignore                   # Fichiers exclus du dépôt
│
├── model/
│   ├── PLACE_MODEL_HERE.txt     # Placeholder
│   └── vgg16_skin_cancer.h5     # ⚠️ Modèle (non versionné, à obtenir séparément)
│
├── static/
│   ├── style.css                # Thème Dark Neon
│   └── uploads/                 # Images uploadées (créées dynamiquement)
│
└── templates/
    ├── base.html                # Template parent (navbar, layout)
    ├── login.html               # Page de connexion
    ├── dashboard.html           # Tableau de bord + graphiques
    ├── predict.html             # Formulaire d'analyse
    ├── result.html              # Résultat de la prédiction
    ├── patients.html            # Liste des patients
    ├── edit_patient.html        # Édition d'un patient
    ├── 404.html                 # Erreur 404
    └── 500.html                 # Erreur 500
```

---

## 🛠️ Installation

### 1. Prérequis
- **Python 3.10 ou 3.11** (TensorFlow ne supporte pas encore Python 3.13/3.14)
- **XAMPP** (ou tout serveur MySQL local) — [téléchargement](https://www.apachefriends.org/)
- Le fichier modèle **`vgg16_skin_cancer.h5`** (~130 Mo, fourni par votre enseignante)

### 2. Cloner le projet

```bash
git clone https://github.com/belhajyoussefeya-jpg/Skin_Cancer_App.git
cd Skin_Cancer_App
```

### 3. Placer le modèle
Le fichier `vgg16_skin_cancer.h5` **n'est pas inclus dans le dépôt** (limite de 100 Mo par fichier sur GitHub). Demandez-le à votre enseignante et copiez-le dans le dossier `model/`.

### 4. Démarrer MySQL via XAMPP
- Lancez **XAMPP Control Panel**
- Démarrez les modules **Apache** et **MySQL**
- Ouvrez **phpMyAdmin** : `http://localhost/phpmyadmin`

### 5. Créer la base de données
Dans phpMyAdmin → onglet **SQL** → collez le contenu de `database.sql` → **Exécuter**.

Cela crée :
- la base `skin_cancer_db`
- les tables `users` et `patients`
- l'utilisateur par défaut : **admin / 1234**

### 6. Installer les dépendances Python

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer le venv (Windows)
venv\Scripts\activate
# (macOS/Linux) source venv/bin/activate

# Installer les paquets
pip install -r requirements.txt
```

### 7. Lancer l'application

```bash
python app.py
```

Puis ouvrez **http://127.0.0.1:5000** dans votre navigateur.

---

## 🔑 Utilisation

1. **Connexion** : identifiants par défaut `admin` / `1234`
2. **Tableau de bord** : statistiques + accès rapide aux actions
3. **Nouvelle analyse** : saisir nom + âge du patient + uploader une image de lésion (PNG, JPG, JPEG, BMP, GIF — max 16 Mo)
4. **Résultat** : diagnostic **Bénin** ou **Malin** avec pourcentage de confiance
5. **Patients** : historique complet, recherche, filtres, édition, suppression, export PDF

---

## 🔌 API REST

L'application expose une API JSON pour intégration externe.

### `POST /api/predict`

**Headers :**
```
X-API-Key: demo-key-change-me
```

**Form data :**
| Champ | Type | Requis | Description |
|---|---|---|---|
| `image` | file | ✅ | Image de la lésion |
| `name` | string | ❌ | Nom du patient (pour enregistrement) |
| `age` | int | ❌ | Âge du patient |
| `save` | string | ❌ | `1` pour enregistrer en BDD (défaut), `0` sinon |

**Réponse JSON :**
```json
{
  "result": "Malin",
  "probability": 87.42,
  "prob_malignant_raw": 0.8742,
  "image_path": "uploads/abc123_lesion.jpg",
  "saved": true,
  "patient_id": 42
}
```

> 🔐 Changez la clé API en production via la variable d'environnement `SKIN_API_KEY`.

---

## 🐞 Dépannage

### `ModuleNotFoundError: No module named 'tensorflow'`
→ Vérifiez votre version de Python (3.10 ou 3.11). Réinstallez : `pip install -r requirements.txt`.

### `mysql.connector.errors.ProgrammingError: 1045 Access denied`
→ Vérifiez les identifiants MySQL dans `app.py` (`DB_CONFIG`). Par défaut sur XAMPP : `root` sans mot de passe.

### `Can't connect to MySQL server`
→ Démarrez le module **MySQL** dans XAMPP Control Panel.

### `OSError: Unable to open file vgg16_skin_cancer.h5`
→ Le modèle doit se trouver dans le dossier `model/`.

### Le modèle prédit toujours la même classe
→ La normalisation ne correspond pas à celle de l'entraînement. Dans `app.py`, modifiez la constante en haut du fichier :
```python
PREPROCESSING_MODE = 'rescale'   # ou 'vgg16' ou 'raw'
MALIGNANT_INDEX = 1              # ou 0 si les classes sont inversées
```

---

## ⚠️ Note pédagogique

Ce projet stocke les mots de passe **en clair** dans la base de données (conformément au sujet du TD à des fins pédagogiques).
**Ne jamais faire ça en production** — utilisez `werkzeug.security.generate_password_hash` et `check_password_hash` pour hacher les mots de passe avec un sel cryptographique.

De même, la `secret_key` Flask, la clé API et les identifiants MySQL doivent être chargés depuis des **variables d'environnement** (`.env` + `python-dotenv`) et **jamais commit** dans le dépôt.

---

## 📜 Licence

Projet académique réalisé dans le cadre du module **Introduction à l'IA** à l'**ENSTAB** (2025/2026).
Usage pédagogique uniquement — ne remplace pas un diagnostic médical.
