# SKIN_CANCER_APP

Application web Flask pour la détection du cancer de la peau (Bénin / Malin) à partir d'images de lésions cutanées, en utilisant un modèle Deep Learning VGG16 pré-entraîné.

**TD 8 - Introduction à l'IA - ENSTAB 2025/2026**

---

## 📁 Structure du projet

```
SKIN_CANCER_APP/
├── app.py                 # Application Flask principale
├── database.sql           # Script de création de la base MySQL
├── requirements.txt       # Dépendances Python
├── README.md              # Ce fichier
│
├── model/
│   └── vgg16_skin_cancer.h5    # ⚠️ À placer ici (modèle fourni)
│
├── static/
│   ├── style.css
│   └── uploads/                # Images uploadées (créé automatiquement)
│
└── templates/
    ├── base.html
    ├── login.html
    ├── dashboard.html
    ├── predict.html
    ├── result.html
    ├── patients.html
    ├── 404.html
    └── 500.html
```

---

## 🛠️ Installation

### 1. Prérequis
- **Python 3.10 ou 3.11** (TensorFlow ne supporte pas encore Python 3.14)
- **XAMPP** (ou tout serveur MySQL local)
- Le fichier modèle **`vgg16_skin_cancer.h5`** fourni par votre enseignante

### 2. Cloner le projet

```bash
git clone https://github.com/belhajyoussefeya-jpg/Skin_Cancer_App.git
cd Skin_Cancer_App
```

### 3. Placer le modèle
Le fichier de poids `vgg16_skin_cancer.h5` (~130 Mo) **n'est pas inclus dans le dépôt** car il dépasse la limite de taille de fichier de GitHub. Demandez-le à votre enseignante et copiez-le dans le dossier `model/`.

### 4. Démarrer MySQL via XAMPP
- Lancez XAMPP Control Panel
- Démarrez les modules **Apache** et **MySQL**
- Ouvrez **phpMyAdmin** (`http://localhost/phpmyadmin`)

### 5. Créer la base de données
Dans phpMyAdmin → onglet **SQL** → collez le contenu de `database.sql` → **Exécuter**.

Cela crée :
- la base `skin_cancer_db`
- les tables `users` et `patients`
- l'utilisateur par défaut : **admin / 1234**

### 6. Installer les dépendances Python
Ouvrez un terminal dans le dossier `SKIN_CANCER_APP/` :

```bash
# (Optionnel mais recommandé) créer un environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Installer les paquets
pip install -r requirements.txt
```

### 7. Lancer l'application

```bash
python app.py
```

Puis ouvrez votre navigateur sur **http://127.0.0.1:5000**

---

## 🔑 Utilisation

1. **Connexion** : `admin` / `1234`
2. **Tableau de bord** : trois actions disponibles
   - **Nouvelle analyse** : saisir nom + âge + uploader une image
   - **Patients** : voir l'historique de toutes les analyses
   - **Déconnexion**
3. Le résultat affiche **Bénin** ou **Malin** + probabilité, et stocke l'analyse en BDD.

---

## 🐞 Dépannage

### `ModuleNotFoundError: No module named 'tensorflow'`
→ Vérifiez que vous utilisez Python 3.10 ou 3.11. Réinstallez : `pip install -r requirements.txt`.

### `mysql.connector.errors.ProgrammingError: 1045 Access denied`
→ Vérifiez les identifiants MySQL dans `app.py` (par défaut : `root` sans mot de passe pour XAMPP).

### `Can't connect to MySQL server`
→ Démarrez le module **MySQL** dans XAMPP Control Panel.

### `OSError: Unable to open file vgg16_skin_cancer.h5`
→ Le modèle doit se trouver dans le dossier `model/`.

### Le modèle prédit toujours la même classe
→ Le modèle a peut-être été entraîné avec une normalisation différente. Dans `app.py`, fonction `preprocess_image`, remplacez :
```python
arr = preprocess_input(arr)
```
par :
```python
arr = arr / 255.0
```

---

## ⚠️ Note pédagogique

Ce projet stocke les mots de passe en **clair** (comme demandé dans le sujet du TD).
**Ne jamais faire ça en production** : utilisez `werkzeug.security.generate_password_hash`.
