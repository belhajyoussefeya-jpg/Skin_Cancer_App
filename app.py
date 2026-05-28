"""
SKIN_CANCER_APP - Application Web IA (version corrigée)
TD 8 - Introduction à l'IA - ENSTAB 2025/2026
Framework : Flask + VGG16 + MySQL
Theme    : Dark Neon AI

CORRECTIONS PRINCIPALES :
  - Prétraitement : passage de VGG16 preprocess_input à rescale 1/255
    (cause du bug "tout est malin")
  - Configuration centralisée du prétraitement et de l'index de classe
  - Log de la sortie brute du modèle dans la console pour debug
  - Si tu vois encore tout "malin", change PREPROCESSING_MODE ou MALIGNANT_INDEX
    en haut du fichier (voir commentaires).
"""

import os
import io
import uuid
from datetime import date, datetime, timedelta
from functools import wraps

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, session, jsonify, send_file, abort)
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Image as RLImage, Table, TableStyle)

import tensorflow as tf
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
except ImportError:
    from keras.models import load_model
    from keras.applications.vgg16 import preprocess_input as vgg16_preprocess

import mysql.connector


# ===========================================================================
# CONFIGURATION DU PRETRAITEMENT (TRES IMPORTANT)
# ===========================================================================
# Si le modele predit toujours la meme classe, change PREPROCESSING_MODE :
#   'rescale' = pixels / 255.0       (le plus courant, ImageDataGenerator)
#   'vgg16'   = preprocess_input VGG16 (BGR + soustraction moyenne ImageNet)
#   'raw'     = pixels bruts 0-255    (rare)
#
# Si les classes sont inversees (modele dit "Benin" pour des cas malins),
# change MALIGNANT_INDEX entre 0 et 1.
# ===========================================================================
PREPROCESSING_MODE = 'rescale'   # <-- CHANGE ICI SI BESOIN
MALIGNANT_INDEX = 1              # <-- CHANGE A 0 SI CLASSES INVERSEES
# ===========================================================================


# ---------------------------------------------------------------------------
# Configuration Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'change-me-in-production'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'skin_cancer_db'
}

MODEL_PATH = os.path.join('model', 'vgg16_skin_cancer.h5')
model = None

API_KEY = os.environ.get('SKIN_API_KEY', 'demo-key-change-me')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_skin_cancer_model():
    global model
    if model is None:
        try:
            model = load_model(MODEL_PATH)
            print(f"[OK] Modele charge depuis {MODEL_PATH}")
            print(f"[OK] Output shape : {model.output_shape}")
            print(f"[OK] Mode pretraitement : {PREPROCESSING_MODE}")
        except Exception as e:
            print(f"[ERREUR] Impossible de charger le modele : {e}")
            model = None
    return model


def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def preprocess_image(image_path):
    """Pretraitement configurable. Voir constantes en haut du fichier."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    if PREPROCESSING_MODE == 'rescale':
        arr = arr / 255.0
    elif PREPROCESSING_MODE == 'vgg16':
        arr = vgg16_preprocess(arr)
    elif PREPROCESSING_MODE == 'raw':
        pass  # pixels bruts 0-255
    else:
        raise ValueError(f"PREPROCESSING_MODE inconnu : {PREPROCESSING_MODE}")

    return arr


def run_prediction(image_path):
    """Run the model on an image and return (result, probability_pct, prob_malignant_raw)."""
    mdl = load_skin_cancer_model()
    if mdl is None:
        raise RuntimeError("Modele indisponible (model/vgg16_skin_cancer.h5)")

    arr = preprocess_image(image_path)
    prediction = mdl.predict(arr, verbose=0)

    # Log debug : tres utile pour comprendre ce qui sort du modele
    print(f"[Prediction] Image: {os.path.basename(image_path)}")
    print(f"[Prediction] Sortie brute: {prediction}")

    if prediction.shape[-1] == 1:
        # Sortie sigmoid : valeur unique entre 0 et 1
        prob_malignant = float(prediction[0][0])
    else:
        # Sortie softmax : on prend l'index configure
        prob_malignant = float(prediction[0][MALIGNANT_INDEX])

    print(f"[Prediction] prob_malignant = {prob_malignant:.4f}")

    if prob_malignant >= 0.5:
        result = 'Malin'
        probability = prob_malignant
    else:
        result = 'Benin'
        probability = 1.0 - prob_malignant

    # Pour rester compatible avec le reste du code (templates, BDD)
    # on garde l'orthographe avec accent dans la valeur stockee.
    if result == 'Benin':
        result = 'Bénin'

    return result, round(probability * 100, 2), prob_malignant


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash("Veuillez vous connecter d'abord.", 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def build_empty_timeseries():
    out = []
    for i in range(6, -1, -1):
        d = date.today() - timedelta(days=i)
        out.append({'day': d.strftime('%a %d/%m'),
                    'benign': 0, 'malignant': 0})
    return out


def get_dashboard_stats():
    """Compute all statistics for the dashboard."""
    empty = {
        'total': 0, 'benign': 0, 'malignant': 0,
        'today': 0, 'avg_age': 0,
        'malignant_rate': 0,
        'timeseries': build_empty_timeseries(),
        'recent': []
    }

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT COUNT(*) AS total FROM patients")
        total = cursor.fetchone()['total']

        cursor.execute("SELECT result, COUNT(*) AS c FROM patients GROUP BY result")
        rows = cursor.fetchall()
        benign = sum(r['c'] for r in rows if r['result'] == 'Bénin')
        malignant = sum(r['c'] for r in rows if r['result'] == 'Malin')

        cursor.execute(
            "SELECT COUNT(*) AS c FROM patients WHERE DATE(created_at) = CURDATE()"
        )
        today_count = cursor.fetchone()['c']

        cursor.execute("SELECT AVG(age) AS avg_age FROM patients")
        avg_age = cursor.fetchone()['avg_age'] or 0

        cursor.execute("""
            SELECT DATE(created_at) AS day,
                   SUM(CASE WHEN result='Bénin' THEN 1 ELSE 0 END) AS benign,
                   SUM(CASE WHEN result='Malin' THEN 1 ELSE 0 END) AS malignant
            FROM patients
            WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 6 DAY)
            GROUP BY DATE(created_at)
            ORDER BY day
        """)
        ts_raw = cursor.fetchall()

        cursor.execute(
            "SELECT * FROM patients ORDER BY created_at DESC LIMIT 5"
        )
        recent = cursor.fetchall()

        cursor.close()
        conn.close()

        ts_dict = {row['day']: row for row in ts_raw}
        timeseries = []
        for i in range(6, -1, -1):
            d = date.today() - timedelta(days=i)
            row = ts_dict.get(d)
            timeseries.append({
                'day': d.strftime('%a %d/%m'),
                'benign': int(row['benign']) if row else 0,
                'malignant': int(row['malignant']) if row else 0,
            })

        rate = round((malignant / total) * 100, 1) if total else 0

        return {
            'total': total,
            'benign': benign,
            'malignant': malignant,
            'today': today_count,
            'avg_age': round(float(avg_age), 1),
            'malignant_rate': rate,
            'timeseries': timeseries,
            'recent': recent,
        }

    except mysql.connector.Error as err:
        print(f"[Stats error] {err}")
        return empty


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash("Veuillez remplir tous les champs.", 'warning')
            return render_template('login.html')

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM users WHERE username=%s AND password=%s",
                (username, password)
            )
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user:
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash(f"Bienvenue, {user['username']} !", 'success')
                return redirect(url_for('dashboard'))
            else:
                flash("Identifiants incorrects.", 'danger')

        except mysql.connector.Error as err:
            flash(f"Erreur de base de donnees : {err}", 'danger')

    return render_template('login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    stats = get_dashboard_stats()
    return render_template(
        'dashboard.html',
        username=session.get('username'),
        stats=stats
    )


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        age_raw = request.form.get('age', '').strip()

        if not name or not age_raw:
            flash("Veuillez fournir le nom et l'age du patient.", 'warning')
            return redirect(url_for('predict'))

        try:
            age = int(age_raw)
            if age <= 0 or age > 130:
                raise ValueError
        except ValueError:
            flash("L'age doit etre un nombre entier valide.", 'warning')
            return redirect(url_for('predict'))

        if 'image' not in request.files:
            flash("Aucune image telechargee.", 'warning')
            return redirect(url_for('predict'))

        file = request.files['image']
        if file.filename == '':
            flash("Aucune image selectionnee.", 'warning')
            return redirect(url_for('predict'))

        if not allowed_file(file.filename):
            flash("Format non autorise (png, jpg, jpeg, bmp, gif).", 'warning')
            return redirect(url_for('predict'))

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)

        try:
            result, probability, _ = run_prediction(filepath)
        except RuntimeError as e:
            flash(str(e), 'danger')
            return redirect(url_for('predict'))
        except Exception as e:
            flash(f"Erreur lors de la prediction : {e}", 'danger')
            return redirect(url_for('predict'))

        image_path_relative = os.path.join('uploads', unique_name).replace('\\', '/')
        patient_id = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO patients
                   (name, age, result, probability, image_path)
                   VALUES (%s, %s, %s, %s, %s)""",
                (name, age, result, probability, image_path_relative)
            )
            patient_id = cursor.lastrowid
            conn.commit()
            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            flash(f"Erreur d'enregistrement : {err}", 'danger')

        return render_template(
            'result.html',
            patient_id=patient_id,
            name=name, age=age,
            result=result, probability=probability,
            image_path=image_path_relative,
        )

    return render_template('predict.html')


@app.route('/patients')
@login_required
def patients():
    q = request.args.get('q', '').strip()
    f = request.args.get('filter', 'all')

    sql = "SELECT * FROM patients WHERE 1=1"
    params = []
    if q:
        sql += " AND name LIKE %s"
        params.append(f"%{q}%")
    if f == 'benign':
        sql += " AND result = %s"
        params.append('Bénin')
    elif f == 'malignant':
        sql += " AND result = %s"
        params.append('Malin')
    sql += " ORDER BY created_at DESC"

    patients_list = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, tuple(params))
        patients_list = cursor.fetchall()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        flash(f"Erreur de base de donnees : {err}", 'danger')
    return render_template('patients.html', patients=patients_list, q=q, current_filter=f)


def _fetch_patient(patient_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM patients WHERE id=%s", (patient_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row


@app.route('/patients/<int:patient_id>/edit', methods=['GET', 'POST'])
@login_required
def patient_edit(patient_id):
    try:
        patient = _fetch_patient(patient_id)
    except mysql.connector.Error as err:
        flash(f"Erreur de base de donnees : {err}", 'danger')
        return redirect(url_for('patients'))
    if not patient:
        abort(404)

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        age_raw = request.form.get('age', '').strip()
        if not name or not age_raw:
            flash("Nom et age requis.", 'warning')
            return render_template('edit_patient.html', patient=patient)
        try:
            age = int(age_raw)
            if age <= 0 or age > 130:
                raise ValueError
        except ValueError:
            flash("L'age doit etre un entier valide.", 'warning')
            return render_template('edit_patient.html', patient=patient)

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE patients SET name=%s, age=%s WHERE id=%s",
                (name, age, patient_id)
            )
            conn.commit()
            cursor.close()
            conn.close()
            flash("Patient mis a jour.", 'success')
            return redirect(url_for('patients'))
        except mysql.connector.Error as err:
            flash(f"Erreur d'enregistrement : {err}", 'danger')

    return render_template('edit_patient.html', patient=patient)


@app.route('/patients/<int:patient_id>/delete', methods=['POST'])
@login_required
def patient_delete(patient_id):
    try:
        patient = _fetch_patient(patient_id)
        if not patient:
            abort(404)

        if patient.get('image_path'):
            img_full = os.path.join('static', patient['image_path'])
            if os.path.isfile(img_full):
                try:
                    os.remove(img_full)
                except OSError:
                    pass

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM patients WHERE id=%s", (patient_id,))
        conn.commit()
        cursor.close()
        conn.close()
        flash("Patient supprime.", 'info')
    except mysql.connector.Error as err:
        flash(f"Erreur de suppression : {err}", 'danger')
    return redirect(url_for('patients'))


def _build_pdf(patient):
    """Generate a PDF report for a patient analysis. Returns a BytesIO."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'title', parent=styles['Title'],
        textColor=colors.HexColor('#0a2540'), fontSize=20, spaceAfter=18)
    h_style = ParagraphStyle(
        'h', parent=styles['Heading2'],
        textColor=colors.HexColor('#0a2540'), fontSize=13, spaceAfter=8)
    body = styles['BodyText']

    story = []
    story.append(Paragraph("SKIN.AI &mdash; Rapport de diagnostic", title_style))
    story.append(Paragraph(
        f"Genere le {datetime.now().strftime('%d/%m/%Y a %H:%M')}", body))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Informations patient", h_style))
    info = [
        ['ID', f"#{patient['id']:03d}"],
        ['Nom', patient['name']],
        ['Age', f"{patient['age']} ans"],
        ['Date analyse', patient['created_at'].strftime('%d/%m/%Y %H:%M')
            if patient.get('created_at') else '-'],
    ]
    t = Table(info, colWidths=[4 * cm, 11 * cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eef3f8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0a2540')),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cdd7e0')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("Resultat IA (VGG16)", h_style))
    color = '#c0392b' if patient['result'] == 'Malin' else '#16a34a'
    story.append(Paragraph(
        f"<b>Diagnostic :</b> <font color='{color}'>{patient['result']}</font>", body))
    story.append(Paragraph(
        f"<b>Confiance :</b> {patient['probability']:.2f}%", body))
    story.append(Spacer(1, 0.4 * cm))

    img_full = os.path.join('static', patient['image_path']) if patient.get('image_path') else None
    if img_full and os.path.isfile(img_full):
        story.append(Paragraph("Image analysee", h_style))
        try:
            story.append(RLImage(img_full, width=8 * cm, height=8 * cm))
        except Exception:
            pass
    story.append(Spacer(1, 0.6 * cm))

    rec = ("Une consultation dermatologique specialisee est fortement recommandee "
           "pour confirmation et prise en charge."
           if patient['result'] == 'Malin'
           else "Lesion d'apparence benigne. Maintenir une surveillance dermatologique reguliere.")
    story.append(Paragraph("Recommandation", h_style))
    story.append(Paragraph(rec, body))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        "<i>Ce rapport est genere automatiquement a des fins educatives "
        "et ne remplace pas un avis medical.</i>", body))

    doc.build(story)
    buf.seek(0)
    return buf


@app.route('/patients/<int:patient_id>/pdf')
@login_required
def patient_pdf(patient_id):
    try:
        patient = _fetch_patient(patient_id)
    except mysql.connector.Error as err:
        flash(f"Erreur de base de donnees : {err}", 'danger')
        return redirect(url_for('patients'))
    if not patient:
        abort(404)

    pdf = _build_pdf(patient)
    safe_name = secure_filename(patient['name']) or 'patient'
    filename = f"rapport_{patient['id']:03d}_{safe_name}.pdf"
    return send_file(pdf, mimetype='application/pdf',
                     as_attachment=True, download_name=filename)


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------
def _check_api_key():
    key = request.headers.get('X-API-Key') or request.form.get('api_key')
    return key == API_KEY


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not _check_api_key():
        return jsonify({'error': 'invalid_api_key'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'missing_image'}), 400
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'invalid_file'}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    try:
        result, probability, raw = run_prediction(filepath)
    except RuntimeError as e:
        return jsonify({'error': 'model_unavailable', 'detail': str(e)}), 503
    except Exception as e:
        return jsonify({'error': 'prediction_failed', 'detail': str(e)}), 500

    image_path_relative = os.path.join('uploads', unique_name).replace('\\', '/')
    name = request.form.get('name', '').strip()
    age_raw = request.form.get('age', '').strip()
    save = request.form.get('save', '1') in ('1', 'true', 'yes')

    patient_id = None
    if save and name and age_raw:
        try:
            age = int(age_raw)
            if not (0 < age <= 130):
                raise ValueError
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO patients
                   (name, age, result, probability, image_path)
                   VALUES (%s, %s, %s, %s, %s)""",
                (name, age, result, probability, image_path_relative)
            )
            patient_id = cursor.lastrowid
            conn.commit()
            cursor.close()
            conn.close()
        except (ValueError, mysql.connector.Error) as err:
            return jsonify({
                'result': result,
                'probability': probability,
                'prob_malignant_raw': raw,
                'image_path': image_path_relative,
                'saved': False,
                'save_error': str(err),
            })

    return jsonify({
        'result': result,
        'probability': probability,
        'prob_malignant_raw': raw,
        'image_path': image_path_relative,
        'saved': patient_id is not None,
        'patient_id': patient_id,
    })


@app.route('/logout')
def logout():
    session.clear()
    flash("Vous etes deconnecte.", 'info')
    return redirect(url_for('login'))


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    load_skin_cancer_model()
    app.run(debug=True, host='127.0.0.1', port=5000)
