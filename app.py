import os
import time
import joblib
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, abort
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg"}
DEBUG_MODE = True

# local background image path (from conversation assets)
BG_IMAGE_LOCAL_PATH = "/mnt/data/cebbc929-4526-45b6-82f8-f75526fa3b3e.png"

app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = "replace-with-a-secure-random-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ---------------- In-memory users (demo) ----------------
USERS = {}
class User(UserMixin):
    def __init__(self, id, username, password_hash, email=None):
        self.id = str(id)
        self.username = username
        self.password_hash = password_hash
        self.email = email
    def verify_password(self, raw):
        return check_password_hash(self.password_hash, raw)

@login_manager.user_loader
def load_user(user_id):
    return USERS.get(str(user_id))

# ---------------- Demo model (keeps your previous logic) ----------------
MODEL_FILE = "models.pkl"
SCALER_FILE = "scaler.pkl"

sklearn_clf = None
scaler = None

def create_demo_model(seed: int = 42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    np.random.seed(seed)

    D = 224 * 224 * 3
    n_per_class = 30
    bright = np.random.normal(loc=0.85, scale=0.03, size=(n_per_class, D))
    dark   = np.random.normal(loc=0.15, scale=0.03, size=(n_per_class, D))
    X = np.vstack([bright, dark]).astype(np.float32)
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    scaler_local = StandardScaler()
    Xs = scaler_local.fit_transform(X)
    clf_local = LogisticRegression(max_iter=200)
    clf_local.fit(Xs, y)

    joblib.dump(clf_local, MODEL_FILE)
    joblib.dump(scaler_local, SCALER_FILE)

# load or create demo model
if os.path.exists(MODEL_FILE):
    try:
        sklearn_clf = joblib.load(MODEL_FILE)
    except Exception:
        sklearn_clf = None

if os.path.exists(SCALER_FILE):
    try:
        scaler = joblib.load(SCALER_FILE)
    except Exception:
        scaler = None

if sklearn_clf is None or scaler is None:
    try:
        create_demo_model()
        sklearn_clf = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    except Exception:
        sklearn_clf = None
        scaler = None

# ---------------- Helpers ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def load_image_rgb(path, size=(224,224)):
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

def preprocess_for_sklearn(path):
    arr = load_image_rgb(path)
    return arr.flatten().reshape(1, -1)

def heuristic_predict_from_lesion(path, bright_threshold=0.55):
    img = Image.open(path).convert("L").resize((256,256))
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean_val = float(arr.mean())
    label = "Malignant" if mean_val < bright_threshold else "Benign"
    confidence = round(abs(mean_val - bright_threshold) * 100, 2)
    return label, confidence

def map_to_binary_label_from_raw(raw, prob=None, pred_index=None, classes=None):
    if isinstance(raw, (int, np.integer)):
        return "Malignant" if int(raw) == 1 else "Benign"
    if "malign" in str(raw).lower(): return "Malignant"
    if "benign" in str(raw).lower(): return "Benign"
    if pred_index is not None:
        return "Malignant" if pred_index == 1 else "Benign"
    return "Benign"

# ---------------- Context processor to inject bg image URL ----------------
@app.context_processor
def inject_globals():
    try:
        bg_url = url_for("bg_image")
    except Exception:
        bg_url = None
    return dict(bg_image_url=bg_url)

# route to serve the background image file via HTTP
@app.route("/bg-image")
def bg_image():
    if not os.path.exists(BG_IMAGE_LOCAL_PATH):
        abort(404)
    return send_file(BG_IMAGE_LOCAL_PATH, conditional=True)

# ---------------- Routes ----------------
@app.route("/")
def home():
    # render index.html if exists, else fallback to upload page
    if os.path.exists(os.path.join(app.template_folder, "index.html")):
        return render_template("index.html")
    return redirect(url_for("upload"))

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    username = (request.form.get("username") or "").strip()
    email = (request.form.get("email") or "").strip()
    password = (request.form.get("password") or "")
    password2 = (request.form.get("password2") or "")
    if not username or not email or not password:
        flash("Please enter username, email and password.", "error")
        return redirect(url_for("register"))
    if password != password2:
        flash("Passwords do not match.", "error")
        return redirect(url_for("register"))
    for u in USERS.values():
        if u.username == username:
            flash("Username already exists.", "error")
            return redirect(url_for("register"))
        if u.email and u.email.lower() == email.lower():
            flash("Email already registered.", "error")
            return redirect(url_for("register"))
    uid = str(len(USERS) + 1)
    USERS[uid] = User(uid, username, generate_password_hash(password), email=email)
    flash("Registered! Please login.", "success")
    return redirect(url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    loginid = (request.form.get("username") or request.form.get("loginid") or "").strip()
    password = (request.form.get("password") or "")
    if not loginid or not password:
        flash("Enter both username/email and password", "error")
        return redirect(url_for("login"))
    matched = None
    for u in USERS.values():
        if u.username == loginid or (u.email and u.email.lower() == loginid.lower()):
            matched = u
            break
    if matched and check_password_hash(matched.password_hash, password):
        login_user(matched)
        flash("Logged in successfully.", "success")
        return redirect(url_for("upload"))
    flash("Invalid credentials", "error")
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user)

# MAIN upload route: accepts file, predicts, renders result.html
@app.route("/upload", methods=["GET","POST"])
def upload():
    # If GET: show upload page
    if request.method == "GET":
        return render_template("upload.html")

    # POST: process uploaded file
    if "image" not in request.files:
        flash("No file selected", "error")
        return redirect(request.url)

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected", "error")
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash("Invalid file type", "error")
        return redirect(request.url)

    # Save file with unique name
    filename = secure_filename(file.filename)
    timestamp = int(time.time() * 1000)
    filename = f"{timestamp}_{filename}"
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(saved_path)

    # Predict
    label = "Unknown"
    confidence = 0.0
    try:
        if sklearn_clf is not None:
            features = preprocess_for_sklearn(saved_path)
            if scaler is not None:
                features = scaler.transform(features)

            classes = getattr(sklearn_clf, "classes_", None)
            if hasattr(sklearn_clf, "predict_proba"):
                prob = sklearn_clf.predict_proba(features)[0]
                pred_index = int(np.argmax(prob))
                raw = classes[pred_index] if classes is not None else pred_index
                label = map_to_binary_label_from_raw(raw, prob, pred_index, classes)
                confidence = round(float(prob[pred_index]) * 100, 2)
            else:
                raw = sklearn_clf.predict(features)[0]
                label = map_to_binary_label_from_raw(raw)
                confidence = 90.0
        else:
            label, confidence = heuristic_predict_from_lesion(saved_path)
    except Exception:
        label, confidence = heuristic_predict_from_lesion(saved_path)

    image_url = url_for("static", filename="uploads/" + filename)
    # Render result page (no redirect) — this returns result.html with prediction context
    return render_template("result.html", image_url=image_url, label=label, confidence=confidence)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out", "info")
    return render_template("logout.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET","POST"])
def contact():
    if request.method == "POST":
        flash("Message received.", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")

# convenience: allow direct URL /<page>.html to render templates from templates/ (whitelist)
ALLOWED_TEMPLATE_HTML = {
    "login.html", "register.html", "upload.html", "result.html", "index.html", "home.html", "about.html", "contact.html"
}
@app.route("/<path:page_name>.html")
def serve_template_html(page_name):
    fname = f"{page_name}.html"
    if fname not in ALLOWED_TEMPLATE_HTML:
        abort(404)
    template_path = os.path.join(app.template_folder or "templates", fname)
    if not os.path.exists(template_path):
        abort(404)
    return render_template(fname)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=DEBUG_MODE)
