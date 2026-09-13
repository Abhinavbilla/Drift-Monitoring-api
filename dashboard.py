import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import requests
from streamlit_google_auth import Authenticate
import os
from dotenv import load_dotenv
import time 
import urllib.parse
import io
import csv
import gzip
import zipfile
import importlib.util
import time
import base64
import jwt
DEFAULT_BASELINE_SAMPLE_SIZE = 50000
# Mirrors drift/embedding_detector.py's thresholds (kept as a local constant
# since the dashboard and backend are separately deployed services that only
# talk over HTTP, matching this file's existing pattern of not importing
# backend modules directly).
EMBEDDING_RECOMMENDED_MIN_SAMPLES = 40
EMBEDDING_HARD_MIN_SAMPLES = 4
DEFAULT_PRODUCTION_BATCH_SIZE = 25000
# ---------------------------------------------------------
# 1. PAGE SETUP & ADAPTIVE UI CSS (Must be first!)
# ---------------------------------------------------------
st.set_page_config(page_title="Drift Sentinel", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    .premium-card {
        background-color: var(--secondary-background-color);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid var(--faded-text-color);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        gap: 20px;
        transition: transform 0.2s ease;
    }
    .premium-card:hover { transform: translateY(-4px); }
    
    .mini-card {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid var(--faded-text-color);
        text-align: center;
    }
    
    .icon-box {
        width: 60px; height: 60px;
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
    }
    .blue-glow { background: rgba(59, 130, 246, 0.15); color: #3B82F6; }
    .red-glow { background: rgba(239, 68, 68, 0.15); color: #EF4444; }
    .purple-glow { background: rgba(168, 85, 247, 0.15); color: #A855F7; }
    
    .card-title { color: var(--text-color); opacity: 0.7; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .card-value { color: var(--text-color); font-size: 2rem; font-weight: 800; line-height: 1; }
    .mini-value { color: var(--text-color); font-size: 1.3rem; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# Load the .env file
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def mint_session_token(user_info: dict) -> str:
    """
    Mints a short-lived signed session token asserting the Google-verified
    email, using COOKIE_KEY — a secret already shared between this service
    and the backend (see docker-compose.yml). This replaces the old
    API-key /register round trip entirely: no network call is needed since
    Google login already verified the user within this same process, and
    minting is cheap enough to call fresh on every request rather than
    caching (a stale cached value could outlive the token's expiry).
    """
    payload = {
        "email": user_info.get("email", "").lower(),
        "name": user_info.get("name", "User"),
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    return jwt.encode(payload, os.getenv("COOKIE_KEY"), algorithm="HS256")


# ---------------------------------------------------------
# HELPER: Read either a .csv or .dat uploaded file
# ---------------------------------------------------------

def _is_module_available(module_name: str) -> bool:
    """
    Checks if a module can be imported without actually importing it.
    Used to give a friendly install message instead of a raw ImportError
    when optional dependencies (openpyxl, xlrd, pyarrow) are missing.
    """
    return importlib.util.find_spec(module_name) is not None

def _read_csv_with_encoding_fallback(file_or_bytes, **kwargs):
    """
    Shared helper: tries multiple encodings until one works.
    Accepts either a file-like object or raw bytes.
    """
    encodings = ['utf-8-sig', 'cp1252', 'latin1', 'iso-8859-1']
    last_error = None

    for enc in encodings:
        try:
            if isinstance(file_or_bytes, bytes):
                buf = io.BytesIO(file_or_bytes)
            else:
                file_or_bytes.seek(0)
                buf = file_or_bytes
            return pd.read_csv(buf, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"Could not decode file with any common encoding. Last error: {last_error}")


def read_uploaded_file(file):
    """
    Reads a Streamlit UploadedFile object.
    Supports: CSV, TSV, Excel (.xlsx, .xls), JSON, JSON Lines, Parquet,
    ARFF, libsvm .dat, and gzip/zip compressed CSVs.
    Auto-detects encoding for text files.
    """
    filename = file.name.lower()
    raw_bytes = file.getvalue()

    # --------------------------------------------------------
    # 1. ARFF (Weka) – handles % comments and @ATTRIBUTE headers
    # --------------------------------------------------------
    if filename.endswith(".arff"):
        encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
        content = None
        for enc in encodings:
            try:
                content = raw_bytes.decode(enc).splitlines()
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            raise ValueError("Could not decode .arff file with any common encoding.")

        column_names = []
        data_rows = []
        in_data = False

        for line in content:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.upper().startswith('@RELATION'):
                continue
            if line.upper().startswith('@ATTRIBUTE'):
                parts = line.split()
                if len(parts) >= 3:
                    attr = parts[1].strip("'\"")
                    column_names.append(attr)
                continue
            if line.upper().startswith('@DATA'):
                in_data = True
                continue
            if in_data and line:
                # FIX: Use csv.reader instead of naive line.split(',') so
                # quoted strings with embedded commas (e.g. "Smith, John")
                # are parsed correctly — common in real-world ARFF exports.
                reader = csv.reader([line], skipinitialspace=True)
                row = next(reader)
                data_rows.append([x.strip() for x in row])

        if not column_names:
            # Fallback – treat as normal CSV
            return _read_csv_with_encoding_fallback(file, sep=None, engine='python')

        df = pd.DataFrame(data_rows, columns=column_names)

        # FIX: pd.to_numeric(errors='ignore') was removed in pandas 3.0.
        # Convert column-by-column with explicit try/except instead.
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass  # leave as text — intended "ignore" behaviour
        return df

    # --------------------------------------------------------
    # 2. DAT (libsvm style, with proper fallback to generic text)
    # --------------------------------------------------------
    if filename.endswith(".dat"):
        encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
        content = None
        for enc in encodings:
            try:
                content = raw_bytes.decode(enc).splitlines()
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            raise ValueError("Could not decode .dat file with any common encoding.")

        parsed_rows = []
        for line in content:
            parts = line.strip().split()
            if not parts:
                continue
            row = {}
            is_libsvm = False
            for token in parts[1:]:
                if ':' in token:
                    is_libsvm = True
                    idx, val = token.split(':')
                    row[f"Sensor_{idx}"] = float(val)
            if is_libsvm:
                parsed_rows.append(row)

        if parsed_rows:
            return pd.DataFrame(parsed_rows)

        # FIX: Previously this just did file.seek(0) and returned None.
        # Now it actually falls through to whitespace/CSV parsing.
        try:
            # Most non-libsvm .dat files are whitespace-delimited
            return _read_csv_with_encoding_fallback(file, sep=r'\s+', engine='python')
        except Exception:
            # Final fallback: try comma-separated in case it's CSV-like
            return _read_csv_with_encoding_fallback(file, sep=None, engine='python')

    # --------------------------------------------------------
    # 3. Excel (.xlsx, .xls) – reads all sheets, concatenates them
    # --------------------------------------------------------
    if filename.endswith((".xlsx", ".xls")):
        engine = 'openpyxl' if filename.endswith('.xlsx') else 'xlrd'
        required_module = 'openpyxl' if filename.endswith('.xlsx') else 'xlrd'
 
        if not _is_module_available(required_module):
            st.error(
                f"Please install {required_module} for Excel support: "
                f"`pip install {required_module}`"
            )
            raise ImportError(f"Missing {required_module}")
 
        # Read ALL sheets and concatenate, not just the first one.
        # Production Excel exports often have data split across sheets.
        all_sheets = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=None, engine=engine)
        if len(all_sheets) == 1:
            return next(iter(all_sheets.values()))
        return pd.concat(all_sheets.values(), ignore_index=True)

    # --------------------------------------------------------
    # 4. JSON and JSON Lines (.json, .jsonl, .ndjson)
    # --------------------------------------------------------
    if filename.endswith((".json", ".jsonl", ".ndjson")):
        # FIX: Standard JSON and JSON Lines need different read modes.
        # Try standard JSON first, fall back to lines=True for JSONL exports
        # (common in production logging systems).
        try:
            return pd.read_json(io.BytesIO(raw_bytes))
        except ValueError:
            return pd.read_json(io.BytesIO(raw_bytes), lines=True)

    # --------------------------------------------------------
    # 5. Parquet
    # --------------------------------------------------------
    if filename.endswith(".parquet"):
        try:
            import pyarrow  # noqa: F401
            return pd.read_parquet(io.BytesIO(raw_bytes))
        except ImportError:
            st.error("Please install pyarrow for Parquet support: `pip install pyarrow`")
            raise

    # --------------------------------------------------------
    # 6. TSV (tab-separated)
    # --------------------------------------------------------
    if filename.endswith(".tsv"):
        return _read_csv_with_encoding_fallback(file, sep='\t', engine='python')

    # --------------------------------------------------------
    # 7. Compressed CSV (.csv.gz, .gz)
    # --------------------------------------------------------
    if filename.endswith(".gz"):
        decompressed = gzip.decompress(raw_bytes)
        return _read_csv_with_encoding_fallback(decompressed, sep=None, engine='python')

    # --------------------------------------------------------
    # 8. Zipped CSV (.zip containing a single CSV/TXT file)
    # --------------------------------------------------------
    if filename.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
            inner_names = [n for n in z.namelist() if not n.endswith('/')]
            if not inner_names:
                raise ValueError("Zip file contains no readable files.")
            with z.open(inner_names[0]) as f:
                inner_bytes = f.read()
            return _read_csv_with_encoding_fallback(inner_bytes, sep=None, engine='python')

    # --------------------------------------------------------
    # 9. CSV / TXT / other text – auto-detect delimiter and encoding
    # --------------------------------------------------------
    return _read_csv_with_encoding_fallback(file, sep=None, engine='python')

# ---------------------------------------------------------
# v2.0: TEXT / IMAGE (EMBEDDING-BASED) MONITORING HELPERS
# ---------------------------------------------------------

def render_embedding_fit_ui(modality: str, session_token: str, key_prefix: str):
    """
    Shared 'lock a baseline' flow for text/image projects (Domain Classifier
    Test). Mirrors the tabular flow's shape (model ID -> upload -> fit ->
    success) but with modality-appropriate ingestion. No schema-confirmation
    step, since there's no per-column classification for a single embedding
    batch — only one baseline embedding matrix per project.
    """
    new_model_id = st.text_input(
        f"New Model ID (e.g., {modality}_monitor_v1)", key=f"{key_prefix}_{modality}_id"
    )

    reference_texts, reference_images = None, None
    sample_count = 0

    if modality == "text":
        uploaded = st.file_uploader(
            "Upload Reference Text (.txt = one document per line, or .csv)",
            type=["txt", "csv"], key=f"{key_prefix}_{modality}_file"
        )
        st.caption(
            f"Recommended: {EMBEDDING_RECOMMENDED_MIN_SAMPLES}+ documents for reliable drift "
            "detection (small batches produce noisy results). For .txt, one document per line — "
            "blank lines are skipped."
        )
        if uploaded is not None:
            if uploaded.name.lower().endswith(".csv"):
                df = _read_csv_with_encoding_fallback(uploaded, sep=None, engine='python')
                text_col = st.selectbox("Which column holds the text?", df.columns, key=f"{key_prefix}_{modality}_col")
                reference_texts = df[text_col].dropna().astype(str).tolist()
            else:
                lines = uploaded.getvalue().decode("utf-8", errors="ignore").splitlines()
                reference_texts = [line for line in lines if line.strip()]
            sample_count = len(reference_texts)
    else:
        uploaded_images = st.file_uploader(
            "Upload Reference Images (JPEG/PNG, multiple files)",
            type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"{key_prefix}_{modality}_files"
        )
        st.caption(
            f"Recommended: {EMBEDDING_RECOMMENDED_MIN_SAMPLES}+ images (JPEG or PNG) for reliable "
            "drift detection. Mixed sizes are fine — each image is resized automatically."
        )
        if uploaded_images:
            reference_images = [base64.b64encode(f.getvalue()).decode("ascii") for f in uploaded_images]
            sample_count = len(reference_images)

    if sample_count > 0:
        if sample_count < EMBEDDING_HARD_MIN_SAMPLES:
            st.error(f"Only {sample_count} sample(s) loaded — at least {EMBEDDING_HARD_MIN_SAMPLES} are required.")
        elif sample_count < EMBEDDING_RECOMMENDED_MIN_SAMPLES:
            st.warning(
                f"{sample_count} samples loaded — below the recommended {EMBEDDING_RECOMMENDED_MIN_SAMPLES}. "
                "Drift verdicts on small batches can be noisy."
            )
        else:
            st.caption(f"✓ {sample_count} samples loaded.")

    ready = bool(new_model_id and sample_count >= EMBEDDING_HARD_MIN_SAMPLES)

    if st.button("Start Monitoring Model", type="primary", key=f"{key_prefix}_{modality}_btn"):
        if not ready:
            st.warning(f"Please provide a Model ID and upload at least {EMBEDDING_HARD_MIN_SAMPLES} reference samples.")
        else:
            with st.spinner(f"Embedding {modality} baseline and locking monitoring pipeline..."):
                try:
                    headers = {"Authorization": f"Bearer {session_token}"}
                    safe_model_id = new_model_id.strip().replace(" ", "%20")
                    payload = (
                        {"reference_texts": reference_texts} if modality == "text"
                        else {"reference_images": reference_images}
                    )
                    response = requests.post(
                        f"{BACKEND_URL}/fit/{safe_model_id}/{modality}",
                        json=payload, headers=headers
                    )
                    if response.status_code == 200:
                        st.success(f"Successfully started monitoring '{new_model_id}' ({modality})!")
                        st.balloons()
                        time.sleep(1.5)
                        st.session_state.selected_model = new_model_id
                        st.rerun()
                    else:
                        st.error(f"Backend Error: {response.text}")
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")


def render_embedding_analyze_ui(modality: str, project_id: str, session_token: str):
    """
    Minimal batch-analysis panel for text/image projects: upload a
    comparison batch, run the Domain Classifier Test, show the AUC +
    verdict inline. There's no live per-item log stream here since
    real-time/streaming embedding scoring is out of scope (batch-only,
    matching the existing tabular limitation).
    """
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        f"This is a **{modality}** monitoring project. Drift is evaluated batch-by-batch "
        "via the Domain Classifier Test (no live per-item telemetry for this modality)."
    )

    production_texts, production_images = None, None
    sample_count = 0

    if modality == "text":
        uploaded = st.file_uploader(
            "Upload a Production Text Batch (.txt or .csv) to Analyze",
            type=["txt", "csv"], key=f"analyze_{modality}_{project_id}"
        )
        st.caption(
            f"Recommended: {EMBEDDING_RECOMMENDED_MIN_SAMPLES}+ documents, same format as the "
            "baseline upload (.txt = one document per line, or .csv)."
        )
        if uploaded is not None:
            if uploaded.name.lower().endswith(".csv"):
                df = _read_csv_with_encoding_fallback(uploaded, sep=None, engine='python')
                text_col = st.selectbox("Which column holds the text?", df.columns, key=f"analyze_{modality}_col_{project_id}")
                production_texts = df[text_col].dropna().astype(str).tolist()
            else:
                lines = uploaded.getvalue().decode("utf-8", errors="ignore").splitlines()
                production_texts = [line for line in lines if line.strip()]
            sample_count = len(production_texts)
    else:
        uploaded_images = st.file_uploader(
            "Upload a Production Image Batch to Analyze",
            type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"analyze_{modality}_{project_id}"
        )
        st.caption(f"Recommended: {EMBEDDING_RECOMMENDED_MIN_SAMPLES}+ images (JPEG or PNG).")
        if uploaded_images:
            production_images = [base64.b64encode(f.getvalue()).decode("ascii") for f in uploaded_images]
            sample_count = len(production_images)

    if sample_count > 0:
        if sample_count < EMBEDDING_HARD_MIN_SAMPLES:
            st.error(f"Only {sample_count} sample(s) loaded — at least {EMBEDDING_HARD_MIN_SAMPLES} are required.")
        elif sample_count < EMBEDDING_RECOMMENDED_MIN_SAMPLES:
            st.warning(
                f"{sample_count} samples loaded — below the recommended {EMBEDDING_RECOMMENDED_MIN_SAMPLES}. "
                "The AUC verdict may be noisy at this batch size."
            )
        else:
            st.caption(f"✓ {sample_count} samples loaded.")

    ready = sample_count >= EMBEDDING_HARD_MIN_SAMPLES

    if st.button("Analyze Batch", type="primary", key=f"analyze_btn_{modality}_{project_id}"):
        if not ready:
            st.warning(f"Please upload at least {EMBEDDING_HARD_MIN_SAMPLES} production samples first.")
        else:
            with st.spinner("Running Domain Classifier Test..."):
                headers = {"Authorization": f"Bearer {session_token}"}
                payload = (
                    {"production_texts": production_texts} if modality == "text"
                    else {"production_images": production_images}
                )
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/analyze/{project_id}/{modality}",
                        json=payload, headers=headers, timeout=120
                    )
                    if resp.status_code == 200:
                        metric = resp.json()["feature_metrics"]["embedding_drift"]
                        auc = metric["statistic"]
                        drifted = metric["drift_detected"]

                        mc1, mc2 = st.columns(2)
                        mc1.markdown(f"<div class='mini-card'><div class='card-title'>Domain Classifier AUC</div><div class='mini-value'>{auc:.4f}</div></div>", unsafe_allow_html=True)
                        color = "#EF4444" if drifted else "#10B981"
                        verdict = "DRIFT" if drifted else "STABLE"
                        mc2.markdown(f"<div class='mini-card'><div class='card-title'>Verdict</div><div class='mini-value' style='color:{color};'>{verdict}</div></div>", unsafe_allow_html=True)

                        if drifted:
                            st.error(f"🚨 Drift detected — AUC {auc:.4f} means the classifier can reliably tell the production batch apart from the baseline.")
                        else:
                            st.success(f"✅ No significant drift — AUC {auc:.4f} is close to 0.5 (indistinguishable from baseline).")
                    else:
                        st.error(f"Backend Error: {resp.text}")
                except Exception as e:
                    st.error(f"Error analyzing batch: {str(e)}")


# ---------------------------------------------------------
# 2. AUTHENTICATION & GATEKEEPER
# ---------------------------------------------------------
_RENDER_CREDENTIALS_PATH = '/etc/secrets/google_credentials.json'
_LOCAL_CREDENTIALS_PATH = 'google_credentials.json'
credentials_path = (
    _RENDER_CREDENTIALS_PATH if os.path.exists(_RENDER_CREDENTIALS_PATH)
    else _LOCAL_CREDENTIALS_PATH
)

authenticator = Authenticate(
    secret_credentials_path=credentials_path,
    cookie_name='drift_cookie',
    cookie_key=os.getenv('COOKIE_KEY'),
    redirect_uri=os.getenv("REDIRECT_URI", "https://drift-monitoring-dashboard.onrender.com")
)
authenticator.check_authentification()
authenticator.login()


if not st.session_state.get('connected'):
    st.stop() # If not logged in, the script STOPS here.

# Grab the current user's email
user_info = st.session_state.get('user_info', {})
current_user_email = user_info.get('email', '').lower()

# 1. Mint a session token from the verified Google login
user_api_key = mint_session_token(user_info)

# 2. Ask the backend for the list of projects using the session token
allowed_projects = []
if user_api_key:
    headers = {"Authorization": f"Bearer {user_api_key}"}
    try:
        response = requests.get(f"{BACKEND_URL}/projects", headers=headers, timeout=10)
        if response.status_code == 200:
            allowed_projects = response.json().get("projects", [])
    except Exception:
        allowed_projects = []

# 3. Format the list for the sidebar
if not allowed_projects:
    allowed_projects = ["No Models Assigned"]
else:
    allowed_projects.append("➕ Add New Model")


# --- 3. RENDER YOUR SIDEBAR ---
st.sidebar.success(f"Logged in as: {user_info.get('name', 'User')}")

if st.sidebar.button("Logout"):
    authenticator.logout()

st.sidebar.markdown("<h2>Control Panel</h2>", unsafe_allow_html=True)

# 2. Set default if state is missing
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = allowed_projects[0]

# 3. Calculate index dynamically based on the CURRENT list
try:
    # This finds the position of the model name in our fresh project list
    default_index = allowed_projects.index(st.session_state.selected_model)
except ValueError:
    # If the model name isn't found (e.g., first run), default to 0
    default_index = 0

# 4. Render sidebar & Capture Selection
dynamic_key = f"model_select_{len(allowed_projects)}"

selected = st.sidebar.selectbox(
    "Active Model", 
    options=allowed_projects, 
    index=default_index,
    key="model_selector"         
)

# 5. Manual State Sync (Replaces the callback)
if selected != st.session_state.selected_model:
    st.session_state.selected_model = selected
    st.rerun()

active_project = st.session_state.selected_model
# ---------------------------------------------------------
# 4. THE VAULT DOOR (Only runs if they have a project)
# ---------------------------------------------------------
if active_project == "➕ Add New Model":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Start Monitoring a New Model")
    st.write("Upload your training datasets to initialize a new monitoring pipeline.")
    
    # Mint a session token from the verified Google login
    user_api_key = mint_session_token(user_info)

    modality_choice = st.radio(
        "Model Type", ["Tabular", "Text", "Image"], key="add_new_modality", horizontal=True
    )

    if modality_choice == "Text":
        render_embedding_fit_ui("text", user_api_key, key_prefix="add_new")
    elif modality_choice == "Image":
        render_embedding_fit_ui("image", user_api_key, key_prefix="add_new")
    else:
        # New unique keys so Streamlit doesn't throw DuplicateElement errors
        new_model_id = st.text_input("New Model ID (e.g., churn_predictor_v2)", key="add_new_proj_id")

        uploaded_files = st.file_uploader(
            "Upload Training Data (CSV / DAT) - Max 2GB total",
            type=["csv", "dat"],
            key="add_new_files",
            accept_multiple_files=True
        )

        st.warning(
            "⚠️ **Notice:** If your combined training files approach or exceed the max 2GB limit, "
            "please upload your primary dataset first, then upload subsequent files sequentially."
        )

        # ==========================================
        # NEW: SMART SCHEMA CONFIGURATION UI
        # ==========================================
        schema_mapping = {}  # ← Correct indentation (same level as new_model_id)

        if uploaded_files:
            st.markdown("<br> 🗂️ Step 2: Verify Data Schema", unsafe_allow_html=True)

            # 1. Load full files and take an unbiased random sample
            train_dfs = []
            for f in uploaded_files:
                train_dfs.append(read_uploaded_file(f))

            # Combine all uploaded files into one massive dataframe
            combined_df = pd.concat(train_dfs, ignore_index=True)

            preview_df = combined_df.sample(n=min(10000,len(combined_df)), random_state=42).reset_index(drop=True)

            # 2. Ask the backend for mathematical recommendations
            with st.spinner("Analyzing dataset signals..."):
                headers = {"Authorization": f"Bearer {user_api_key}"}
                raw_dict = preview_df.to_dict(orient="list")

                clean_dict = {
                    col: [None if pd.isna(v) else v for v in vals]
                    for col, vals in raw_dict.items()
                }

                payload = {"reference_data": clean_dict}

                try:
                    profile_response = requests.post(f"{BACKEND_URL}/profile", json=payload, headers=headers)
                    if profile_response.status_code == 200:
                        smart_profiles = {p["name"]: p for p in profile_response.json()}
                    else:
                        smart_profiles = {}
                        st.error(f"Profiler Error: {profile_response.text}")
                except requests.exceptions.ConnectionError:
                    smart_profiles = {}
                    st.warning("Could not reach profiler API. Using default schema.")

            st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)

            # 3. Build the Stateful UI using the AI's recommendations
            for col in preview_df.columns:
                col_intel = smart_profiles.get(col, {})
                monitor = col_intel.get("monitor", "Review")
                reason = col_intel.get("reason", "Could not classify — recommend manual review")

                if monitor is True:
                    suggested_idx = 0
                elif monitor == "Categorical":
                    suggested_idx = 1
                elif monitor is False:
                    suggested_idx = 2
                else:
                    best = col_intel.get("best_guess", "ignore")
                    if best == "continuous":
                        suggested_idx = 0
                    elif best == "categorical":
                        suggested_idx = 1
                    else:
                        suggested_idx = 2

                state_key = f"schema_{col}"

                if state_key not in st.session_state or st.session_state.get(f"last_rec_{col}") != suggested_idx:
                    st.session_state[state_key] = suggested_idx
                    st.session_state[f"last_rec_{col}"] = suggested_idx

                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"**{col}**")
                    st.caption(f"Reason: {reason}")
                with c2:
                    if monitor == "Review":
                        st.warning(f"⚠️ **{col}** needs review.")

                    role_options = [
                        "📊 Continuous Feature (Monitor for Drift)",
                        "🔠 Categorical Feature (Monitor for Drift)",
                        "📝 Ignore (Unique IDs / Free Text / Target)"
                    ]

                    current_index = st.session_state[state_key]
                    if isinstance(current_index, str):
                        current_index = role_options.index(current_index) if current_index in role_options else 0
                        st.session_state[state_key] = current_index

                    role = st.selectbox(
                        "Column Role",
                        options=role_options,
                        index=current_index,
                        key=state_key,
                        label_visibility="collapsed"
                    )
                    schema_mapping[col] = role

            st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)

        # ==========================================
        # STEP 3: START MONITORING WITH FILTERED SCHEMA
        # ==========================================
        if st.button("Start Monitoring Model", type="primary", key="btn_add_new_model"):
            if not new_model_id or not uploaded_files:
                st.warning("Please provide a Model ID and upload at least one file.")
            else:
                with st.spinner("Processing datasets and starting monitoring..."):
                    try:
                        # Split confirmed schema into continuous vs categorical
                        continuous_cols = [
                            col for col, role in schema_mapping.items()
                            if "Continuous" in role and col in combined_df.columns
                        ]
                        categorical_cols = [
                            col for col, role in schema_mapping.items()
                            if "Categorical" in role and col in combined_df.columns
                        ]

                        if not continuous_cols and not categorical_cols:
                            st.warning(
                                "No columns selected for monitoring. "
                                "Please mark at least one column as Continuous or Categorical."
                            )
                            st.stop()

                        # Prepare continuous data
                        continuous_df = combined_df[continuous_cols].select_dtypes(
                            include=['number']
                        ).dropna()

                        if len(continuous_df) > DEFAULT_BASELINE_SAMPLE_SIZE:
                            continuous_df = continuous_df.sample(n=DEFAULT_BASELINE_SAMPLE_SIZE, random_state=42)

                        # Prepare categorical data
                        categorical_df = combined_df[categorical_cols].dropna() \
                            if categorical_cols else pd.DataFrame()
                        if len(categorical_df) > DEFAULT_BASELINE_SAMPLE_SIZE:
                            categorical_df = categorical_df.sample(n=DEFAULT_BASELINE_SAMPLE_SIZE, random_state=42)

                        # NaN scrub both payloads before sending
                        def scrub(df):
                            raw = df.to_dict(orient="list")
                            return {
                                col: [None if pd.isna(v) else v for v in vals]
                                for col, vals in raw.items()
                            }

                        payload = {
                            "reference_data": scrub(continuous_df),
                            "categorical_data": scrub(categorical_df) if not categorical_df.empty else {}
                        }

                        safe_model_id = new_model_id.strip().replace(" ", "%20")
                        headers = {"Authorization": f"Bearer {user_api_key}"}

                        # ✅ FIXED: Uses BACKEND_URL instead of hardcoded http://api
                        response = requests.post(
                            f"{BACKEND_URL}/fit/{safe_model_id}",
                            json=payload,
                            headers=headers
                        )

                        if response.status_code == 200:
                            result = response.json()
                            st.success(
                                f"Successfully started monitoring '{new_model_id}'! "
                                f"Watching {len(continuous_cols)} continuous and "
                                f"{len(categorical_cols)} categorical features."
                            )
                            st.balloons()
                            time.sleep(1.5)
                            st.session_state.selected_model = new_model_id
                            st.rerun()
                        else:
                            st.error(f"Backend Error: {response.text}")

                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")

elif active_project != "No Models Assigned":
    
    col_head, col_btn = st.columns([6, 1])
    with col_head:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3B82F6" width="40" height="40"><path fill-rule="evenodd" d="M14.615 1.595a.75.75 0 0 1 .359.852L12.982 9.75h7.268a.75.75 0 0 1 .548 1.262l-10.5 11.25a.75.75 0 0 1-1.272-.71l1.992-7.302H3.75a.75.75 0 0 1-.548-1.262l10.5-11.25a.75.75 0 0 1 .913-.143Z" clip-rule="evenodd" /></svg>
                <h1 style='margin: 0; padding: 0;'>Drift Sentinel</h1>
            </div>
            <p style='font-size: 1.1rem; opacity: 0.8;'>Enterprise telemetry for: <b>{}</b></p>
        """.format(active_project), unsafe_allow_html=True)

    with col_btn:
        st.write("") 
        if st.button("Sync Telemetry", use_container_width=True):
            st.rerun()

    # --- FETCH DATA FOR THE ACTIVE PROJECT VIA API ---

    if 'user_api_key' not in locals() or not user_api_key:
        user_api_key = mint_session_token(user_info)

    fences = {}
    logs = []
    modality = "tabular"

    if user_api_key:
        headers = {"Authorization": f"Bearer {user_api_key}"}

        # 1. Fetch baseline (IQR fences + modality)
        try:
            resp = requests.get(f"{BACKEND_URL}/baseline/{active_project}", headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                modality = data.get("modality", "tabular")
                for f in data.get("fences", []):
                    if "q1" in f and "q3" in f:
                        fences[f["feature_name"]] = {"q1": f["q1"], "q3": f["q3"]}
            elif resp.status_code != 404:
                st.warning("Could not fetch baseline data. Status code: {}".format(resp.status_code))
        except Exception as e:
            st.warning("Error fetching baseline: {}".format(str(e)))

        # 2. Fetch logs (last 1000) — tabular-only, since text/image drift is batch-based only
        if modality == "tabular":
            try:
                resp = requests.get(f"{BACKEND_URL}/logs/{active_project}", headers=headers, timeout=10)
                if resp.status_code == 200:
                    logs = resp.json()
                elif resp.status_code != 404:
                    st.warning("Could not fetch logs. Status code: {}".format(resp.status_code))
            except Exception as e:
                st.warning("Error fetching logs: {}".format(str(e)))
    else:
        st.error("Could not establish a session. Please log in again.")
        st.stop()

    if modality != "tabular":
        render_embedding_analyze_ui(modality, active_project, user_api_key)
    elif not logs:
        st.info(f"No real-time logs found for {active_project} yet.")
    else:
        # Convert logs to DataFrame
        logs_df = pd.DataFrame(logs)  # columns: input_data, score, is_ood
        # Reverse to have chronological order (oldest first, but the API returns newest first)
        logs_df = logs_df.iloc[::-1].reset_index(drop=True)

        # --- DATA PROCESSING ---
        parsed_features = logs_df["input_data"].apply(json.loads)
        features_df = pd.json_normalize(parsed_features)
        full_df = pd.concat([features_df, logs_df[["score", "is_ood"]]], axis=1)
        full_df['Timeline'] = [f"Window Req #{i+1}" for i in range(len(full_df))]

        total_requests = len(full_df)
        anomalies = full_df["is_ood"].sum()
        anomaly_rate = (anomalies / total_requests) * 100 if total_requests > 0 else 0

    # ... (the rest of your code for tabs, metrics, charts remains unchanged)
        # --- CUSTOM ADAPTIVE METRICS ---
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
                <div class="premium-card">
                    <div class="icon-box blue-glow">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" width="32" height="32"><path stroke-linecap="round" stroke-linejoin="round" d="M21.75 17.25v-.228a4.5 4.5 0 0 0-.12-1.03l-2.268-9.64a3.375 3.375 0 0 0-3.285-2.602H7.923a3.375 3.375 0 0 0-3.285 2.602l-2.268 9.64a4.5 4.5 0 0 0-.12 1.03v.228m19.5 0a3 3 0 0 1-3 3H5.25a3 3 0 0 1-3-3m19.5 0a3 3 0 0 0-3-3H5.25a3 3 0 0 0-3 3m16.5 0h.008v.008h-.008v-.008Zm-3 0h.008v.008h-.008v-.008Z" /></svg>
                    </div>
                    <div><div class="card-title">Recent API Traffic</div><div class="card-value">{total_requests:,}</div></div>
                </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
                <div class="premium-card">
                    <div class="icon-box red-glow">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" width="32" height="32"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3Z" /></svg>
                    </div>
                    <div><div class="card-title">Anomalies Detected</div><div class="card-value" style="color: #EF4444;">{anomalies:,}</div></div>
                </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
                <div class="premium-card">
                    <div class="icon-box purple-glow">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" width="32" height="32"><path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z" /></svg>
                    </div>
                    <div><div class="card-title">Failure Rate</div><div class="card-value">{anomaly_rate:.2f}%</div></div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # --- UI TABS ---
        tab1, tab3, tab2 = st.tabs(["⚡ Real-Time Telemetry", "📊 Batch Drift Analysis", "🗄️ Audit Vault"])

        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            for feature in features_df.columns:
                is_numeric = pd.api.types.is_numeric_dtype(full_df[feature].dropna())
                
                fig = go.Figure()

                # --- 1A: CONTINUOUS FEATURE VISUALS (Existing) ---
                if is_numeric and feature in fences:
                    q1 = fences[feature]["q1"]
                    q3 = fences[feature]["q3"]
                    
                    fig.add_hrect(
                        y0=q1, y1=q3, line_width=0, fillcolor="#10B981", opacity=0.1,
                        annotation_text="Normal Bounds (IQR)", annotation_position="top left",
                        annotation_font_color="#10B981"
                    )
                    fig.add_hline(y=q3, line_dash="dash", line_color="#94A3B8", opacity=0.5)
                    fig.add_hline(y=q1, line_dash="dash", line_color="#94A3B8", opacity=0.5)

                    fig.add_trace(go.Scatter(
                        x=full_df['Timeline'], y=full_df[feature],
                        mode='lines+markers', name='API Traffic',
                        line=dict(color='#3B82F6', width=3), 
                        marker=dict(size=8, color='#3B82F6')
                    ))

                    anomalies_only = full_df[(full_df['is_ood'] == 1) & (full_df[feature].notna())]
                    if not anomalies_only.empty:
                        fig.add_trace(go.Scatter(
                            x=anomalies_only['Timeline'], y=anomalies_only[feature],
                            mode='markers', name='Critical Anomaly',
                            marker=dict(color='#EF4444', size=14, symbol='star-diamond', line=dict(color='white', width=2))
                        ))
                        
                    fig.update_layout(yaxis=dict(showgrid=True, title=f"{feature} Value"))

                # --- 1B: CATEGORICAL FEATURE VISUALS (New) ---
                elif not is_numeric:
                    fig.add_trace(go.Scatter(
                        x=full_df['Timeline'], y=full_df[feature].astype(str),
                        mode='markers', name='Category Stream',
                        marker=dict(size=12, color='#A855F7', symbol='square', opacity=0.8)
                    ))
                    fig.update_layout(yaxis=dict(type='category', title=f"{feature} Categories"))
                
                # Skip rendering if we don't have data/fences
                else:
                    continue

                # --- COMMON LAYOUT ---
                fig.update_layout(
                    title=dict(text=f"Live Feed: <b>{feature.upper()}</b>", font=dict(size=18)),
                    xaxis=dict(showgrid=True, title="", categoryorder='array', categoryarray=full_df['Timeline']),
                    hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40), height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                st.markdown("<hr style='opacity: 0.2; margin-bottom: 30px;'>", unsafe_allow_html=True)

        with tab3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p style='color: var(--text-color); opacity: 0.8;'>Evaluating long-term distributional shifts across continuous and categorical data.</p><br>", unsafe_allow_html=True)
            
            for feature in features_df.columns:
                if feature.lower() == "class":
                    continue
                    
                current_data = full_df[feature].dropna()
                if len(current_data) < 2:
                    continue

                is_numeric = pd.api.types.is_numeric_dtype(current_data)

                # --- 3A: CONTINUOUS KS-TEST VISUALS (Existing) ---
                if is_numeric and feature in fences:
                    q1 = fences[feature]["q1"]
                    q3 = fences[feature]["q3"]
                    
                    mu = (q1 + q3) / 2.0
                    sigma = max((q3 - q1) / 1.34896, 0.1) 
                    
                    ks_stat, p_value = stats.kstest(current_data.values, 'norm', args=(mu, sigma))
                    drift_detected = "YES" if p_value < 0.05 else "NO"
                    drift_color = "#EF4444" if drift_detected == "YES" else "#10B981"
                    
                    # (Your existing metric cards)
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.markdown(f"<div class='mini-card'><div class='card-title'>KS Stat (Max Gap)</div><div class='mini-value'>{ks_stat:.4f}</div></div>", unsafe_allow_html=True)
                    mc2.markdown(f"<div class='mini-card'><div class='card-title'>P-Value</div><div class='mini-value'>{p_value:.4f}</div></div>", unsafe_allow_html=True)
                    mc3.markdown(f"<div class='mini-card'><div class='card-title'>Drift Type</div><div class='mini-value'>Continuous</div></div>", unsafe_allow_html=True)
                    mc4.markdown(f"<div class='mini-card'><div class='card-title'>Statistical Drift?</div><div class='mini-value' style='color: {drift_color};'>{drift_detected}</div></div>", unsafe_allow_html=True)
                    
                    # Draw Bell Curves
                    x_min, x_max = min(current_data.min(), mu - 4*sigma), max(current_data.max(), mu + 4*sigma)
                    x_vals = np.linspace(x_min, x_max, 200)
                    baseline_pdf = stats.norm.pdf(x_vals, mu, sigma)
                    
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Scatter(
                        x=x_vals, y=baseline_pdf, fill='tozeroy', name='Expected Baseline Shape',
                        line=dict(color='#10B981', width=2, dash='dash'), fillcolor='rgba(16, 185, 129, 0.1)'
                    ))
                    
                    try:
                        kde = stats.gaussian_kde(current_data)
                        fig_dist.add_trace(go.Scatter(
                            x=x_vals, y=kde(x_vals), fill='tozeroy', name='Actual Production Shape',
                            line=dict(color='#3B82F6', width=3), fillcolor='rgba(59, 130, 246, 0.3)'
                        ))
                    except np.linalg.LinAlgError:
                        pass
                        
                    fig_dist.update_layout(yaxis_title="Probability Density")

                # --- 3B: CATEGORICAL FREQUENCY VISUALS (New) ---
                elif not is_numeric:
                    # Calculate production frequencies
                    val_counts = current_data.value_counts(normalize=True).reset_index()
                    val_counts.columns = ['Category', 'Frequency']
                    
                    mc1, mc2 = st.columns(2)
                    mc1.markdown(f"<div class='mini-card'><div class='card-title'>Unique Categories</div><div class='mini-value'>{len(val_counts)}</div></div>", unsafe_allow_html=True)
                    mc2.markdown(f"<div class='mini-card'><div class='card-title'>Drift Evaluation Engine</div><div class='mini-value' style='color: #A855F7;'>PSI Backend Evaluated</div></div>", unsafe_allow_html=True)

                    # Draw Bar Chart
                    fig_dist = go.Figure(go.Bar(
                        x=val_counts['Category'].astype(str),
                        y=val_counts['Frequency'],
                        marker_color='#A855F7',
                        opacity=0.8,
                        name="Production Frequencies",
                        text=(val_counts['Frequency'] * 100).round(1).astype(str) + '%',
                        textposition='auto'
                    ))
                    fig_dist.update_layout(yaxis_title="Observed Frequency (%)", yaxis=dict(tickformat=".0%"))
                
                else:
                    continue

                # --- COMMON LAYOUT ---
                fig_dist.update_layout(
                    title=dict(text=f"Distribution Analysis: <b>{feature.upper()}</b>", font=dict(size=18)),
                    xaxis_title=f"{feature} Values/Categories",
                    hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40), height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_dist, use_container_width=True, theme="streamlit")
                st.markdown("<hr style='opacity: 0.2; margin-bottom: 30px;'>", unsafe_allow_html=True)

        with tab2:
            st.markdown("<br><h3>Encrypted SQLite Vault</h3>", unsafe_allow_html=True)
            display_df = full_df[['Timeline', 'is_ood', 'score'] + list(features_df.columns)]
            
            def highlight_anomalies(row):
                return ['background-color: rgba(239, 68, 68, 0.15)' if row.is_ood else '' for _ in row]
            
            st.dataframe(display_df.style.apply(highlight_anomalies, axis=1), use_container_width=True)

    # --- 6. DANGER ZONE (Model Deletion) ---
    st.markdown("<br><hr style='opacity: 0.2;'>", unsafe_allow_html=True)

    with st.expander("⚙️ Danger Zone: Stop Monitoring"):
        st.warning(f"Warning: This will permanently delete the monitoring pipeline and all associated telemetry for **{active_project}**.")
        
        if st.button("🗑️ Delete Model & Stop Monitoring", type="primary", use_container_width=True):
            with st.spinner("Wiping model data from backend database..."):

                user_api_key = mint_session_token(user_info)
                
                safe_model_id = urllib.parse.quote(active_project.strip())
                headers = {"Authorization": f"Bearer {user_api_key}"}
                
                response = requests.delete(
                    f"{BACKEND_URL}/models/{safe_model_id}", 
                    headers=headers
                )
                
                if response.status_code == 200:
                    st.success(f"Successfully stopped monitoring {active_project}. All data wiped.")
                    time.sleep(1)
                    
                    if "selected_model" in st.session_state:
                        del st.session_state["selected_model"]
                    
                    st.rerun()
                else:
                    st.error(f"Failed to delete model: {response.text}")
# ---------------------------------------------------------
# 5. ONBOARDING / LOCKOUT (If they have no models)
# ---------------------------------------------------------
else:
    st.title("Welcome to Drift Sentinel")
    user_api_key = mint_session_token(user_info)

    # ==========================================
    # ONBOARDING: LOGGED IN, NO MODELS YET
    # ==========================================
    st.success("You're logged in! Let's get your first model online.")
    st.markdown("### Start Monitoring First Model")
    st.write("Upload your training datasets to start monitoring the model.")

    onboarding_modality = st.radio(
        "Model Type", ["Tabular", "Text", "Image"], key="onboarding_modality", horizontal=True
    )

    if onboarding_modality == "Text":
        render_embedding_fit_ui("text", user_api_key, key_prefix="onboarding")
    elif onboarding_modality == "Image":
        render_embedding_fit_ui("image", user_api_key, key_prefix="onboarding")
    else:
        new_model_id = st.text_input("New Model ID (e.g : fraud_detection_v1)", key="new_proj")

        uploaded_files = st.file_uploader(
            "Upload Training Data (CSV / DAT) - Max 2GB total",
            type=["csv", "dat"],
            key="new_file",
            accept_multiple_files=True
        )

        st.warning(
            "⚠️ **Notice:** If your combined training files approach or exceed the max 2GB limit, "
            "please upload your primary dataset first, then upload subsequent files sequentially."
        )

        # ==========================================
        # NEW: SMART SCHEMA CONFIGURATION UI
        # ==========================================
        schema_mapping = {}

        if uploaded_files:
            st.markdown("<br>🗂️ Step 2: Verify Data Schema", unsafe_allow_html=True)

            # 1. Load full files and take an unbiased random sample
            train_dfs = []
            for f in uploaded_files:
                train_dfs.append(read_uploaded_file(f))

            # Combine all uploaded files into one massive dataframe
            combined_df = pd.concat(train_dfs, ignore_index=True)
            st.session_state.combined_df = combined_df  # store to avoid recompute on button click


            preview_df = combined_df.sample(
                n=min(10000,len(combined_df)), random_state=42
            ).reset_index(drop=True)

            # 2. Ask the backend for mathematical recommendations
            with st.spinner("Analyzing dataset signals..."):
                headers = {"Authorization": f"Bearer {user_api_key}"}
                raw_dict = preview_df.to_dict(orient="list")

                # Scrub every Pandas NaN/NaT and replace with standard Python None
                clean_dict = {
                    col: [None if pd.isna(v) else v for v in vals]
                    for col, vals in raw_dict.items()
                }

                payload = {"reference_data": clean_dict}

                try:
                    profile_response = requests.post(f"{BACKEND_URL}/profile", json=payload, headers=headers)
                    if profile_response.status_code == 200:
                        smart_profiles = {p["name"]: p for p in profile_response.json()}
                    else:
                        smart_profiles = {}
                        st.error(f"Profiler Error: {profile_response.text}")
                except requests.exceptions.ConnectionError:
                    smart_profiles = {}
                    st.warning("Could not reach profiler API. Using default schema.")

            st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)

            # 3. Build the Stateful UI using the AI's recommendations
            for col in preview_df.columns:
                col_intel = smart_profiles.get(col, {})
                monitor = col_intel.get("monitor", "Review")
                reason = col_intel.get("reason", "Could not classify — recommend manual review")

                # Determine the 'suggested' index from the profiler
                if monitor is True:
                    suggested_idx = 0
                elif monitor == "Categorical":
                    suggested_idx = 1
                else:
                    suggested_idx = 2

                state_key = f"schema_{col}"

                # If this is a new run OR the profiler recommendation changed, update the state
                if state_key not in st.session_state or st.session_state.get(f"last_rec_{col}") != suggested_idx:
                    st.session_state[state_key] = suggested_idx
                    st.session_state[f"last_rec_{col}"] = suggested_idx

                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"**{col}**")
                    st.caption(f"Reason: {reason}")

                with c2:
                    if monitor == "Review":
                        st.warning(f"⚠️ **{col}** needs review.")

                    role_options = [
                        "📊 Continuous Feature (Monitor for Drift)",
                        "🔠 Categorical Feature (Monitor for Drift)",
                        "📝 Ignore (Unique IDs / Free Text / Target)"
                    ]

                    current_index = st.session_state[state_key]
                    if isinstance(current_index, str):
                        current_index = role_options.index(current_index) if current_index in role_options else 0
                        st.session_state[state_key] = current_index

                    role = st.selectbox(
                        "Column Role",
                        options=role_options,
                        index=current_index,
                        key=state_key,
                        label_visibility="collapsed"
                    )
                    schema_mapping[col] = role

            st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)

        # ==========================================
        # THE START MONITORING BUTTON (with schema)
        # ==========================================
        if st.button("Start Monitoring Model", type="primary", key="unlocked_btn"):
            if not new_model_id:
                st.warning("Please provide a Model ID.")
            elif not uploaded_files:
                st.warning("Please upload at least one CSV or DAT file.")
            elif not schema_mapping:
                st.warning("Please verify the data schema above before starting monitoring.")
            else:
                with st.spinner("Processing datasets and starting monitoring..."):
                    try:
                        # Retrieve combined_df from session state to avoid recompute
                        combined_df = st.session_state.get("combined_df", pd.DataFrame())

                        # Split confirmed schema into continuous vs categorical
                        continuous_cols = [
                            col for col, role in schema_mapping.items()
                            if "Continuous" in role and col in combined_df.columns
                        ]
                        categorical_cols = [
                            col for col, role in schema_mapping.items()
                            if "Categorical" in role and col in combined_df.columns
                        ]

                        if not continuous_cols and not categorical_cols:
                            st.warning(
                                "No columns selected for monitoring. "
                                "Please mark at least one column as Continuous or Categorical."
                            )
                            st.stop()

                        # Prepare continuous data

                        continuous_df = combined_df[continuous_cols].select_dtypes(
                            include=['number']
                        ).dropna()
                        if len(continuous_df) > DEFAULT_BASELINE_SAMPLE_SIZE:
                            continuous_df = continuous_df.sample(n=DEFAULT_BASELINE_SAMPLE_SIZE, random_state=42)

                        # Prepare categorical data
                        categorical_df = combined_df[categorical_cols].dropna() \
                            if categorical_cols else pd.DataFrame()
                        if len(categorical_df) > DEFAULT_BASELINE_SAMPLE_SIZE:
                            categorical_df = categorical_df.sample(n=DEFAULT_BASELINE_SAMPLE_SIZE, random_state=42)

                        # NaN scrub both payloads before sending
                        def scrub(df):
                            if df.empty:
                                return {}
                            raw = df.to_dict(orient="list")
                            return {
                                col: [None if pd.isna(v) else v for v in vals]
                                for col, vals in raw.items()
                            }

                        payload = {
                            "reference_data": scrub(continuous_df),
                            "categorical_data": scrub(categorical_df)
                        }

                        safe_model_id = new_model_id.strip().replace(" ", "%20")
                        headers = {"Authorization": f"Bearer {user_api_key}"}

                        response = requests.post(f"{BACKEND_URL}/fit/{safe_model_id}", json=payload, headers=headers)

                        if response.status_code == 200:
                            result = response.json()
                            st.success(
                                f"Successfully started monitoring '{new_model_id}'! "
                                f"Watching {len(continuous_cols)} continuous and "
                                f"{len(categorical_cols)} categorical features."
                            )
                            st.balloons()
                            time.sleep(1.5)
                            st.session_state.selected_model = new_model_id

                            st.rerun()
                        else:
                            st.error(f"Backend Error: {response.text}")

                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")

    st.write("---")
    if st.button("Logout", key="logout_fallback"):
        authenticator.logout()

