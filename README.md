# TUDO — Fall Risk Assessment

An AI-powered gait analysis application that estimates fall risk (Low / Medium / High) from walking videos using pose estimation and a trained 1-D CNN model. No backend server required — inference runs directly inside the Streamlit app.

---

## Table of Contents

1. [How it works](#how-it-works)
2. [Repository structure](#repository-structure)
3. [Run locally](#run-locally)
4. [Deploy to Streamlit Cloud via GitHub](#deploy-to-streamlit-cloud-via-github)
   - [Step 1 — Create a GitHub repository](#step-1--create-a-github-repository)
   - [Step 2 — Push this folder as the repo root](#step-2--push-this-folder-as-the-repo-root)
   - [Step 3 — Connect to Streamlit Cloud](#step-3--connect-to-streamlit-cloud)
   - [Step 4 — Configure and launch](#step-4--configure-and-launch)
5. [App pages](#app-pages)
6. [Model parameters (sidebar)](#model-parameters-sidebar)
7. [Known limitations on Streamlit Cloud](#known-limitations-on-streamlit-cloud)
8. [Troubleshooting](#troubleshooting)

---

## How it works

1. A walking video is uploaded through the browser.
2. [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) extracts joint positions frame-by-frame.
3. A sliding window of 90 frames (3 seconds at 30 fps) is fed into **GaitCNNv3Soft** — a residual 1-D CNN with squeeze-and-excitation blocks.
4. The model outputs probabilities for three classes: **LOW**, **MEDIUM**, **HIGH** fall risk.
5. An annotated MP4 is generated with the risk label overlaid on every frame and returned for playback and download.

---

## Repository structure

```
.
├── app.py                          # Main Streamlit application (self-contained)
├── inference.py                    # Model definition + inference engine
├── requirements.txt                # All Python dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml                 # App theme (sage-green clinical palette)
├── models/
│   └── fall_risk_cnn_occu_v3_soft.pt   # Trained model weights (~2 MB)
└── assets/
    ├── logo.png
    ├── 1.jpg – 5.jpg               # Next-steps image grid
    └── Untitled design (1).jpg
```

> `assessments_history.json` and `last_analysis.json` are created automatically at runtime and are excluded from version control via `.gitignore`.

---

## Run locally

**Requirements:** Python 3.10 or 3.11 recommended.

```bash
# 1. Clone or download this repository
git clone https://github.com/<your-github-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Deploy to Streamlit Cloud via GitHub

Streamlit Cloud reads directly from a GitHub repository. Every `git push` to your connected branch triggers an automatic redeploy.

### Step 1 — Create a GitHub repository

1. Go to [github.com/new](https://github.com/new).
2. Give it a name (e.g. `tudo-fall-risk`).
3. Set visibility to **Public** (required for the free Streamlit Cloud tier) or **Private** (available on paid tiers).
4. **Do not** initialise with a README — you will push your own files.
5. Click **Create repository**.

---

### Step 2 — Push this folder as the repo root

Open a terminal **inside the `streamlit_deploy` folder** (the folder that contains `app.py`).

```bash
# Initialise a new git repository at this folder
git init

# Stage everything (the .gitignore will automatically exclude .venv, cache, etc.)
git add .

# Create the first commit
git commit -m "Initial commit: TUDO fall-risk Streamlit app"

# Point to your new GitHub repository
git remote add origin https://github.com/<your-github-username>/<your-repo-name>.git

# Push
git branch -M main
git push -u origin main
```

After this, your GitHub repository root should look like:

```
README.md
app.py
inference.py
requirements.txt
.gitignore
.streamlit/config.toml
models/fall_risk_cnn_occu_v3_soft.pt
assets/logo.png
assets/1.jpg
...
```

> **Important:** `app.py` must be at the repository root (not inside a subfolder) for Streamlit Cloud's default detection to work. If you place it in a subfolder, you must specify the path manually in Step 4.

---

### Step 3 — Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
2. Click **"New app"** (top-right).

---

### Step 4 — Configure and launch

Fill in the deployment form:

| Field | Value |
|---|---|
| **Repository** | `<your-github-username>/<your-repo-name>` |
| **Branch** | `main` |
| **Main file path** | `app.py` |
| **App URL** (optional) | Choose a custom subdomain, e.g. `tudo-fall-risk` |

Click **Deploy**.

Streamlit Cloud will:
1. Clone your repository.
2. Install all packages listed in `requirements.txt` (this takes 3–8 minutes the first time due to `torch` and `mediapipe`).
3. Start the app and give you a public URL:
   ```
   https://<your-chosen-subdomain>.streamlit.app
   ```

Your app is now live. Every future `git push origin main` will automatically trigger a redeploy.

---

### Updating the deployed app

```bash
# Make changes to your files, then:
git add .
git commit -m "describe your change"
git push origin main
```

Streamlit Cloud detects the push and redeploys within ~1 minute.

---

## App pages

| Page | Description |
|---|---|
| **Home** | Landing page with links to start an assessment or view history |
| **New Assessment** | Patient form (name, age, sex, diagnosis, previous falls) + video upload + analysis |
| **Results** | Annotated video playback, risk metrics bar, interactive probability timeline, per-window table, download button |
| **Past Assessments** | History of all submitted assessments with risk badge, name, age, date |
| **Next Steps** | Risk-level-specific clinical guidance with image grid; active risk level is highlighted |
| **Live Camera** | Real-time webcam analysis via streamlit-webrtc (see limitations below) |

---

## Model parameters (sidebar)

Open the sidebar (arrow on the top-left of the app) to adjust:

| Parameter | Default | Description |
|---|---|---|
| **HIGH Threshold** | 0.64 | P(HIGH) must exceed this to classify a window as HIGH risk |
| **MEDIUM Threshold** | 0.33 | P(MEDIUM) must exceed this to classify a window as MEDIUM |
| **Aggregation** | p90 | How recent window scores are combined — `p90`, `p75`, `max`, or `mean` |
| **Min HIGH Windows** | 2 | Minimum consecutive HIGH windows before the final verdict is HIGH |
| **Show Pose Skeleton** | On | Draw MediaPipe landmark skeleton on the annotated video |

---

## Known limitations on Streamlit Cloud

### Webcam (Live Camera page)
The live webcam page uses **WebRTC** (via `streamlit-webrtc`). WebRTC requires a direct peer-to-peer connection between your browser and the server.

- **Local deployment:** Works out of the box.
- **Streamlit Cloud:** May fail on networks with strict NAT or firewalls because no TURN server is configured. The video upload analysis page is not affected and will work fully on Cloud.

### Ephemeral filesystem
Streamlit Cloud does not provide persistent storage. Files written to disk (`assessments_history.json`, `last_analysis.json`) are lost when the app restarts or redeploys. History will start fresh after each restart. To persist data across restarts you would need an external database (e.g. Supabase, Firebase, or a Streamlit-compatible storage backend).

### CPU-only inference
Streamlit Cloud free tier runs on CPU. Processing a 30-second video (pose estimation + CNN inference) typically takes **60–120 seconds**. For best results:
- Keep videos to **10–20 seconds**.
- Ensure the subject is fully visible from head to feet.
- Use good lighting and a plain background.

---

## Troubleshooting

**App fails to start — `ModuleNotFoundError: No module named 'inference'`**
Make sure `inference.py` is at the repository root alongside `app.py`, not inside a subfolder.

**`torch.load` warning about `weights_only`**
This is a non-fatal deprecation warning from PyTorch ≥ 2.0. The app explicitly passes `weights_only=False` to suppress it. No action needed.

**`mediapipe` version conflict**
`requirements.txt` pins `mediapipe==0.10.14`. Do not upgrade it without testing — the pose landmark API changed in 0.10.x and later versions may break feature extraction.

**Annotated video has no audio**
Expected — the inference pipeline processes video only. Audio is not used or preserved.

**Video plays in VLC but not in the browser**
Streamlit's `st.video` requires a browser-compatible container. The app encodes output with `libx264` + `yuv420p` which is universally supported. If you supply an input video in a format that OpenCV cannot decode (e.g. some `.mov` encodings), pre-convert it with `ffmpeg -i input.mov -vcodec libx264 output.mp4`.

**`streamlit-webrtc` shows "Please wait..." indefinitely on Cloud**
This is the TURN server limitation described above. Use the video upload page instead.
