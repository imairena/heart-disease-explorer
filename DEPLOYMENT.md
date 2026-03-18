# Deploy Heart Disease Explorer to Streamlit Community Cloud

Follow these steps to deploy your app for free (takes ~5 minutes).

---

## Step 1: Create a GitHub Repository

1. Go to [github.com](https://github.com) and sign in (or create an account).
2. Click the **+** in the top-right → **New repository**.
3. Name it `heart-disease-explorer` (or any name).
4. Choose **Public**, then click **Create repository**.
5. **Do not** initialize with README (we already have files).

---

## Step 2: Push Your Code to GitHub

Open Terminal and run:

```bash
cd /Users/ianmairenajarquin/heart-disease-app

# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Heart Disease Explorer app"

# Add your GitHub repo as remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-explorer.git

# Push (use main or master depending on your default branch)
git branch -M main
git push -u origin main
```

---

## Step 3: Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Sign in with your **GitHub** account.
3. Click **New app**.
4. Fill in:
   - **Repository**: `YOUR_USERNAME/heart-disease-explorer`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: (optional) e.g. `heart-disease-explorer`
5. Click **Deploy**.

---

## Step 4: Wait for Deployment

- First deploy takes 2–5 minutes.
- Streamlit will install dependencies from `requirements.txt`.
- When it’s done, you’ll get a URL like: `https://heart-disease-explorer-xxx.streamlit.app`

---

## Step 5: Configure API Secrets

The AI Medical Assistant feature requires a Google Gemini API key to function. You must add this key to your Streamlit Cloud environment to avoid committing it to GitHub.

1. Go to your dashboard at [share.streamlit.io](https://share.streamlit.io).
2. Click the **⋮ (three dots)** menu next to your `heart-disease-explorer` app and select **Settings** (or click "Manage App" from inside the app, and click the three dots in the corner).
3. Select **Secrets** from the settings menu.
4. In the text area, paste your API key in TOML format:
   ```toml
   GEMINI_API_KEY = "your_actual_api_key_here"
   ```
5. Click **Save**. Your app will securely read the key and the AI Chatbot will be fully operational.

---

## Important: Data Must Be in the Repo

The app uses `data/heart_disease_cleaned.csv`. Make sure it’s committed:

```bash
git add data/heart_disease_cleaned.csv
git commit -m "Add cleaned dataset"
git push
```

If you prefer to use raw data, also commit the processed `.data` files in `data/`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| App crashes on load | Check that `data/heart_disease_cleaned.csv` exists in the repo. |
| Module not found | Ensure `requirements.txt` lists all dependencies. |
| Slow first load | Normal; Streamlit Cloud spins up on first visit. |
| Need to redeploy | Push new commits; Streamlit redeploys automatically. |
| Chatbot not responding | Ensure your `GEMINI_API_KEY` is configured correctly in App Settings -> Secrets. |

---

## Automated CI/CD (GitHub Actions)

This repository includes a preconfigured GitHub Actions continuous integration pipeline (`.github/workflows/ci.yml`). This ensures that your deployed code meets code quality standards by automatically running **`pytest`** and **`flake8`** on every push to your main branch.

---

## Pre-configured Theme

The code is pre-packaged with Streamlit custom styling (`.streamlit/config.toml`). The color scheme and design values will be automatically parsed when rendering on Streamlit Community Cloud and matches the layout design properly.
