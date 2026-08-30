# GitHub and Streamlit Cloud Deployment

## 1. Create a new GitHub repository

1. Log in to GitHub.
2. Click the green `New repository` button.
3. Choose a repository name such as `customer-segmentation-dashboard`.
4. Select `Public` or `Private`.
5. Click `Create repository`.

## 2. Initialize local git repository

```bash
git init
git add .
git commit -m "Initial commit: customer segmentation dashboard"
```

## 3. Connect to GitHub

```bash
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```

## 4. Deploy to Streamlit Cloud

1. Open https://streamlit.io/cloud.
2. Sign in with GitHub.
3. Click `New app`.
4. Select the repository and branch.
5. Choose the main app file: `app.py`.
6. Set the Python version to 3.11 or newer.
7. Click `Deploy`.
8. Wait for the build to finish.
9. Share the live URL.

## 5. Deployment configuration

- Ensure `requirements.txt` is included in the repo.
- Ensure `app.py` is the entry point.
- Keep `online_retail_II.xlsx` in the repo if you want local sample data, or point the app to a remote URL.

## 6. Optional production setup

- Add environment variables for secrets.
- Configure custom domain if needed.
- Enable authentication or access restrictions if required.
