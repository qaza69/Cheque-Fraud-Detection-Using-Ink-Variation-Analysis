# Deployment guide

```
 Browser  --->  Vercel (React app, frontend/)  --->  Render (FastAPI, backend/)
                static files, free                    runs the ink-analysis models, free plan
```

The API is a Render web service defined by [`render.yaml`](render.yaml); the website is a
Vite/React app in [`frontend/`](frontend) hosted on Vercel. Both have free plans, and both
can sign in with your GitHub account.

## 0. Before you start

* The code and the files the server needs (`backend/artifacts/`: blank bank templates and the
  trained Random Forest model, about 33 MB) must be on GitHub, because Render builds from the repository.
* Demo cheques are read from `data/samples/` (already in the repository).
* Only the **Random Forest** model is served on the free plan. The two neural models need PyTorch
  (about 2 GB of RAM); see "Serving all three models" below.

## 1. Deploy the API on Render

1. Sign up at <https://render.com> with GitHub ("Continue with GitHub") and allow it to read this repository.
2. **New +** -> **Blueprint** -> choose this repository. Render reads `render.yaml`.
3. It asks for `FRONTEND_ORIGINS`. Enter `http://localhost:5173` for now (you will replace it in step 3).
   Leave `CORS_ORIGIN_REGEX` empty.
4. **Apply**. The first build takes 3-6 minutes.
5. Open `https://<service-name>.onrender.com/health`. It should show
   `{"status":"ok","models":["classical"],...}`. `/docs` shows the interactive API documentation.
   Note the URL: this is your **API URL**.

## 2. Deploy the website on Vercel

1. Sign up at <https://vercel.com> with GitHub.
2. **Add New...** -> **Project** -> import this repository.
3. Set **Root Directory** to `frontend` (Framework Preset: Vite is detected automatically).
4. Under **Environment Variables** add `VITE_API_URL` = your Render API URL, with no trailing slash,
   for example `https://cheque-fraud-api.onrender.com`.
5. **Deploy**. You get a URL such as `https://cheque-fraud-xyz.vercel.app`.

## 3. Let the website call the API (CORS)

In Render: your service -> **Environment** -> set `FRONTEND_ORIGINS` to the Vercel URL
(`https://cheque-fraud-xyz.vercel.app`, no trailing slash; several URLs can be comma-separated).
Optional: set `CORS_ORIGIN_REGEX` to `^https://cheque-fraud-.*\.vercel\.app$` so Vercel preview
deployments work too. Save; Render restarts the service.

Open the Vercel URL, pick a demo cheque and press **Detect fraud**.

## Things to know

* **Cold start.** Render's free plan puts the service to sleep after 15 minutes without traffic;
  the first request then takes 30-60 seconds. The website shows "Waking the server up" while it waits.
  Open the site a minute before a demonstration.
* **Memory.** The Random Forest path uses about 270 MB of the 512 MB free limit. Only one analysis runs at a
  time (`MAX_CONCURRENT=1`); further requests wait.
* **Privacy.** Uploaded cheques are processed in memory and never stored or logged. Cheques contain account
  details: use only data you are allowed to share.
* **Limits.** Uploads up to 12 MB (`MAX_UPLOAD_MB`) and 40 megapixels (`MAX_MEGAPIXELS`).

## Local development

```bash
# terminal 1 - API (http://localhost:8000, docs at /docs)
pip install -r backend/requirements.txt
ENABLED_MODELS=classical uvicorn backend.main:app --reload --port 8000

# terminal 2 - website (http://localhost:5173)
cd frontend
cp .env.example .env.local        # VITE_API_URL=http://localhost:8000
npm install
npm run dev
```

Tests: `python -m pytest tests` (includes `tests/test_api.py`).

## Refreshing the shipped model

After retraining (`python -m src.train`) copy the new files into the repository and push:

```bash
python backend/prepare_artifacts.py --no-samples
git add backend/artifacts && git commit -m "Update model" && git push
```

Render redeploys automatically (`autoDeploy: true`).

## Serving all three models

On a paid Render plan with at least 2 GB of RAM: change the build command to
`pip install -r backend/requirements-torch.txt` and set `ENABLED_MODELS=classical,cnn,siamese`
(or leave it empty to serve every model whose weights are present).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Site says "server unreachable" | Render service asleep or `VITE_API_URL` wrong. Open `<API URL>/health` directly. |
| Browser console: CORS error | `FRONTEND_ORIGINS` does not match the Vercel URL exactly (scheme, no trailing slash). |
| Build fails on Render | Check the Python version (`PYTHON_VERSION`) and the pinned versions in `backend/requirements.txt`. |
| "Out of memory" on Render | More than one analysis at once, or a neural model enabled on the free plan. |
| Vercel shows an old API URL | `VITE_API_URL` is read at build time: change it, then redeploy. |
