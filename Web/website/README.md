# HNRS — local full-stack setup

The deployed app runs entirely on the hosted stack (TanStack Start + Lovable Cloud
Postgres) and uses an AI vision bridge for recognition, because the hosted edge
runtime cannot execute TensorFlow or PyTorch.

This folder contains the local variant that loads **your** trained weights.

```text
website/
  backend/
    server.py                  FastAPI inference API used by the web UI (port 8000)
    requirements.txt
    server.js                  Express API (predict + history, SQLite persistence)
    package.json
    database/                  hnrs.sqlite created on first run
    models_bridge/
      inference.py             Loads .keras / .pt weights and returns JSON
      requirements.txt
```

## Run it

```bash
cd website/backend

# inference server the UI talks to (http://127.0.0.1:8000/predict)
pip install -r requirements.txt
python server.py

# history API (optional, http://localhost:5000)
npm install
pip install -r models_bridge/requirements.txt
npm start
```

`POST /predict` takes `{ task, imageDataUrl }` where `task` is `auto`, `digit`,
`letter` or `text`. The image is binarised, split into characters with connected
components and each glyph is normalised MNIST-style (20×20 aspect fit, centred by
centre of mass) before classification. With `auto`, every character goes through
both `digit_cnn_model.keras` and `letter_customcnn.pth` and the more confident
answer wins, so mixed strings such as `12A` are read correctly. `text` runs
`best_crnn_model.pt` over the whole line with CTC greedy decoding.

Set `GEMINI_API_KEY` to enable the optional Gemini post-check
(`pip install google-generativeai`); without the key the local models are used as-is.

Place your trained weights in the repository-level `Models/` folder:

```text
Models/
  digit_cnn_model.keras
  letter_cnn_model.keras
  text_crnn_model.pt
```

## API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Backend status and discovered model files |
| POST | `/api/predict` | `{ imageDataUrl, model }` -> prediction, confidence, class probabilities, latency |
| GET | `/api/history` | Last 500 stored predictions |
| POST | `/api/history` | Store a prediction record |
| DELETE | `/api/history/:id` | Delete a record |

To point the web UI at this backend instead of the hosted inference bridge, set
`VITE_HNRS_API_URL=http://localhost:5000` and call `/api/predict` from
`src/lib/inference.functions.ts`.