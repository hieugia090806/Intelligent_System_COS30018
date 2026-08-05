/**
 * HNRS local Express backend.
 * Bridges the web UI to the real Keras / PyTorch weights in ../../Models
 * and persists prediction history in database/hnrs.sqlite.
 *
 *   cd website/backend
 *   npm install
 *   npm start        # http://localhost:5000
 */
const path = require("node:path");
const fs = require("node:fs");
const { spawn } = require("node:child_process");
const express = require("express");
const cors = require("cors");
const Database = require("better-sqlite3");

const app = express();
const PORT = process.env.PORT || 5000;
const ROOT = path.resolve(__dirname, "..", "..");
const MODELS_DIR = path.join(ROOT, "Models");
const DB_DIR = path.join(__dirname, "database");

fs.mkdirSync(DB_DIR, { recursive: true });
const db = new Database(path.join(DB_DIR, "hnrs.sqlite"));
db.exec(`
  CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    input_type TEXT NOT NULL,
    model_used TEXT NOT NULL,
    predicted_text TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    execution_time_ms INTEGER NOT NULL,
    image_data_url TEXT
  );
`);

app.use(cors());
app.use(express.json({ limit: "12mb" }));

app.get("/api/health", (_req, res) => {
  const models = fs.existsSync(MODELS_DIR)
    ? fs.readdirSync(MODELS_DIR).filter((f) => f.endsWith(".keras") || f.endsWith(".pt"))
    : [];
  res.json({ status: "ok", modelsDir: MODELS_DIR, models });
});

/** POST /api/predict { imageDataUrl, model } -> real model inference via Python bridge. */
app.post("/api/predict", (req, res) => {
  const { imageDataUrl, model } = req.body || {};
  if (!imageDataUrl || !model) {
    return res.status(400).json({ error: "imageDataUrl and model are required" });
  }

  const script = path.join(__dirname, "models_bridge", "inference.py");
  const python = process.env.PYTHON_BIN || "python";
  const child = spawn(python, [script, "--model", model, "--models-dir", MODELS_DIR]);

  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => (stdout += chunk));
  child.stderr.on("data", (chunk) => (stderr += chunk));
  child.on("error", (error) => res.status(500).json({ error: error.message }));
  child.on("close", (code) => {
    if (code !== 0) {
      console.error("[HNRS] python bridge failed:", stderr);
      return res.status(500).json({ error: "Inference failed", detail: stderr });
    }
    try {
      res.json(JSON.parse(stdout));
    } catch {
      res.status(500).json({ error: "Invalid bridge response", raw: stdout });
    }
  });

  child.stdin.write(JSON.stringify({ imageDataUrl }));
  child.stdin.end();
});

app.get("/api/history", (_req, res) => {
  res.json(
    db.prepare("SELECT * FROM prediction_history ORDER BY id DESC LIMIT 500").all(),
  );
});

app.post("/api/history", (req, res) => {
  const {
    input_type,
    model_used,
    predicted_text,
    confidence_score,
    execution_time_ms,
    image_data_url,
  } = req.body || {};
  const info = db
    .prepare(
      `INSERT INTO prediction_history
         (input_type, model_used, predicted_text, confidence_score, execution_time_ms, image_data_url)
       VALUES (?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input_type,
      model_used,
      predicted_text,
      Number(confidence_score) || 0,
      Number(execution_time_ms) || 0,
      image_data_url ?? null,
    );
  res.status(201).json(
    db.prepare("SELECT * FROM prediction_history WHERE id = ?").get(info.lastInsertRowid),
  );
});

app.delete("/api/history/:id", (req, res) => {
  db.prepare("DELETE FROM prediction_history WHERE id = ?").run(Number(req.params.id));
  res.status(204).end();
});

app.listen(PORT, () => {
  console.log(`HNRS backend listening on http://localhost:${PORT}`);
});