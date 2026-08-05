"""HNRS Python inference bridge.

Reads {"imageDataUrl": "data:image/png;base64,..."} from stdin, runs the
requested model from ../../../Models and prints a JSON result to stdout:

  {"predictedText": "7", "confidence": 0.98, "probabilities": [...], "latencyMs": 42}

Usage (called by server.js):
  python inference.py --model digit_cnn_model.keras --models-dir /path/to/Models
"""

import argparse
import base64
import io
import json
import os
import string
import sys
import time

import numpy as np
from PIL import Image, ImageOps

DIGITS = [str(i) for i in range(10)]
LETTERS = list(string.ascii_uppercase)


def labels_for(model_name: str):
    return DIGITS if "digit" in model_name.lower() else LETTERS


def decode_image(data_url: str) -> np.ndarray:
    """data URL -> (1, 28, 28, 1) float32 array normalised to 0..1 (white ink)."""
    encoded = data_url.split(",", 1)[-1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L")
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype="float32") / 255.0
    if array.mean() > 0.5:  # dark ink on light paper -> invert
        array = 1.0 - array
    return array.reshape(1, 28, 28, 1)


def predict_keras(path: str, tensor: np.ndarray) -> np.ndarray:
    from tensorflow import keras  # noqa: PLC0415

    model = keras.models.load_model(path)
    return np.asarray(model.predict(tensor, verbose=0))[0]


def predict_torch(path: str, tensor: np.ndarray) -> np.ndarray:
    import torch  # noqa: PLC0415

    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    with torch.no_grad():
        batch = torch.from_numpy(tensor.transpose(0, 3, 1, 2))
        logits = model(batch)
        probs = torch.softmax(logits, dim=-1)
    return probs.numpy()[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--models-dir", required=True)
    args = parser.parse_args()

    payload = json.loads(sys.stdin.read() or "{}")
    data_url = payload.get("imageDataUrl")
    if not data_url:
        print(json.dumps({"error": "imageDataUrl missing"}))
        return 1

    model_path = os.path.join(args.models_dir, args.model)
    if not os.path.exists(model_path):
        print(json.dumps({"error": f"model not found: {model_path}"}))
        return 1

    tensor = decode_image(data_url)
    started = time.time()
    if model_path.endswith(".keras") or model_path.endswith(".h5"):
        probs = predict_keras(model_path, tensor)
    else:
        probs = predict_torch(model_path, tensor)
    latency_ms = int((time.time() - started) * 1000)

    labels = labels_for(args.model)[: len(probs)]
    best = int(np.argmax(probs))
    print(
        json.dumps(
            {
                "predictedText": labels[best] if best < len(labels) else "?",
                "confidence": float(probs[best]),
                "probabilities": [
                    {"label": label, "probability": float(prob)}
                    for label, prob in zip(labels, probs)
                ],
                "latencyMs": latency_ms,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())