import os
import io
import base64
import string
from typing import List, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="HNRS Model Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIGIT_LABELS = [str(i) for i in range(10)]
LETTER_LABELS = list(string.ascii_uppercase)

def get_models_dir():
    candidates = [
        os.path.join(BASE_DIR, "Models"),
        os.path.join(BASE_DIR, "..", "Models"),
        os.path.join(BASE_DIR, "..", "..", "Models"),
        os.path.join(BASE_DIR, "..", "..", "..", "Models"),
    ]
    for path in candidates:
        abs_p = os.path.abspath(path)
        if os.path.exists(abs_p):
            return abs_p
    return os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Models"))

MODELS_DIR = get_models_dir()

class LetterCNN(nn.Module):
    """Same architecture as Notebook/EngText_CustomCNN.ipynb (letter_customcnn.pth)."""

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.10),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.15),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),
        )
        self.fc_features = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(1152, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, 26),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.stage3(self.stage2(self.stage1(x)))
        return self.softmax(self.classifier_head(self.fc_features(x)))

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(7, 1))
        )
        self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, 27)

    def forward(self, x):
        out = self.cnn(x)
        out = out.squeeze(2).permute(0, 2, 1)
        out, _ = self.rnn(out)
        out = self.fc(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
digit_model = None
letter_model = None
crnn_model = None

@app.on_event("startup")
def load_all_models():
    global digit_model, letter_model, crnn_model
    keras_path = os.path.join(MODELS_DIR, "digit_cnn_model.keras")
    if os.path.exists(keras_path):
        digit_model = tf.keras.models.load_model(keras_path)

    letter_path = os.path.join(MODELS_DIR, "letter_customcnn.pth")
    if os.path.exists(letter_path):
        try:
            state = torch.load(letter_path, map_location=device)
            if isinstance(state, nn.Module):
                letter_model = state
            else:
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                letter_model = LetterCNN()
                letter_model.load_state_dict(state)
            letter_model.to(device).eval()
        except Exception as e:
            print(f"Letter CNN load error: {e}")

    pt_path = os.path.join(MODELS_DIR, "best_crnn_model.pt")
    if os.path.exists(pt_path):
        try:
            checkpoint = torch.load(pt_path, map_location=device)
            crnn_model = CRNN().to(device)
            if isinstance(checkpoint, dict) and 'cnn' in checkpoint:
                crnn_model.cnn.load_state_dict(checkpoint['cnn'])
                crnn_model.rnn.load_state_dict(checkpoint['rnn'])
                crnn_model.fc.load_state_dict(checkpoint['fc'])
            else:
                crnn_model.load_state_dict(checkpoint)
            crnn_model.eval()
        except Exception as e:
            print(f"CRNN load error: {e}")

class InferenceRequest(BaseModel):
    task: str
    imageDataUrl: str

def call_gemini_fallback(image_bytes: bytes, local_prediction: str) -> str:
    """Optional post-check of the local prediction with Gemini (skipped without a key)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return local_prediction
    try:
        import google.generativeai as genai  # imported lazily: optional dependency

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        image = Image.open(io.BytesIO(image_bytes))

        prompt = (
            "Look at this handwritten image containing digits and/or English letters. "
            f"The local OCR model predicted: '{local_prediction}'. "
            "Please correct it if it's wrong, and output ONLY the exact final text string "
            "without any markdown, explanations, or extra words."
        )
        response = model.generate_content([image, prompt])
        corrected = response.text.strip()
        return corrected if corrected else local_prediction
    except Exception as e:
        print(f"Gemini Fallback Error: {e}")
        return local_prediction

def to_ink_mask(image_bytes: bytes) -> np.ndarray:
    """Decode a data URL payload into a binary mask where ink is 255 on a 0 background."""
    pil_img = Image.open(io.BytesIO(image_bytes))
    if pil_img.mode in ("RGBA", "LA", "P"):
        pil_img = pil_img.convert("RGBA")
        flat = Image.new("RGBA", pil_img.size, (0, 0, 0, 255))
        flat.alpha_composite(pil_img)
        pil_img = flat
    gray = np.array(pil_img.convert("L"))

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if mask.mean() > 127:  # dark ink on a light page -> invert so ink is white
        mask = 255 - mask
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

def segment_characters(mask: np.ndarray) -> List[List[int]]:
    """Left-to-right character boxes from connected components, merging split strokes."""
    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    min_area = max(20.0, 0.0002 * height * width)
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < min_area or w < 2 or h < 2:
            continue
        boxes.append([int(x), int(y), int(w), int(h)])

    boxes.sort(key=lambda b: b[0])
    merged: List[List[int]] = []
    for box in boxes:
        if merged:
            prev = merged[-1]
            overlap = min(prev[0] + prev[2], box[0] + box[2]) - max(prev[0], box[0])
            if overlap > 0.5 * min(prev[2], box[2]):
                x0 = min(prev[0], box[0])
                y0 = min(prev[1], box[1])
                x1 = max(prev[0] + prev[2], box[0] + box[2])
                y1 = max(prev[1] + prev[3], box[1] + box[3])
                merged[-1] = [x0, y0, x1 - x0, y1 - y0]
                continue
        merged.append(box)

    if not merged:
        merged = [[0, 0, width, height]]
    return merged

def to_mnist_tensor(crop: np.ndarray) -> np.ndarray:
    """MNIST-style normalisation: 20x20 aspect-preserving fit, centered by mass in 28x28."""
    ys, xs = np.nonzero(crop)
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.float32)
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = crop.shape
    if h > w:
        new_h, new_w = 20, max(1, int(round(w * 20 / h)))
    else:
        new_w, new_h = 20, max(1, int(round(h * 20 / w)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    canvas /= 255.0

    moments = cv2.moments(canvas)
    if moments["m00"] > 0:
        shift_x = 14 - moments["m10"] / moments["m00"]
        shift_y = 14 - moments["m01"] / moments["m00"]
        translation = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, translation, (28, 28))
    return canvas

def digit_probabilities(tensor: np.ndarray) -> Optional[np.ndarray]:
    if digit_model is None:
        return None
    return digit_model.predict(tensor.reshape(1, 28, 28, 1), verbose=0)[0]

def letter_probabilities(tensor: np.ndarray) -> Optional[np.ndarray]:
    if letter_model is None:
        return None
    batch = torch.tensor(tensor.reshape(1, 1, 28, 28), dtype=torch.float32).to(device)
    with torch.no_grad():
        return letter_model(batch).cpu().numpy()[0]

def classify_character(tensor: np.ndarray, task: str) -> dict:
    """Classify one normalised glyph. 'auto' runs both nets and keeps the confident one."""
    digit_probs = digit_probabilities(tensor) if task in ("digit", "auto") else None
    letter_probs = letter_probabilities(tensor) if task in ("letter", "auto") else None

    best_digit = float(np.max(digit_probs)) if digit_probs is not None else -1.0
    best_letter = float(np.max(letter_probs)) if letter_probs is not None else -1.0

    if digit_probs is not None and best_digit >= best_letter:
        probs, labels, model_used = digit_probs, DIGIT_LABELS, "digit_cnn_model.keras"
    elif letter_probs is not None:
        probs, labels, model_used = letter_probs, LETTER_LABELS, "letter_customcnn.pth"
    else:
        return {"char": "?", "confidence": 0.0, "model": "none", "probabilities": []}

    index = int(np.argmax(probs))
    return {
        "char": labels[index],
        "confidence": float(probs[index]),
        "model": model_used,
        "probabilities": [
            {"label": label, "probability": float(prob)} for label, prob in zip(labels, probs)
        ],
    }

def read_text_with_crnn(mask: np.ndarray) -> dict:
    """CTC greedy decoding of a whole handwritten line (blank index 0)."""
    if crnn_model is None:
        return {"text": "", "confidence": 0.0}

    ys, xs = np.nonzero(mask)
    if len(xs):
        mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = mask.shape
    new_width = max(28, int(round(width * 28 / max(1, height))))
    resized = cv2.resize(mask, (new_width, 28), interpolation=cv2.INTER_AREA).astype("float32") / 255.0

    batch = torch.tensor(resized.reshape(1, 1, 28, new_width), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = crnn_model(batch)
        probs = torch.softmax(logits, dim=2)[0].cpu().numpy()

    tokens = probs.argmax(axis=1)
    chars, confidences, previous = [], [], -1
    for step, token in enumerate(tokens):
        if token != 0 and token != previous and 1 <= token <= 26:
            chars.append(LETTER_LABELS[token - 1])
            confidences.append(float(probs[step, token]))
        previous = token
    return {
        "text": "".join(chars),
        "confidence": float(np.mean(confidences)) if confidences else 0.0,
    }

@app.post("/predict")
async def predict(req: InferenceRequest):
    try:
        encoded = req.imageDataUrl.split(",", 1)[-1]
        image_bytes = base64.b64decode(encoded)
        mask = to_ink_mask(image_bytes)

        task = req.task if req.task in ("digit", "letter", "text", "auto") else "auto"

        if task == "text":
            decoded = read_text_with_crnn(mask)
            local_result = decoded["text"]
            characters = []
            top_classes = []
            avg_conf = decoded["confidence"]
            models_used = {"best_crnn_model.pt"}
        else:
            boxes = segment_characters(mask)
            characters = []
            models_used = set()
            for x, y, w, h in boxes:
                prediction = classify_character(to_mnist_tensor(mask[y:y + h, x:x + w]), task)
                prediction["box"] = {"x": x, "y": y, "width": w, "height": h}
                characters.append(prediction)
                models_used.add(prediction["model"])

            local_result = "".join(c["char"] for c in characters)
            confidences = [c["confidence"] for c in characters]
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            top_classes = characters[0]["probabilities"] if characters else []

        final_result = call_gemini_fallback(image_bytes, local_result)
        if final_result != local_result:
            models_used.add("gemini-1.5-flash")

        return {
            "predictedText": final_result,
            "confidence": avg_conf,
            "top_classes": top_classes,
            "characters": [
                {"char": c["char"], "confidence": c["confidence"], "model": c["model"]}
                for c in characters
            ],
            "reasoning": (
                f"Task: {task}. Segmented {len(characters) if characters else 1} region(s). "
                f"Models used: {sorted(models_used)}"
                + ("" if final_result == local_result else f" (local read: '{local_result}')")
            ),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
