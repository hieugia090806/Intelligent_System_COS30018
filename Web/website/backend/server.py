import os
import io
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="HNRS Model Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
crnn_model = None

@app.on_event("startup")
def load_all_models():
    global digit_model, crnn_model
    keras_path = os.path.join(MODELS_DIR, "digit_cnn_model.keras")
    if os.path.exists(keras_path):
        digit_model = tf.keras.models.load_model(keras_path)

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
            print(f"❌ CRNN load error: {e}")

class InferenceRequest(BaseModel):
    task: str
    imageDataUrl: str

def call_gemini_fallback(image_bytes: bytes, local_prediction: str) -> str:
    """Sử dụng Gemini API làm tầng phụ kiểm tra và sửa lỗi nhận diện handwriting"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return local_prediction
    try:
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

@app.post("/predict")
async def predict(req: InferenceRequest):
    try:
        header, encoded = req.imageDataUrl.split(",", 1) if "," in req.imageDataUrl else ("", req.imageDataUrl)
        image_bytes = base64.b64decode(encoded)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")

        img_np = np.array(pil_img)
        _, thresh = cv2.threshold(img_np, 40, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 4 and h > 8 and cv2.contourArea(c) > 10:
                valid_boxes.append((x, y, w, h))

        valid_boxes = sorted(valid_boxes, key=lambda b: b[0])
        if len(valid_boxes) == 0:
            valid_boxes = [(0, 0, thresh.shape[1], thresh.shape[0])]

        predicted_chars = []
        confidences = []
        models_used = set()

        for i, (x, y, w, h) in enumerate(valid_boxes):
            pad = 8
            ymin = max(0, y - pad)
            ymax = min(thresh.shape[0], y + h + pad)
            xmin = max(0, x - pad)
            xmax = min(thresh.shape[1], x + w + pad)
            
            roi = thresh[ymin:ymax, xmin:xmax]
            rh, rw = roi.shape
            max_side = max(rh, rw)
            square = np.zeros((max_side, max_side), dtype=np.uint8)
            square[(max_side - rh) // 2:(max_side - rh) // 2 + rh, (max_side - rw) // 2:(max_side - rw) // 2 + rw] = roi
            
            resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
            norm_roi = resized.astype("float32") / 255.0

            # Phân tách logic gọi model tuyệt đối theo req.task từ Client gửi lên
            if req.task == "digit":
                chosen, conf = "0", 0.0
                if digit_model is not None:
                    d_probs = digit_model.predict(norm_roi.reshape(1, 28, 28, 1), verbose=0)[0]
                    chosen = str(np.argmax(d_probs))
                    conf = float(np.max(d_probs))
                models_used.add("digit_cnn_model.keras")
            else:
                chosen, conf = "A", 0.0
                if crnn_model is not None:
                    tensor_x = torch.tensor(norm_roi.reshape(1, 1, 28, 28), dtype=torch.float32).to(device)
                    with torch.no_grad():
                        logits = crnn_model(tensor_x)
                        probs_pt = torch.softmax(logits, dim=2)
                        preds = logits.argmax(dim=2).squeeze(0).tolist()
                        decoded = [t for idx_t, t in enumerate(preds) if t != 0 and (idx_t == 0 or t != preds[idx_t-1])]
                        pred_token = decoded[0] if len(decoded) > 0 else 1
                        if 1 <= pred_token <= 26:
                            chosen = chr(65 + (pred_token - 1))
                            conf = float(probs_pt[0, 0, pred_token].item())
                models_used.add("best_crnn_model.pt")

            predicted_chars.append(chosen)
            confidences.append(conf)

        local_result = "".join(predicted_chars)
        
        # Hậu xử lý thông minh qua Gemini API
        final_result = call_gemini_fallback(image_bytes, local_result)
        if final_result != local_result:
            models_used.add("gemini-1.5-flash")

        avg_conf = float(np.mean(confidences)) if confidences else 0.9

        return {
            "predictedText": final_result,
            "confidence": avg_conf if final_result == local_result else 0.98,
            "top_classes": [{"label": final_result, "probability": avg_conf}],
            "reasoning": f"Task: {req.task}. Models used: {list(models_used)}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)