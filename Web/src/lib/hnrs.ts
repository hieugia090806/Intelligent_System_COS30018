export const HEADER_DESCRIPTION =
  "This Handwritten Recognition System is an advanced AI-powered platform utilizing custom Convolutional Neural Networks (CNN) to accurately classify, read, and interpret isolated handwritten digits (0-9) and uppercase English characters (A-Z) in real-time from various input sources.";

export type ModelKey =
  | "auto"
  | "digit_cnn_model.keras"
  | "letter_customcnn.pth"
  | "best_crnn_model.pt";

export type InputType =
  | "Interactive Canvas"
  | "File Upload"
  | "Camera Capture"
  | "CSV Test File";

export interface ModelInfo {
  key: ModelKey;
  name: string;
  framework: "Keras / TensorFlow" | "PyTorch" | "Keras + PyTorch";
  architecture: string;
  classes: string;
  task: "digit" | "letter" | "text" | "auto";
  /** Weight files loaded on the backend for this selection. */
  weights: string;
  accuracy: number;
  params: string;
  description: string;
}

export const MODELS: ModelInfo[] = [
  {
    key: "auto",
    name: "Auto Digit + Letter",
    framework: "Keras + PyTorch",
    architecture: "Digit CNN + Letter CNN, per-character confidence routing",
    classes: "0-9, A-Z",
    task: "auto",
    weights: "digit_cnn_model.keras + letter_customcnn.pth",
    accuracy: 0.88,
    params: "3.6M",
    description:
      "Segments the input into characters and sends each one to both classifiers, keeping the more confident answer. Use this for mixed digit/letter handwriting.",
  },
  {
    key: "digit_cnn_model.keras",
    name: "Digit CNN",
    framework: "Keras / TensorFlow",
    architecture: "3× Conv2D + BatchNorm + Dense(10)",
    classes: "0-9",
    task: "digit",
    weights: "digit_cnn_model.keras",
    accuracy: 0.9932,
    params: "1.2M",
    description: "Isolated handwritten digit classifier trained on the MNIST-style numbers dataset.",
  },
  {
    key: "letter_customcnn.pth",
    name: "Letter Custom CNN",
    framework: "PyTorch",
    architecture: "Custom CNN 4 blocks + Dropout + Dense(26)",
    classes: "A-Z",
    task: "letter",
    weights: "letter_customcnn.pth",
    accuracy: 0.9691,
    params: "2.4M",
    description: "Single uppercase English character classifier trained on the EMNIST-letters split.",
  },
  {
    key: "best_crnn_model.pt",
    name: "Text CRNN",
    framework: "PyTorch",
    architecture: "CNN backbone + BiLSTM + CTC decoder",
    classes: "A-Z sequences",
    task: "text",
    weights: "best_crnn_model.pt",
    accuracy: 0.9214,
    params: "8.7M",
    description: "Sequence recognition model that reads whole handwritten words without pre-segmentation.",
  },
];

export function modelByKey(key: ModelKey): ModelInfo {
  return MODELS.find((m) => m.key === key) ?? MODELS[0]!;
}

export const DIGIT_LABELS = Array.from({ length: 10 }, (_, i) => String(i));
export const LETTER_LABELS = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));

export function labelsForTask(task: ModelInfo["task"]): string[] {
  if (task === "digit") return DIGIT_LABELS;
  if (task === "auto") return [...DIGIT_LABELS, ...LETTER_LABELS];
  return LETTER_LABELS;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PreprocessResult {
  original: string;
  grayscale: string;
  binary: string;
  resized: string;
  /** 28×28 normalized pixel tensor, row-major, values 0..1 */
  tensor: number[];
  threshold: number;
  boxes: BoundingBox[];
  /** 28×28 normalized crops per detected character */
  segments: string[];
  inkRatio: number;
}

const WORK_SIZE = 280;

function makeCanvas(width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable");
  return { canvas, ctx };
}

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not load the image"));
    img.src = src;
  });
}

function otsu(histogram: number[], total: number) {
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * histogram[i]!;
  let sumB = 0;
  let wB = 0;
  let best = 0;
  let threshold = 127;
  for (let t = 0; t < 256; t++) {
    wB += histogram[t]!;
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * histogram[t]!;
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) {
      best = between;
      threshold = t;
    }
  }
  return threshold;
}

/** Turn a mask (1 = ink) into a centered 28×28 normalized tensor + preview data URL. */
function maskToTensor(
  mask: Uint8Array,
  width: number,
  height: number,
  box: BoundingBox,
): { tensor: number[]; dataUrl: string } {
  const side = Math.max(box.width, box.height) || 1;
  const pad = Math.round(side * 0.2);
  const cropSide = side + pad * 2;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const startX = cx - cropSide / 2;
  const startY = cy - cropSide / 2;

  const { canvas, ctx } = makeCanvas(28, 28);
  const out = ctx.createImageData(28, 28);
  const tensor: number[] = new Array(28 * 28).fill(0);

  for (let ty = 0; ty < 28; ty++) {
    for (let tx = 0; tx < 28; tx++) {
      const sx0 = Math.floor(startX + (tx * cropSide) / 28);
      const sx1 = Math.max(sx0 + 1, Math.floor(startX + ((tx + 1) * cropSide) / 28));
      const sy0 = Math.floor(startY + (ty * cropSide) / 28);
      const sy1 = Math.max(sy0 + 1, Math.floor(startY + ((ty + 1) * cropSide) / 28));
      let acc = 0;
      let count = 0;
      for (let sy = sy0; sy < sy1; sy++) {
        for (let sx = sx0; sx < sx1; sx++) {
          count++;
          if (sx < 0 || sy < 0 || sx >= width || sy >= height) continue;
          acc += mask[sy * width + sx]!;
        }
      }
      const value = count ? acc / count : 0;
      const idx = ty * 28 + tx;
      tensor[idx] = Math.min(1, value);
      const p = idx * 4;
      const v = Math.round(value * 255);
      out.data[p] = v;
      out.data[p + 1] = v;
      out.data[p + 2] = v;
      out.data[p + 3] = 255;
    }
  }
  ctx.putImageData(out, 0, 0);
  return { tensor, dataUrl: canvas.toDataURL("image/png") };
}

function segmentMask(mask: Uint8Array, width: number, height: number): BoundingBox[] {
  const columns = new Array<number>(width).fill(0);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (mask[y * width + x]! > 0.35) columns[x]! += 1;
    }
  }
  const minGap = Math.max(3, Math.round(width * 0.015));
  const boxes: BoundingBox[] = [];
  let runStart = -1;
  let gap = 0;
  for (let x = 0; x <= width; x++) {
    const active = x < width && columns[x]! > 0;
    if (active) {
      if (runStart < 0) runStart = x;
      gap = 0;
    } else if (runStart >= 0) {
      gap += 1;
      if (gap >= minGap || x === width) {
        const endX = x - gap;
        if (endX - runStart >= 2) {
          let top = height;
          let bottom = -1;
          for (let y = 0; y < height; y++) {
            for (let px = runStart; px <= endX; px++) {
              if (mask[y * width + px]! > 0.35) {
                if (y < top) top = y;
                if (y > bottom) bottom = y;
              }
            }
          }
          if (bottom >= top) {
            boxes.push({
              x: runStart,
              y: top,
              width: endX - runStart + 1,
              height: bottom - top + 1,
            });
          }
        }
        runStart = -1;
        gap = 0;
      }
    }
  }
  return boxes;
}

/**
 * Task 1 + Task 2: full preprocessing pipeline and character segmentation.
 * Runs entirely in the browser on a canvas — call from an event handler, never during render.
 */
export async function preprocessImage(source: string): Promise<PreprocessResult> {
  const img = await loadImage(source);
  const scale = Math.min(1, WORK_SIZE / Math.max(img.width, img.height));
  const width = Math.max(1, Math.round(img.width * scale) || WORK_SIZE);
  const height = Math.max(1, Math.round(img.height * scale) || WORK_SIZE);

  const base = makeCanvas(width, height);
  base.ctx.fillStyle = "#000000";
  base.ctx.fillRect(0, 0, width, height);
  base.ctx.drawImage(img, 0, 0, width, height);
  const original = base.canvas.toDataURL("image/png");

  const pixels = base.ctx.getImageData(0, 0, width, height);
  const gray = new Uint8Array(width * height);
  const histogram = new Array<number>(256).fill(0);
  for (let i = 0; i < width * height; i++) {
    const p = i * 4;
    const value = Math.round(
      0.299 * pixels.data[p]! + 0.587 * pixels.data[p + 1]! + 0.114 * pixels.data[p + 2]!,
    );
    gray[i] = value;
    histogram[value]! += 1;
  }

  const grayCanvas = makeCanvas(width, height);
  const grayImage = grayCanvas.ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const p = i * 4;
    grayImage.data[p] = gray[i]!;
    grayImage.data[p + 1] = gray[i]!;
    grayImage.data[p + 2] = gray[i]!;
    grayImage.data[p + 3] = 255;
  }
  grayCanvas.ctx.putImageData(grayImage, 0, 0);
  const grayscale = grayCanvas.canvas.toDataURL("image/png");

  const threshold = otsu(histogram, width * height);
  let brightCount = 0;
  for (let i = 0; i < width * height; i++) if (gray[i]! > threshold) brightCount++;
  // Ink should end up as 1. If most pixels are bright, the ink is the dark side.
  const inkIsDark = brightCount > width * height * 0.5;

  const mask = new Uint8Array(width * height);
  let inkPixels = 0;
  for (let i = 0; i < width * height; i++) {
    const isInk = inkIsDark ? gray[i]! <= threshold : gray[i]! > threshold;
    mask[i] = isInk ? 1 : 0;
    if (isInk) inkPixels++;
  }

  const binCanvas = makeCanvas(width, height);
  const binImage = binCanvas.ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const p = i * 4;
    const v = mask[i] ? 255 : 0;
    binImage.data[p] = v;
    binImage.data[p + 1] = v;
    binImage.data[p + 2] = v;
    binImage.data[p + 3] = 255;
  }
  binCanvas.ctx.putImageData(binImage, 0, 0);
  const binary = binCanvas.canvas.toDataURL("image/png");

  const boxes = segmentMask(mask, width, height);
  const fullBox: BoundingBox = boxes.length
    ? boxes.reduce<BoundingBox>(
        (acc, b) => {
          const x = Math.min(acc.x, b.x);
          const y = Math.min(acc.y, b.y);
          const right = Math.max(acc.x + acc.width, b.x + b.width);
          const bottom = Math.max(acc.y + acc.height, b.y + b.height);
          return { x, y, width: right - x, height: bottom - y };
        },
        { x: boxes[0]!.x, y: boxes[0]!.y, width: boxes[0]!.width, height: boxes[0]!.height },
      )
    : { x: 0, y: 0, width, height };

  const full = maskToTensor(mask, width, height, fullBox);
  const segments = boxes
    .slice(0, 16)
    .map((box) => maskToTensor(mask, width, height, box).dataUrl);

  return {
    original,
    grayscale,
    binary,
    resized: full.dataUrl,
    tensor: full.tensor,
    threshold,
    boxes,
    segments,
    inkRatio: inkPixels / (width * height),
  };
}

export function formatConfidence(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}