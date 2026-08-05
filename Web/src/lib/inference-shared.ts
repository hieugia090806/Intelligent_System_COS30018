import { z } from "zod";

export const InferenceInput = z.object({
  imageDataUrl: z.string().min(32),
  task: z.enum(["digit", "letter", "text"]),
  model: z.string().min(1),
});

export interface ClassProbability {
  label: string;
  probability: number;
}

export interface InferenceResult {
  predictedText: string;
  confidence: number;
  probabilities: ClassProbability[];
  latencyMs: number;
  reasoning: string;
}

export function labelsFor(task: "digit" | "letter" | "text") {
  return task === "digit"
    ? Array.from({ length: 10 }, (_, i) => String(i))
    : Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));
}

export function softmaxFallback(
  labels: string[],
  top: string,
  confidence: number,
): ClassProbability[] {
  const rest = Math.max(0, 1 - confidence) / Math.max(1, labels.length - 1);
  return labels.map((label) => ({
    label,
    probability: label === top ? confidence : rest,
  }));
}