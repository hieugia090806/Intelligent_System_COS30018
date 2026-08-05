import { z } from "zod";

export const InferenceInput = z.object({
  imageDataUrl: z.string().min(32),
  task: z.enum(["digit", "letter", "text", "auto"]),
  model: z.string().min(1),
});

export type Task = z.infer<typeof InferenceInput>["task"];

export interface ClassProbability {
  label: string;
  probability: number;
}

export interface CharacterPrediction {
  char: string;
  confidence: number;
  model: string;
}

export interface InferenceResult {
  predictedText: string;
  confidence: number;
  probabilities: ClassProbability[];
  characters: CharacterPrediction[];
  latencyMs: number;
  reasoning: string;
}

const DIGITS = Array.from({ length: 10 }, (_, i) => String(i));
const LETTERS = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));

export function labelsFor(task: Task) {
  if (task === "digit") return DIGITS;
  if (task === "auto") return [...DIGITS, ...LETTERS];
  return LETTERS;
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