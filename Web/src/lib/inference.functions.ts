import { createServerFn } from "@tanstack/react-start";
import {
  InferenceInput,
  labelsFor,
  softmaxFallback,
  type CharacterPrediction,
  type ClassProbability,
  type InferenceResult,
} from "./inference-shared";

export const runInference = createServerFn({ method: "POST" })
  .validator(InferenceInput)
  .handler(async ({ data }): Promise<InferenceResult> => {
    const started = Date.now();
    const labels = labelsFor(data.task);

    try {
      // 1. Call the local Python Backend Server loading the actual model weights
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          task: data.task, // 'digit', 'letter', or 'engtext'
          imageDataUrl: data.imageDataUrl,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("[HNRS API Error]:", response.status, errorText);
        throw new Error(`Model server error: ${response.statusText}`);
      }

      const result = (await response.json()) as {
        predictedText?: string;
        confidence?: number;
        top_classes?: { label: string; probability: number }[];
        characters?: { char?: string; confidence?: number; model?: string }[];
        reasoning?: string;
      };

      const latencyMs = Date.now() - started;

      const allowed = new Set(labels);
      const predictedText = String(result.predictedText ?? "")
        .toUpperCase()
        .split("")
        .filter((ch) => allowed.has(ch))
        .join("");

      const confidence = Math.min(
        1,
        Math.max(0, typeof result.confidence === "number" ? result.confidence : 0)
      );

      // Build probabilities array for UI visualization
      const reported = (result.top_classes ?? [])
        .map((entry) => ({
          label: String(entry.label ?? "").toUpperCase(),
          probability: Math.min(1, Math.max(0, Number(entry.probability ?? 0))),
        }))
        .filter((entry) => allowed.has(entry.label));

      const probabilities: ClassProbability[] = reported.length
        ? reported
        : softmaxFallback(labels, predictedText.charAt(0) || labels[0]!, confidence);

      const characters: CharacterPrediction[] = (result.characters ?? [])
        .map((entry) => ({
          char: String(entry.char ?? "").toUpperCase(),
          confidence: Math.min(1, Math.max(0, Number(entry.confidence ?? 0))),
          model: String(entry.model ?? ""),
        }))
        .filter((entry) => allowed.has(entry.char));

      return {
        predictedText: predictedText || "?",
        confidence: predictedText ? confidence : 0,
        probabilities,
        characters,
        latencyMs,
        reasoning: result.reasoning || "Executed using local trained model weights.",
      };
    } catch (error: any) {
      console.error("[HNRS Inference Connection Error]:", error);
      throw new Error(
        error.message || "Failed to connect to local Python Model Server at http://127.0.0.1:8000."
      );
    }
  });