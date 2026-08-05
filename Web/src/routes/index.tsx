import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useServerFn } from "@tanstack/react-start";
import { Cpu, FileSpreadsheet, Play, Upload } from "lucide-react";
import { toast } from "sonner";
import { Header } from "@/components/Header";
import { DrawCanvas } from "@/components/DrawCanvas";
import { CameraCapture } from "@/components/CameraCapture";
import { PreprocessVisualizer } from "@/components/PreprocessVisualizer";
import { PredictionCard } from "@/components/PredictionCard";
import { HistoryTable } from "@/components/HistoryTable";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  MODELS,
  modelByKey,
  preprocessImage,
  type InputType,
  type ModelKey,
  type PreprocessResult,
} from "@/lib/hnrs";
import { runInference } from "@/lib/inference.functions";
import type { InferenceResult } from "@/lib/inference-shared";
import { historyQueryKey, savePrediction } from "@/services/historyService";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "HNRS Dashboard | Handwritten Recognition System" },
      {
        name: "description",
        content:
          "Recognise handwritten digits (0-9) and uppercase letters (A-Z) in real time from canvas, camera, upload or CSV with CNN and CRNN model inference.",
      },
      { property: "og:title", content: "HNRS Dashboard | Handwritten Recognition System" },
      {
        property: "og:description",
        content:
          "Recognise handwritten digits (0-9) and uppercase letters (A-Z) in real time from canvas, camera, upload or CSV with CNN and CRNN model inference.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Dashboard,
});

function csvToDataUrl(text: string): string | null {
  const firstRow = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.split(",").map((cell) => Number(cell.trim())))
    .find((values) => values.filter((v) => Number.isFinite(v)).length >= 784);
  if (!firstRow) return null;
  const numbers = firstRow.filter((v) => Number.isFinite(v));
  const pixels = numbers.length > 784 ? numbers.slice(numbers.length - 784) : numbers;
  const canvas = document.createElement("canvas");
  canvas.width = 28;
  canvas.height = 28;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const image = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const raw = pixels[i] ?? 0;
    const value = raw <= 1 ? Math.round(raw * 255) : Math.round(raw);
    const p = i * 4;
    image.data[p] = value;
    image.data[p + 1] = value;
    image.data[p + 2] = value;
    image.data[p + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  const scaled = document.createElement("canvas");
  scaled.width = 224;
  scaled.height = 224;
  const sctx = scaled.getContext("2d");
  if (!sctx) return null;
  sctx.imageSmoothingEnabled = false;
  sctx.drawImage(canvas, 0, 0, 224, 224);
  return scaled.toDataURL("image/png");
}

function Dashboard() {
  const queryClient = useQueryClient();
  const infer = useServerFn(runInference);
  const [modelKey, setModelKey] = useState<ModelKey>("digit_cnn_model.keras");
  const [inputType, setInputType] = useState<InputType>("Interactive Canvas");
  const [source, setSource] = useState<string | null>(null);
  const [preprocessed, setPreprocessed] = useState<PreprocessResult | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);

  const model = modelByKey(modelKey);

  const acceptImage = async (dataUrl: string | null, type: InputType) => {
    setInputType(type);
    setResult(null);
    setSource(dataUrl);
    if (!dataUrl) {
      setPreprocessed(null);
      return;
    }
    try {
      setPreprocessed(await preprocessImage(dataUrl));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Preprocessing failed");
    }
  };

  const onFile = (file: File | undefined, type: InputType) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? "");
      if (type === "CSV Test File") {
        const dataUrl = csvToDataUrl(text);
        if (!dataUrl) {
          toast.error("Could not find a 784-pixel row in that CSV file.");
          return;
        }
        void acceptImage(dataUrl, type);
      } else {
        void acceptImage(text, type);
      }
    };
    if (type === "CSV Test File") reader.readAsText(file);
    else reader.readAsDataURL(file);
  };

  const predict = useMutation({
    mutationFn: async () => {
      if (!preprocessed) throw new Error("Provide an input image first.");
      const inference = await infer({
        data: {
          imageDataUrl: preprocessed.binary,
          task: model.task,
          model: model.key,
        },
      });
      await savePrediction({
        input_type: inputType,
        model_used: model.key,
        predicted_text: inference.predictedText,
        confidence_score: inference.confidence,
        execution_time_ms: inference.latencyMs,
        image_data_url: preprocessed.resized,
      });
      return inference;
    },
    onSuccess: (inference) => {
      setResult(inference);
      toast.success(`Recognised "${inference.predictedText}"`);
      void queryClient.invalidateQueries({ queryKey: historyQueryKey });
    },
    onError: (error: Error) => toast.error(error.message),
  });

  return (
    <main className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6 sm:px-6 lg:px-8">
      <Header />

      <section className="panel p-5">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-wider text-primary">
              Control panel · multi-model inference
            </p>
            <h2 className="mt-1 text-lg font-semibold">Model selection</h2>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Select value={modelKey} onValueChange={(value) => setModelKey(value as ModelKey)}>
              <SelectTrigger className="w-72">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {MODELS.map((item) => (
                  <SelectItem key={item.key} value={item.key}>
                    {item.name} · {item.classes}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              onClick={() => predict.mutate()}
              disabled={!preprocessed || predict.isPending}
              className="min-w-40"
            >
              <Play className="size-4" />
              {predict.isPending ? "Running…" : "Run inference"}
            </Button>
          </div>
        </div>
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <div className="rounded-lg border border-border bg-secondary/40 p-3">
            <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Framework</p>
            <p className="mt-1 flex items-center gap-2 text-sm">
              <Cpu className="size-4 text-primary" />
              {model.framework}
            </p>
          </div>
          <div className="rounded-lg border border-border bg-secondary/40 p-3">
            <p className="text-[11px] uppercase tracking-wider text-muted-foreground">
              Architecture
            </p>
            <p className="mt-1 text-sm">{model.architecture}</p>
          </div>
          <div className="rounded-lg border border-border bg-secondary/40 p-3">
            <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Weights</p>
            <p className="mt-1 font-mono text-xs">../Models/{model.key}</p>
          </div>
        </div>
      </section>

      <HistoryTable compact />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <section className="panel p-5">
          <p className="font-mono text-[11px] uppercase tracking-wider text-primary">
            Input sources
          </p>
          <h2 className="mt-1 text-lg font-semibold">Capture handwriting</h2>
          <Tabs defaultValue="canvas" className="mt-4">
            <TabsList className="w-full">
              <TabsTrigger value="canvas">Canvas</TabsTrigger>
              <TabsTrigger value="camera">Camera</TabsTrigger>
              <TabsTrigger value="upload">Upload</TabsTrigger>
              <TabsTrigger value="csv">CSV</TabsTrigger>
            </TabsList>
            <TabsContent value="canvas" className="mt-4">
              <DrawCanvas onChange={(dataUrl) => void acceptImage(dataUrl, "Interactive Canvas")} />
            </TabsContent>
            <TabsContent value="camera" className="mt-4">
              <CameraCapture
                onCapture={(dataUrl) => void acceptImage(dataUrl, "Camera Capture")}
              />
            </TabsContent>
            <TabsContent value="upload" className="mt-4">
              <label className="grid-canvas flex h-56 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border text-center">
                <Upload className="size-6 text-primary" />
                <span className="text-sm text-muted-foreground">
                  Click to upload a handwriting image (PNG / JPG)
                </span>
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(event) => onFile(event.target.files?.[0], "File Upload")}
                />
              </label>
            </TabsContent>
            <TabsContent value="csv" className="mt-4">
              <label className="grid-canvas flex h-56 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border text-center">
                <FileSpreadsheet className="size-6 text-accent" />
                <span className="text-sm text-muted-foreground">
                  Upload a test CSV row (784 pixel values from the Datasets folder)
                </span>
                <input
                  type="file"
                  accept=".csv,text/csv"
                  className="hidden"
                  onChange={(event) => onFile(event.target.files?.[0], "CSV Test File")}
                />
              </label>
            </TabsContent>
          </Tabs>
          {source && (
            <p className="mt-3 font-mono text-[11px] text-muted-foreground">
              active input · {inputType}
            </p>
          )}
        </section>

        <PredictionCard result={result} model={model} pending={predict.isPending} />
      </div>

      <section className="panel p-5">
        <p className="font-mono text-[11px] uppercase tracking-wider text-primary">
          Task 1 &amp; 2 · preprocessing and segmentation
        </p>
        <h2 className="mt-1 mb-4 text-lg font-semibold">Image pipeline visualizer</h2>
        <PreprocessVisualizer result={preprocessed} />
      </section>
    </main>
  );
}
