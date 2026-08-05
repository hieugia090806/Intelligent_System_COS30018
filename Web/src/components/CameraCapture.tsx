import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, CameraOff, RefreshCcw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface CameraCaptureProps {
  onCapture: (dataUrl: string) => void;
}

export function CameraCapture({ onCapture }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [facing, setFacing] = useState<"environment" | "user">("environment");

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setActive(false);
  }, []);

  useEffect(() => stop, [stop]);

  const start = useCallback(
    async (mode: "environment" | "user") => {
      setError(null);
      try {
        stop();
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: mode, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setActive(true);
      } catch {
        setError("Camera access was blocked or no camera is available on this device.");
        setActive(false);
      }
    },
    [stop],
  );

  const capture = () => {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    onCapture(canvas.toDataURL("image/png"));
  };

  const flip = () => {
    const next = facing === "environment" ? "user" : "environment";
    setFacing(next);
    void start(next);
  };

  return (
    <div className="space-y-4">
      <div className="grid-canvas relative flex h-56 items-center justify-center overflow-hidden rounded-xl border border-border">
        <video
          ref={videoRef}
          playsInline
          muted
          className={`size-full object-cover ${active ? "" : "hidden"}`}
        />
        {!active && (
          <p className="px-6 text-center text-sm text-muted-foreground">
            {error ?? "Start the camera to capture handwriting from paper or a whiteboard."}
          </p>
        )}
        {active && (
          <div className="pointer-events-none absolute inset-8 rounded-lg border-2 border-dashed border-primary/60" />
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        {active ? (
          <>
            <Button size="sm" onClick={capture}>
              <Camera className="size-4" />
              Capture frame
            </Button>
            <Button size="sm" variant="outline" onClick={flip}>
              <RefreshCcw className="size-4" />
              Flip camera
            </Button>
            <Button size="sm" variant="ghost" onClick={stop}>
              <CameraOff className="size-4" />
              Stop
            </Button>
          </>
        ) : (
          <Button size="sm" onClick={() => void start(facing)}>
            <Camera className="size-4" />
            Start camera
          </Button>
        )}
      </div>
    </div>
  );
}