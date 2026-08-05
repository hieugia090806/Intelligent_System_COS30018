import { useEffect, useRef, useState } from "react";
import { Eraser, PenLine } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";

interface DrawCanvasProps {
  onChange: (dataUrl: string | null) => void;
}

export function DrawCanvas({ onChange }: DrawCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawing = useRef(false);
  const [strokeWidth, setStrokeWidth] = useState(18);
  const [hasInk, setHasInk] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#0b0f18";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const pos = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const start = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    drawing.current = true;
    const { x, y } = pos(event);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#f8fafc";
    ctx.lineWidth = strokeWidth;
    ctx.lineTo(x + 0.1, y + 0.1);
    ctx.stroke();
    setHasInk(true);
  };

  const move = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current) return;
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    const { x, y } = pos(event);
    ctx.lineWidth = strokeWidth;
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const end = () => {
    if (!drawing.current) return;
    drawing.current = false;
    const canvas = canvasRef.current;
    if (canvas) onChange(canvas.toDataURL("image/png"));
  };

  const clear = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.fillStyle = "#0b0f18";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setHasInk(false);
    onChange(null);
  };

  return (
    <div className="space-y-4">
      <canvas
        ref={canvasRef}
        width={560}
        height={240}
        onPointerDown={start}
        onPointerMove={move}
        onPointerUp={end}
        onPointerLeave={end}
        className="grid-canvas h-56 w-full cursor-crosshair touch-none rounded-xl border border-border"
        aria-label="Handwriting input canvas"
      />
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex min-w-44 flex-1 items-center gap-3">
          <PenLine className="size-4 text-muted-foreground" />
          <Slider
            value={[strokeWidth]}
            min={6}
            max={34}
            step={2}
            onValueChange={(value) => setStrokeWidth(value[0] ?? 18)}
          />
          <span className="font-mono text-xs text-muted-foreground">{strokeWidth}px</span>
        </div>
        <Button variant="outline" size="sm" onClick={clear} disabled={!hasInk}>
          <Eraser className="size-4" />
          Clear
        </Button>
      </div>
    </div>
  );
}