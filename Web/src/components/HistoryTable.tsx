import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Eye, Search, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { MODELS, formatConfidence } from "@/lib/hnrs";
import {
  deletePrediction,
  historyQueryKey,
  listPredictions,
  type PredictionRecord,
} from "@/services/historyService";

const PAGE_SIZE = 6;

export function HistoryTable({ compact = false }: { compact?: boolean }) {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [modelFilter, setModelFilter] = useState("all");
  const [page, setPage] = useState(0);
  const [detail, setDetail] = useState<PredictionRecord | null>(null);

  const { data = [], isLoading } = useQuery({
    queryKey: historyQueryKey,
    queryFn: listPredictions,
  });

  const remove = useMutation({
    mutationFn: deletePrediction,
    onSuccess: () => {
      toast.success("Prediction deleted");
      void queryClient.invalidateQueries({ queryKey: historyQueryKey });
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const filtered = useMemo(() => {
    const term = search.trim().toUpperCase();
    return data.filter((row) => {
      const matchesModel = modelFilter === "all" || row.model_used === modelFilter;
      const matchesTerm = !term || row.predicted_text.toUpperCase().includes(term);
      return matchesModel && matchesTerm;
    });
  }, [data, search, modelFilter]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const current = Math.min(page, pageCount - 1);
  const rows = filtered.slice(current * PAGE_SIZE, current * PAGE_SIZE + PAGE_SIZE);

  return (
    <section className="panel p-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-wider text-primary">
            Persistent store · prediction_history
          </p>
          <h2 className="mt-1 text-lg font-semibold">Prediction History</h2>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setPage(0);
              }}
              placeholder="Search character…"
              className="w-44 pl-9"
            />
          </div>
          <Select
            value={modelFilter}
            onValueChange={(value) => {
              setModelFilter(value);
              setPage(0);
            }}
          >
            <SelectTrigger className="w-52">
              <SelectValue placeholder="Filter by model" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All models</SelectItem>
              {MODELS.map((model) => (
                <SelectItem key={model.key} value={model.key}>
                  {model.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="mt-4 overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-14">ID</TableHead>
              <TableHead>Timestamp</TableHead>
              <TableHead>Input</TableHead>
              <TableHead>Model used</TableHead>
              <TableHead>Predicted</TableHead>
              <TableHead>Confidence</TableHead>
              {!compact && <TableHead>Latency</TableHead>}
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell colSpan={8} className="py-8 text-center text-sm text-muted-foreground">
                  Loading history…
                </TableCell>
              </TableRow>
            ) : rows.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="py-8 text-center text-sm text-muted-foreground">
                  No predictions recorded yet.
                </TableCell>
              </TableRow>
            ) : (
              rows.map((row) => (
                <TableRow key={row.id}>
                  <TableCell className="font-mono text-xs">{row.id}</TableCell>
                  <TableCell className="whitespace-nowrap text-xs text-muted-foreground">
                    {new Date(row.created_at).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      {row.image_data_url ? (
                        <img
                          src={row.image_data_url}
                          alt={`Input for prediction ${row.id}`}
                          className="size-10 rounded border border-border"
                          style={{ imageRendering: "pixelated" }}
                        />
                      ) : (
                        <div className="size-10 rounded border border-dashed border-border" />
                      )}
                      <span className="text-xs text-muted-foreground">{row.input_type}</span>
                    </div>
                  </TableCell>
                  <TableCell className="font-mono text-xs">{row.model_used}</TableCell>
                  <TableCell className="font-mono text-base text-primary">
                    {row.predicted_text}
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant="outline"
                      className={
                        row.confidence_score >= 0.8
                          ? "border-success/50 text-success"
                          : "border-warning/50 text-warning"
                      }
                    >
                      {formatConfidence(row.confidence_score)}
                    </Badge>
                  </TableCell>
                  {!compact && (
                    <TableCell className="font-mono text-xs">{row.execution_time_ms} ms</TableCell>
                  )}
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-1">
                      <Button size="icon" variant="ghost" onClick={() => setDetail(row)}>
                        <Eye className="size-4" />
                        <span className="sr-only">View details</span>
                      </Button>
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={() => remove.mutate(row.id)}
                        disabled={remove.isPending}
                      >
                        <Trash2 className="size-4 text-destructive" />
                        <span className="sr-only">Delete</span>
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground">
        <span className="font-mono">
          {filtered.length} record(s) · page {current + 1} / {pageCount}
        </span>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => setPage(Math.max(0, current - 1))}
            disabled={current === 0}
          >
            <ChevronLeft className="size-4" />
            Prev
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setPage(Math.min(pageCount - 1, current + 1))}
            disabled={current >= pageCount - 1}
          >
            Next
            <ChevronRight className="size-4" />
          </Button>
        </div>
      </div>

      <Dialog open={Boolean(detail)} onOpenChange={(open) => !open && setDetail(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Prediction #{detail?.id}</DialogTitle>
            <DialogDescription>
              {detail ? new Date(detail.created_at).toLocaleString() : ""}
            </DialogDescription>
          </DialogHeader>
          {detail && (
            <div className="space-y-4">
              {detail.image_data_url && (
                <img
                  src={detail.image_data_url}
                  alt={`Input for prediction ${detail.id}`}
                  className="grid-canvas mx-auto size-32 rounded-lg border border-border"
                  style={{ imageRendering: "pixelated" }}
                />
              )}
              <dl className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <dt className="text-xs uppercase text-muted-foreground">Predicted output</dt>
                  <dd className="font-mono text-lg text-primary">{detail.predicted_text}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase text-muted-foreground">Confidence</dt>
                  <dd className="font-mono">{formatConfidence(detail.confidence_score)}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase text-muted-foreground">Model</dt>
                  <dd className="font-mono text-xs">{detail.model_used}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase text-muted-foreground">Latency</dt>
                  <dd className="font-mono">{detail.execution_time_ms} ms</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase text-muted-foreground">Input type</dt>
                  <dd>{detail.input_type}</dd>
                </div>
              </dl>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </section>
  );
}