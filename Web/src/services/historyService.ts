import { supabase } from "@/integrations/supabase/client";
import type { InputType, ModelKey } from "@/lib/hnrs";

export interface PredictionRecord {
  id: number;
  created_at: string;
  input_type: string;
  model_used: string;
  predicted_text: string;
  confidence_score: number;
  execution_time_ms: number;
  image_data_url: string | null;
}

export interface NewPrediction {
  input_type: InputType;
  model_used: ModelKey;
  predicted_text: string;
  confidence_score: number;
  execution_time_ms: number;
  image_data_url: string | null;
}

export async function savePrediction(record: NewPrediction): Promise<PredictionRecord> {
  const { data, error } = await supabase
    .from("prediction_history")
    .insert(record)
    .select()
    .single();
  if (error) throw new Error(error.message);
  return data as PredictionRecord;
}

export async function listPredictions(): Promise<PredictionRecord[]> {
  const { data, error } = await supabase
    .from("prediction_history")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(500);
  if (error) throw new Error(error.message);
  return (data ?? []) as PredictionRecord[];
}

export async function deletePrediction(id: number): Promise<void> {
  const { error } = await supabase.from("prediction_history").delete().eq("id", id);
  if (error) throw new Error(error.message);
}

export const historyQueryKey = ["prediction_history"] as const;