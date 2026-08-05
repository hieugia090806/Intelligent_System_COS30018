CREATE TABLE public.prediction_history (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  input_type TEXT NOT NULL,
  model_used TEXT NOT NULL,
  predicted_text TEXT NOT NULL,
  confidence_score DOUBLE PRECISION NOT NULL DEFAULT 0,
  execution_time_ms INTEGER NOT NULL DEFAULT 0,
  image_data_url TEXT
);

GRANT SELECT, INSERT, DELETE ON public.prediction_history TO anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.prediction_history TO authenticated;
GRANT ALL ON public.prediction_history TO service_role;
GRANT USAGE, SELECT ON SEQUENCE public.prediction_history_id_seq TO anon, authenticated;
GRANT ALL ON SEQUENCE public.prediction_history_id_seq TO service_role;

ALTER TABLE public.prediction_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public can read prediction history" ON public.prediction_history FOR SELECT USING (true);
CREATE POLICY "Public can insert prediction history" ON public.prediction_history FOR INSERT WITH CHECK (true);
CREATE POLICY "Public can delete prediction history" ON public.prediction_history FOR DELETE USING (true);

CREATE INDEX prediction_history_created_at_idx ON public.prediction_history (created_at DESC);