import { createFileRoute } from "@tanstack/react-router";
import { Header } from "@/components/Header";
import { HistoryTable } from "@/components/HistoryTable";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "Prediction History | HNRS" },
      {
        name: "description",
        content:
          "Browse, search, filter and delete every handwritten recognition run with thumbnails, model used, confidence and latency.",
      },
      { property: "og:title", content: "Prediction History | HNRS" },
      {
        property: "og:description",
        content:
          "Searchable, paginated log of handwritten recognition predictions with confidence and latency metrics.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: HistoryPage,
});

function HistoryPage() {
  return (
    <main className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6 sm:px-6 lg:px-8">
      <Header />
      <HistoryTable />
    </main>
  );
}