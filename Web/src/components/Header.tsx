import { Link } from "@tanstack/react-router";
import { BrainCircuit } from "lucide-react";
import { HEADER_DESCRIPTION } from "@/lib/hnrs";

const NAV = [
  { to: "/", label: "Dashboard" },
  { to: "/evaluation", label: "Model Evaluation" },
  { to: "/history", label: "History" },
] as const;

export function Header() {
  return (
    <header className="hero-surface relative overflow-hidden px-6 py-8 sm:px-10">
      <div className="flex flex-wrap items-center justify-between gap-6">
        <div className="flex items-center gap-3">
          <span className="glow-ring flex size-11 items-center justify-center rounded-xl bg-secondary">
            <BrainCircuit className="size-6 text-primary" />
          </span>
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-muted-foreground">
              COS30018 · Theme B · Group 1
            </p>
            <h1 className="text-2xl font-semibold sm:text-3xl">
              Handwritten <span className="text-gradient">Recognition System</span>
            </h1>
          </div>
        </div>
        <nav className="flex flex-wrap gap-2">
          {NAV.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              activeOptions={{ exact: item.to === "/" }}
              className="rounded-lg border border-border px-4 py-2 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground data-[status=active]:border-primary/60 data-[status=active]:bg-primary/10 data-[status=active]:text-primary"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
      <p className="mt-6 max-w-4xl text-sm leading-relaxed text-muted-foreground sm:text-base">
        {HEADER_DESCRIPTION}
      </p>
    </header>
  );
}