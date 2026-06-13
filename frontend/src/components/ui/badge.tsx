import * as React from "react";

import { cn } from "@/lib/utils";

export function Badge({
  className,
  tone = "muted",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & {
  tone?: "muted" | "warn" | "emerald";
}) {
  const tones = {
    muted: "border-border text-muted-foreground",
    warn: "border-amber-500/40 text-amber-400 bg-amber-500/5",
    emerald: "border-primary/40 text-primary bg-primary/5",
  } as const;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-medium",
        tones[tone],
        className,
      )}
      {...props}
    />
  );
}
