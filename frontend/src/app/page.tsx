"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";

import { api, authUrls, type Me } from "@/lib/api";
import { Button } from "@/components/ui/button";

function GoogleMark() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" aria-hidden>
      <path
        fill="#4285F4"
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1z"
      />
      <path
        fill="#34A853"
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0 0 12 23z"
      />
      <path
        fill="#FBBC05"
        d="M5.84 14.1a6.6 6.6 0 0 1 0-4.2V7.06H2.18a11 11 0 0 0 0 9.88l3.66-2.84z"
      />
      <path
        fill="#EA4335"
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1A11 11 0 0 0 2.18 7.06l3.66 2.84C6.71 7.3 9.14 5.38 12 5.38z"
      />
    </svg>
  );
}

export default function SignInPage() {
  const router = useRouter();
  const [me, setMe] = useState<Me | null>(null);
  const [devEmail, setDevEmail] = useState("");

  useEffect(() => {
    api
      .me()
      .then((m) => {
        if (m.signed_in) router.replace("/dashboard");
        else setMe(m);
      })
      .catch(() => setMe({ signed_in: false, email: null, google_enabled: false, dev_login: false }));
  }, [router]);

  return (
    <main className="flex min-h-dvh items-center justify-center px-6">
      <div className="w-full max-w-sm animate-rise">
        {/* mark */}
        <div className="mb-8 flex flex-col items-center text-center">
          <div className="relative mb-5 size-20 overflow-hidden rounded-2xl border border-border">
            <Image src="/vfte-logo.png" alt="VFTE" fill sizes="80px" priority className="object-cover" />
          </div>
          <h1 className="text-[1.65rem] font-semibold tracking-tight">
            Your voice, your keys.
          </h1>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
            See whether your voiceprint is stored, how it&apos;s used, and control
            it — sealed inside the enclave where even the operator can&apos;t read it.
          </p>
        </div>

        {/* sign-in card */}
        <div className="rounded-2xl border border-border bg-card/60 p-5 backdrop-blur-sm">
          {me === null ? (
            <div className="h-11 w-full animate-pulse rounded-lg bg-secondary" />
          ) : (
            <>
              <Button
                size="lg"
                variant={me.google_enabled ? "default" : "outline"}
                className="w-full"
                disabled={!me.google_enabled}
                onClick={() => {
                  window.location.href = authUrls.google;
                }}
              >
                <GoogleMark />
                Continue with Google
              </Button>
              {!me.google_enabled ? (
                <p className="mt-2 text-center text-[11px] text-muted-foreground">
                  Google sign-in isn&apos;t configured on this server.
                </p>
              ) : null}

              {me.dev_login ? (
                <>
                  <div className="my-4 flex items-center gap-3">
                    <span className="h-px flex-1 bg-border" />
                    <span className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
                      dev
                    </span>
                    <span className="h-px flex-1 bg-border" />
                  </div>
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      if (devEmail.trim())
                        window.location.href = authUrls.devLogin(devEmail.trim());
                    }}
                    className="flex gap-2"
                  >
                    <input
                      value={devEmail}
                      onChange={(e) => setDevEmail(e.target.value)}
                      placeholder="you@company.com"
                      type="email"
                      className="h-10 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none placeholder:text-muted-foreground/60 focus-visible:border-input focus-visible:ring-2 focus-visible:ring-ring/40"
                    />
                    <Button type="submit" variant="outline" size="default">
                      Enter
                    </Button>
                  </form>
                </>
              ) : null}
            </>
          )}
        </div>

        <p className="mt-6 text-center text-[11px] leading-relaxed text-muted-foreground/70">
          A one-time consent to enroll a persistent voiceprint — not a per-meeting
          recording notice.
        </p>
      </div>
    </main>
  );
}
