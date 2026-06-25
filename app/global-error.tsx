"use client";

import { useEffect } from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <html lang="en">
      <body className="min-h-screen bg-[#050816] text-white">
        <main className="flex min-h-screen w-full items-center justify-center px-4 py-4">
          <section className="w-full max-w-2xl rounded-[28px] border border-red-500/20 bg-red-500/10 p-6 shadow-2xl shadow-black/20">
            <p className="text-xs uppercase tracking-[0.2em] text-red-200/80">
              Critical error
            </p>
            <h1 className="mt-3 text-2xl font-semibold">
              Samvaad could not finish loading.
            </h1>
            <p className="mt-3 text-sm leading-6 text-red-100/90">
              The app can be restarted safely. Nothing in the workflow is intentionally changed here.
            </p>

            <div className="mt-6 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={reset}
                className="rounded-full border border-white/10 bg-white/90 px-4 py-2 text-sm font-medium text-slate-950 transition-colors hover:bg-white"
              >
                Try again
              </button>
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-white/8"
              >
                Reload
              </button>
            </div>
          </section>
        </main>
      </body>
    </html>
  );
}