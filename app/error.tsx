"use client";

import { useEffect } from "react";

export default function Error({
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
    <main className="min-h-screen w-full bg-[#050816] px-4 py-4 text-white">
      <div className="mx-auto flex min-h-[calc(100vh-2rem)] w-full max-w-2xl items-center justify-center">
        <section className="w-full rounded-[28px] border border-red-500/20 bg-red-500/10 p-6 shadow-2xl shadow-black/20">
          <p className="text-xs uppercase tracking-[0.2em] text-red-200/80">
            Something went wrong
          </p>
          <h1 className="mt-3 text-2xl font-semibold text-white">
            Samvaad hit an unexpected error.
          </h1>
          <p className="mt-3 text-sm leading-6 text-red-100/90">
            The current session is safe. You can try again without losing the overall app flow.
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
      </div>
    </main>
  );
}