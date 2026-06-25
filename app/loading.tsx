export default function Loading() {
  return (
    <main className="min-h-screen w-full bg-[#050816] px-4 py-4 text-white">
      <div className="mx-auto flex min-h-[calc(100vh-2rem)] w-full max-w-[1600px] flex-col gap-4">
        <div className="rounded-[24px] border border-white/10 bg-white/5 px-4 py-3 shadow-2xl shadow-black/20">
          <div className="flex items-center justify-between gap-4">
            <div className="h-12 w-44 animate-pulse rounded-2xl bg-white/10" />
            <div className="flex gap-2">
              <div className="h-8 w-24 animate-pulse rounded-full bg-white/10" />
              <div className="h-8 w-24 animate-pulse rounded-full bg-white/10" />
            </div>
          </div>
        </div>

        <div className="grid flex-1 min-h-0 gap-4 xl:grid-cols-[minmax(0,2.1fr)_minmax(360px,0.9fr)]">
          <section className="flex min-h-0 flex-col rounded-[28px] border border-white/10 bg-white/5 p-4 shadow-2xl shadow-black/20">
            <div className="h-8 w-40 animate-pulse rounded-2xl bg-white/10" />
            <div className="mt-3 h-4 w-80 animate-pulse rounded-2xl bg-white/10" />

            <div className="mt-4 flex flex-wrap gap-2">
              <div className="h-9 w-24 animate-pulse rounded-full bg-white/10" />
              <div className="h-9 w-24 animate-pulse rounded-full bg-white/10" />
              <div className="h-9 w-24 animate-pulse rounded-full bg-white/10" />
              <div className="h-9 w-28 animate-pulse rounded-full bg-white/10" />
            </div>

            <div className="mt-4 flex-1 min-h-[420px] rounded-[24px] border border-dashed border-white/12 bg-slate-900/70">
              <div className="flex h-full w-full items-center justify-center">
                <div className="text-center">
                  <div className="mx-auto h-12 w-12 animate-pulse rounded-2xl bg-white/10" />
                  <div className="mt-4 h-5 w-48 animate-pulse rounded-2xl bg-white/10" />
                  <div className="mt-3 h-4 w-80 animate-pulse rounded-2xl bg-white/10" />
                </div>
              </div>
            </div>
          </section>

          <aside className="flex min-h-0 flex-col gap-4">
            <section className="rounded-[28px] border border-white/10 bg-white/5 p-4">
              <div className="h-7 w-36 animate-pulse rounded-2xl bg-white/10" />
              <div className="mt-3 h-4 w-56 animate-pulse rounded-2xl bg-white/10" />
              <div className="mt-4 rounded-2xl border border-white/10 bg-slate-900/80 p-4">
                <div className="h-3 w-28 animate-pulse rounded-2xl bg-white/10" />
                <div className="mt-3 h-8 w-44 animate-pulse rounded-2xl bg-white/10" />
                <div className="mt-4 space-y-2">
                  <div className="h-4 w-full animate-pulse rounded-2xl bg-white/10" />
                  <div className="h-4 w-full animate-pulse rounded-2xl bg-white/10" />
                  <div className="h-4 w-full animate-pulse rounded-2xl bg-white/10" />
                </div>
              </div>
            </section>

            <section className="rounded-[28px] border border-white/10 bg-white/5 p-4">
              <div className="h-7 w-36 animate-pulse rounded-2xl bg-white/10" />
              <div className="mt-4 h-36 animate-pulse rounded-2xl border border-white/10 bg-slate-900/80" />
            </section>
          </aside>
        </div>
      </div>
    </main>
  );
}