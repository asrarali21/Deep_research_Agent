import type { ReactNode } from "react";

type AppShellProps = {
  sidebar: ReactNode;
  main: ReactNode;
  sources: ReactNode;
};

export function AppShell({ sidebar, main, sources }: AppShellProps) {
  return (
    <div className="app-grid-bg min-h-screen overflow-hidden">
      <div className="mx-auto flex min-h-screen max-w-[1720px] flex-col px-3 py-3 sm:px-5 lg:px-6">
        <header className="mb-4 grid gap-3 border-b border-white/[0.08] pb-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-end">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-panel border border-accent/25 bg-accent/10 font-display text-sm font-semibold text-accent shadow-[0_12px_34px_rgba(99,222,169,0.12)]">
              DR
            </div>
            <div className="min-w-0">
              <p className="font-display text-xl font-semibold text-text">Deep Research</p>
              <p className="truncate text-sm text-muted">Ask sharper questions. Keep the evidence close.</p>
            </div>
          </div>
          <div className="inline-flex w-fit items-center gap-2 rounded-panel border border-white/[0.09] bg-white/[0.045] px-3 py-2 text-sm text-muted">
            <span className="h-2 w-2 rounded-full bg-success shadow-[0_0_18px_rgba(86,211,148,0.8)]" />
            Research desk
          </div>
        </header>

        <div className="grid flex-1 gap-3 xl:grid-cols-[260px_minmax(0,1fr)_340px] 2xl:grid-cols-[280px_minmax(0,1fr)_360px]">
          <aside className="hidden xl:block">{sidebar}</aside>
          <main className="min-h-0">{main}</main>
          <aside className="hidden xl:block">{sources}</aside>
        </div>

        <div className="mt-4 xl:hidden">{sources}</div>
      </div>
    </div>
  );
}
