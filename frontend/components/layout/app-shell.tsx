import type { ReactNode } from "react";

type AppShellProps = {
  sidebar: ReactNode;
  main: ReactNode;
  sources: ReactNode;
};

export function AppShell({ sidebar, main, sources }: AppShellProps) {
  return (
    <div className="app-grid-bg min-h-screen">
      <div className="mx-auto flex min-h-screen max-w-[1680px] flex-col px-4 py-4 sm:px-6 lg:px-8">
        <header className="mb-4 flex items-center justify-between rounded-panel border border-white/10 bg-surface-1/85 px-5 py-4 backdrop-blur">
          <div>
            <p className="font-display text-lg font-semibold tracking-tight text-text">Deep Research</p>
            <p className="text-sm text-muted">A research workspace built for long-form answers, traceable process, and structured results.</p>
          </div>
          <div className="hidden rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-xs text-muted md:block">
            Live SSE timeline + Markdown report + parsed references
          </div>
        </header>

        <div className="grid flex-1 gap-4 xl:grid-cols-[280px_minmax(0,1fr)_360px]">
          <aside className="hidden xl:block">{sidebar}</aside>
          <main className="min-h-0">{main}</main>
          <aside className="hidden xl:block">{sources}</aside>
        </div>

        <div className="mt-4 xl:hidden">{sources}</div>
      </div>
    </div>
  );
}
