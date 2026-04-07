import type { Metadata } from "next";

import { QueryProvider } from "@/providers/query-provider";

import "./globals.css";

export const metadata: Metadata = {
  title: "Deep Research Workspace",
  description: "Premium research frontend for the Deep Research backend.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="bg-bg font-sans text-text">
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  );
}
