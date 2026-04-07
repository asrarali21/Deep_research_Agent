import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    include: ["tests/frontend/**/*.test.{ts,tsx}"],
    pool: "forks",
  },
  esbuild: {
    jsxInject: 'import React from "react"',
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
});
