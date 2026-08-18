// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built into the FastAPI package so one `uvicorn` serves the app: there is no second web server to
// deploy, and no CORS. `base` matches the mount point in studio/api/app.py -- they must agree, so
// test_the_wizard_is_served asserts the built asset paths resolve under it.
export default defineConfig({
  plugins: [react()],
  base: "/app/",
  build: { outDir: "../api/static/app", emptyOutDir: true, sourcemap: true },
  // Dev only: `npm run dev` serves the UI while uvicorn owns the API on 8765.
  server: { proxy: { "/api": { target: "http://127.0.0.1:8765", changeOrigin: true } } },
  test: { environment: "happy-dom", globals: true, include: ["src/**/*.test.ts?(x)"] },
});
