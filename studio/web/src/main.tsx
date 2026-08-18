// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("no #root element: index.html and main.tsx disagree");
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
