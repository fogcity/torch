import React from "react";
import { createRoot } from "react-dom/client";
import { NN } from "./NN";
import { Test } from "./Test";
const container = document.getElementById("root");
const root = createRoot(container);

root.render(
  <React.StrictMode>
    <Test />
  </React.StrictMode>
);
