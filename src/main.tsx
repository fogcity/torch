import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { softmax } from "../dist/tstorch.es";
const container = document.getElementById("root");
const root = createRoot(container);
const Main = () => {
  const a = [-0.2415, 0.6425, 0.4474];
  console.log(softmax(a));

  return null;
};
root.render(
  <React.StrictMode>
    <Main />
  </React.StrictMode>
);
