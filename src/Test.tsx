import { useEffect } from "react";
import { dot, randn, size } from "../lib/array";
import { linear } from "../lib/layer";
const m = linear(2, 1, false);
const input = randn([1, 2]);
const y = m(input);

export function Test() {
  useEffect(() => {}, []);
  return <></>;
}
