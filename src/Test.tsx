import { useEffect } from "react";
import { add, dot, randn } from "../lib/array";
import { linear } from "../lib/layer";
import { tensor } from "../lib/tensor";
const m = linear(2, 1, false);
const input = randn([1, 2]);
const y = m(input);

const b = tensor([1, 2]);
const c = tensor([3, 4]);
console.log(b[1]);
console.log(b());
b.add(c);
console.log("after", b());

console.log(c());

export function Test() {
  useEffect(() => {}, []);
  return <></>;
}
