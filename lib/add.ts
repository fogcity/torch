import { Tensor } from "./tensor";

export function add(input: number[], other: number[] | number) {
  if (typeof other == "number") {
    return input.map((v) => (v as number) + other);
  } else return input.map((v, i) => (v as number) + other[i]);
}
