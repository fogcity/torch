import { add } from "./add";

import { dot } from "./gpu";
import { heaviside } from "./heaviside";

export function perceptron(
  x: number[][],
  w: number[][],
  b: number[],
  activation = heaviside
) {
  return function (values: number) {
    const wx = dot(w, x);
    const z = add(wx, b);
    return activation(z, values);
  };
}

export class Logical {
  public static AND(input: number[][], values: number) {
    return perceptron(input, [[1.0, 1.0]], [-1.5])(values);
  }
  public static OR(input: number[][], values: number) {
    return perceptron(input, [[1.0, 1.0]], [-1.5])(values);
  }
  public static NOT(input: number[][], values: number) {
    return perceptron(input, [[-1]], [0.5])(values);
  }
}
