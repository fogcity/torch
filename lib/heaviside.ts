import { tensor } from "./tensor";
import { GPU, IKernelFunctionThis } from "gpu.js";

/**
 * Computes the Heaviside step function for each element in input.
 * @param input (Tensor) – the input tensor.
 * @param values (Tensor) – The values to use where input is zero.
 * @returns out (Tensor, optional) – the output tensor.
 */
export function heaviside(input: number[], values: number) {
  const result = [];
  for (let i = 0; i < input.length; i++) {
    const element = input[i];
    result.push(element == 0 ? values : element > 1 ? 1 : -1);
  }
  return result;
}
