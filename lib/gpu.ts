import { Tensor } from "./tensor";
import { GPU } from "gpu.js";
const gpu = new GPU();

// export const dot = (t1: Tensor, t2: Tensor) => {
//   return (
//     gpu
//       .createKernel(function (a: number[], b: number[], size: number) {
//         let sum = 0;
//         for (let i = 0; i < size; i++) {
//           sum += a[i] * b[i];
//         }
//         return sum;
//       })
//       .setOutput([1])(t1, t2, t1.length) as number[]
//   )[0];
// };
export const dot = function (a: number[][], b: number[][]) {
  return a.map((v, i) => {
    let sum = 0;
    for (let j = 0; j < a.length; j++) {
      sum += v[j] * b[i][j];
    }
    return sum;
  });
};
