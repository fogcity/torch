import { Node } from "./node";
import { forwardProp } from "./nn";
type LossOutputFunction = {
  (output: number, target: number, gmama: number): number;
  (output: number, target: number): number;
};
type LossDerFunction = {
  (output: number, target: number, gmama: number): number;
  (output: number, target: number): number;
};
/**
 * An loss function and its derivative.
 */
export interface LossFunction {
  output: LossOutputFunction;
  der: LossDerFunction;
}

/** Built-in loss functions */
export class Loss {
  public static HINGE: LossFunction = {
    output: (output: number, target: number, gmama: number = 1) =>
      Math.max(0, output - target + gmama),
    der: (output: number, target: number, gmama: number = 1) =>
      output - target + gmama > 0 ? 1 : 0,
  };

  public static FOCAL: LossFunction = {
    output: (output: number, target: number, gmama: number = 2) =>
      target
        ? -((1 - output) ** gmama) * Math.log(output)
        : -(output ** gmama) * Math.log(1 - output),
    der: (output: number, target: number, gmama: number = 2) =>
      target
        ? -(gmama * (1 - output)) * -Math.log(output) +
          -((1 - output) ** gmama) / output
        : gmama * output * -Math.log(1 - output) +
          output ** gmama / (1 - output),
  };
  public static BCE: LossFunction = {
    output: (output: number, target: number) =>
      target ? -Math.log(output) : -Math.log(1 - output),
    der: (output: number, target: number) =>
      target ? -1 / output : 1 / (1 - output),
  };
  public static CE: LossFunction = {
    output: (output: number, target: number) => -target * Math.log(output),
    der: (output: number, target: number) => -target / output,
  };
  public static MSE: LossFunction = {
    output: (output: number, target: number) =>
      0.5 * Math.pow(output - target, 2),
    der: (output: number, target: number) => output - target,
  };
  public static MAE: LossFunction = {
    output: (output: number, target: number) => Math.abs(output - target),
    der: (output: number, target: number) => (output - target > 0 ? 1 : -1),
  };
}

/**
 * Runs a forward propagation of the provided input through the provided
 * network. This method modifies the internal state of the network - the
 * total input and output of each node in the network.
 *
 * @param network The neural network.
 * @param inputs The input array. Its length should match the number of input
 *     nodes in the network.
 * @return The final output of the network.
 */
export function getLoss(
  network: Node[][],
  inputs: number[][],
  labels: number[]
): number {
  const loss = inputs.reduce((a, v, i) => {
    let output = forwardProp(network, v);
    const err = Loss.MSE.output(output, labels[i]);
    a += err;
    return a;
  }, 0);
  return loss / inputs.length;
}
