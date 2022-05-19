import { Node } from "./node";
import { forwardProp } from "./nn";
import { mean } from "d3";
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
 * Creates a criterion that measures the mean absolute error (MAE) between each element in the input and target.
 */
export function mea() {}

/**
 * Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input and target.
 */
export function mse(network: Node[][], inputs: number[][], labels: number[]) {
  let i = 0;
  const r = function () {
    i = 0;
    return getLoss(network, inputs, labels, Loss.MSE);
  };
  r.backProp = function () {
    backProp(network, labels[i], Loss.MSE);
    i++;
  };
  return r;
}
/**
 * This criterion computes the cross entropy loss between input and target.
 */
export function crossEntropy() {}
/**
 * Measures the loss given an input tensor xx and a labels tensor yy (containing 1 or -1).
 */
export function hingeEmbedding() {}
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
  labels: number[],
  lossFn: LossFunction
): number {
  const loss = inputs.reduce((a, v, i) => {
    let output = forwardProp(network, v);
    console.log("1", output);

    const err = lossFn.output(output, labels[i]);
    a += err;
    return a;
  }, 0);
  return loss / inputs.length;
}
/**
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the loss
 * derivatives with respect to each node, and each weight
 * in the network.
 */
export function backProp(
  network: Node[][],
  target: number,
  lossFunc: LossFunction
) {
  // The output node is a special case. We use the user-defined loss
  // function for the derivative.

  let outputNode = network[network.length - 1][0];

  outputNode.outputDer = lossFunc.der(outputNode.output, target);
  // Go through the layers backwards.

  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    const currentLayer = network[layerIdx];

    // Compute the loss derivative of each node with respect to:
    // 1) its total input
    // 2) each of its input weights.

    // update node:

    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];

      node.inputDer = node.outputDer * node.activation.der(node.totalInput);
      node.accInputDer += node.inputDer;
      node.numAccumulatedDers++;
    }

    // update links:

    // Loss derivative with respect to each weight coming into the node.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }

        link.lossDer = node.inputDer * link.source.output;
        link.accLossDer += link.lossDer;
        link.numAccumulatedDers++;
      }
    }
    if (layerIdx === 1) {
      continue;
    }
    let prevLayer = network[layerIdx - 1];

    for (let i = 0; i < prevLayer.length; i++) {
      let node = prevLayer[i];
      // Compute the loss derivative with respect to each node's output.
      node.outputDer = 0;
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        node.outputDer += output.weight * output.dest.inputDer;
      }
    }
  }
}
