import { LossFunction } from "./loss";
import { Node } from "./node";
import { Regularization } from "./regularization";

class Optimizer {}
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
    let currentLayer = network[layerIdx];

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

/**
 * Updates the weights of the network using the previously accumulated loss
 * derivatives.
 */
export function updateWeights(
  network: Node[][],
  learningRate: number,
  regularizationRate: number
) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    // Select second layer
    let currentLayer = network[layerIdx];
    // Foreach the second layer's node
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // Update the node's bias.
      if (node.numAccumulatedDers > 0) {
        node.bias -=
          (learningRate * node.accInputDer) / node.numAccumulatedDers;
        node.accInputDer = 0;
        node.numAccumulatedDers = 0;
      }
      // Update the weights coming into this node.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        let regulDer = link.regularization
          ? link.regularization.der(link.weight)
          : 0;
        if (link.numAccumulatedDers > 0) {
          // Update the weight based on dE/dw.
          link.weight =
            link.weight -
            (learningRate / link.numAccumulatedDers) * link.accLossDer;
          // Further update the weight based on regularization.
          let newLinkWeight =
            link.weight - learningRate * regularizationRate * regulDer;
          if (
            link.regularization === Regularization.L1 &&
            link.weight * newLinkWeight < 0
          ) {
            // The weight crossed 0 due to the regularization term. Set it to 0.
            link.weight = 0;
            link.isDead = true;
          } else {
            link.weight = newLinkWeight;
          }
          link.accLossDer = 0;
          link.numAccumulatedDers = 0;
        }
      }
    }
  }
}
