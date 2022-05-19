import { Node } from "./node";
import { Regularization } from "./regularization";
class Optimizer {
  constructor(public network: Node[][], public lr: number, public rr: number) {}
  step() {
    updateWeights(this.network, this.lr, this.rr);
  }
}

export function sgd(network: Node[][], lr: number, rr: number) {
  return new Optimizer(network, lr, rr);
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
          link.weight -=
            (learningRate * link.accLossDer) / link.numAccumulatedDers;
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
