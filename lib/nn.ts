import { ActivationFunction, Activation } from "./activation";
import { Link } from "./link";
import { Node } from "./node";
import { Regularization, RegularizationFunction } from "./regularization";

/**
 * Builds a neural network.
 *
 * @param size The size of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer nand 1 output node.
 * @param activation The activation function of every hidden node.
 * @param outputActivation The activation function for the output nodes.
 * @param regularization The regularization function that computes a penalty
 *     for a given weight (parameter) in the network. If null, there will be
 *     no regularization.
 * @param inputIds List of ids for the input nodes.
 */
export function buildNetwork(
  size: number[],
  activation: ActivationFunction = Activation.LINEAR,
  outputActivation: ActivationFunction = Activation.LINEAR,
  regularization: RegularizationFunction = Regularization.L2,
  initZero?: boolean
): Node[][] {
  let numLayers = size.length;
  let id = 1;
  /** List of layers, with each layer being a list of nodes. */
  let network: Node[][] = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    let isOutputLayer = layerIdx === numLayers - 1;
    let currentLayer: Node[] = [];
    network.push(currentLayer);
    let numNodes = size[layerIdx];
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString();
      id++;
      let node = new Node(
        nodeId,
        isOutputLayer ? outputActivation : activation,
        initZero
      );
      currentLayer.push(node);
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let link = new Link(prevNode, node, regularization, initZero);
          prevNode.outputs.push(link);
          node.inputLinks.push(link);
        }
      }
    }
  }
  return network;
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
export function forwardProp(network: Node[][], inputs: number[]): number {
  let inputLayer = network[0];
  console.log("inputLayer", inputLayer);

  if (inputs.length != inputLayer.length) {
    throw new Error(
      "The number of inputs must match the number of nodes in" +
        " the input layer"
    );
  }
  // Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i];
    node.output = inputs[i];
  }
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.updateOutput();
    }
  }
  return network[network.length - 1][0].output;
}

/**
 * Predict the target of the network
 */
export function predict(network: Node[][], target: number[]) {
  return forwardProp(network, target);
}

/** Iterates over every node in the network/ */
export function mapNode(network: Node[][], accessor: (node: Node) => any) {
  return network.slice(1, network.length).map((layer) => layer.map(accessor));
}

/** Returns the weights in the network. */
export function getWeights(network: Node[][]) {
  return network
    .slice(1, network.length)
    .map((layer) => layer.map((node) => node.inputLinks.map((v) => v.weight)));
} /** Returns the bias in the network. */

export function getBias(network: Node[][]) {
  return network
    .slice(1, network.length)
    .map((layer) => layer.map((node) => node.bias));
}
