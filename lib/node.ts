import { ActivationFunction } from "./activation";
import { Link } from "./link";

/**
 * A node in a neural network. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
export class Node {
  id: string;
  /** List of input Links. */
  inputLinks: Link[] = [];
  bias = 0.01;
  /** List of output Links. */
  outputs: Link[] = [];
  totalInput!: number;
  output!: number;
  /** Loss derivative with respect to this node's output. */
  outputDer = 0;
  /** Loss derivative with respect to this node's total input. */
  inputDer = 0;
  /**
   * Accumulated loss derivative with respect to this node's total input since
   * the last update. This derivative equals dE/db where b is the node's
   * bias term.
   */
  accInputDer = 0;
  /**
   * Number of accumulated err. derivatives with respect to the total input
   * since the last update.
   */
  numAccumulatedDers = 0;
  /** Activation function that takes total input and returns node's output */
  activation: ActivationFunction;

  /**
   * Creates a new node with the provided id and activation function.
   */
  constructor(id: string, activation: ActivationFunction, initZero?: boolean) {
    this.id = id;
    this.activation = activation;
    if (initZero) {
      this.bias = 0;
    }
  }

  /** Recomputes the node's output and returns it. */
  updateOutput(): number {
    // Stores total input into the node.
    this.totalInput = this.bias;
    for (let j = 0; j < this.inputLinks.length; j++) {
      let Link = this.inputLinks[j];
      this.totalInput += Link.weight * Link.source.output;
    }
    this.output = this.activation.output(this.totalInput);
    return this.output;
  }
}
