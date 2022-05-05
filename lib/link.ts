import { Node } from "./node";
import { Regularization, RegularizationFunction } from "./regularization";

/**
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (loss derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
export class Link {
  id: string;
  source: Node;
  dest: Node;
  weight = Math.random() - 0.2;
  isDead = false;
  /** Loss derivative with respect to this weight. */
  lossDer = 0;
  /** Accumulated loss derivative since the last update. */
  accLossDer = 0;
  /** Number of accumulated derivatives since the last update. */
  numAccumulatedDers = 0;
  regularization: RegularizationFunction;

  /**
   * Constructs a link in the neural network initialized with random weight.
   *
   * @param source The source node.
   * @param dest The destination node.
   * @param regularization The regularization function that computes the
   *     penalty for this weight. If null, there will be no regularization.
   */
  constructor(
    source: Node,
    dest: Node,
    regularization: RegularizationFunction,
    initZero?: boolean
  ) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
    this.regularization = regularization;
    if (initZero) {
      this.weight = 0;
    }
  }
}
