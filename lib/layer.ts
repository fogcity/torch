import { Activation, ActivationFunction } from "./activation";
import { dot, randn, transpose } from "./array";

class Layer {
  bias!: number[];
  weight!: number[][];
  activation!: Activation;
  constructor(
    inFeatures: number,
    outFeatures: number,
    bias = true,
    activation: ActivationFunction
  ) {
    this.weight = randn([outFeatures, inFeatures]) as number[][];
    if (bias) {
      this.bias = randn([outFeatures]) as number[];
    }
    this.activation;
  }

  forward(input: number[][]) {
    return input.map((x, i) => {
      const y = this.weight.map((w) => {
        let wx = dot(w, x);
        if (this.bias) {
          wx += this.bias[i];
        }
        return wx;
      });
      return y;
    });
  }
}

/**
 * Applies a linear transformation to the incoming data: y = xA^T + b
 * @param size
 * @returns
 */
export function linear(inFeatures: number, outFeatures: number, bias = true) {
  const l = new Layer(inFeatures, outFeatures, bias, Activation.LINEAR);
  const layer = function (input: number[][]) {
    return l.forward(input);
  };
  layer.weight = l.weight;
  if (l.bias) layer.bias = l.bias;
  return layer;
}
