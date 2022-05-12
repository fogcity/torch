/** A node's activation function and its derivative. */
export interface ActivationFunction {
  output: (input: number) => number;
  der: (input: number) => number;
}
/** Built-in activation functions */
export class Activation {
  public static TANH: ActivationFunction = {
    output: (x) => Math.tanh(x),
    der: (x) => {
      let output = Activation.TANH.output(x);
      return 1 - output * output;
    },
  };
  public static RELU: ActivationFunction = {
    output: (x) => Math.max(0, x),
    der: (x) => (x <= 0 ? 0 : 1),
  };
  public static SIGMOID: ActivationFunction = {
    output: (x) => 1 / (1 + Math.exp(-x)),
    der: (x) => {
      let output = Activation.SIGMOID.output(x);

      return output * (1 - output);
    },
  };
  public static LINEAR: ActivationFunction = {
    output: (x) => x,
    der: (x) => 1,
  };
}
