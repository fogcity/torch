/** Function that computes a penalty cost for a given weight in the network. */
export interface RegularizationFunction {
  output: (weight: number) => number;
  der: (weight: number) => number;
}

/** Build-in regularization functions */
export class Regularization {
  public static L1: RegularizationFunction = {
    output: (w) => Math.abs(w),
    der: (w) => (w < 0 ? -1 : w > 0 ? 1 : 0),
  };
  public static L2: RegularizationFunction = {
    output: (w) => 0.5 * w * w,
    der: (w) => w,
  };
}
