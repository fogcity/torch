export class Tensor {
  item: number[];
  require_grad: boolean;
  grad: number;
  name: string = "Tensor";
  constructor(arr: number[], public shape: number[], require_grad = true) {
    this.item = arr;
    this.require_grad = require_grad;
    this.grad = 0;
  }
}

export function tensor(arr: number[], shape: number[], require_grad = true) {
  return new Tensor(arr, shape, require_grad);
}
