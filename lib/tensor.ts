export class Tensor {
  item: number[];
  require_grad: boolean;
  grad: number;
  name: string = "Tensor";
  constructor(arr: number[], public size: number[], require_grad = true) {
    this.item = arr;
    this.require_grad = require_grad;
    this.grad = 0;
  }
}

export function tensor(arr: number[], size: number[], require_grad = true) {
  return new Tensor(arr, size, require_grad);
}
