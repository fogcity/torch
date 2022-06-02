import { add, div, dot, getSize, mul, sub } from "./array";
// class Tensor {
//   item: number[];
//   size: number[];
//   requires_grad: boolean;
//   grad: number;
//   readonly name: string = "Tensor";
//   private index: number = 0;
//   [Symbol.iterator]() {
//     return this;
//   }

//   constructor(source: number[], size?: number[], requires_grad = false) {
//     this.item = source;
//     source.map((v, i) => ((this as unknown as Array<number>)[i] = v));
//     this.size = size || getSize(source);
//     this.requires_grad = requires_grad;
//     this.grad = 0;
//   }
//   private next() {
//     if (this.index < this.item.length) {
//       return { done: false, value: this.item[this.index++] };
//     } else {
//       this.index = 0;
//       return { done: true };
//     }
//   }
//   map(callbackfn: (value: number, index: number, array: number[]) => any) {
//     return this.item.map(callbackfn);
//   }
//   reduce(
//     callbackfn: (
//       previousValue: number,
//       currentValue: number,
//       currentIndex: number,
//       array: number[]
//     ) => any,
//     initialValue?: any
//   ) {
//     return this.item.reduce(callbackfn, initialValue);
//   }
// }

// export function tensor(arr: number[], size?: number[], requires_grad = true) {
//    return new Tensor(arr, size, requires_grad);
// }
type Tensor = {
  (): number[];
  requires_grad: boolean;
  grad: number[];
  add(target: Tensor): void;
  sub(target: Tensor): void;
  div(target: Tensor): void;
  mul(target: Tensor): void;
  dot(target: Tensor): number;
};
export function tensor(source: number[], requires_grad = false) {
  let item = source;
  const t = () => {
    return item;
  };
  t.requires_grad = requires_grad;
  if (requires_grad) {
    t.grad = source.map((v) => 0);
  }
  t.add = (target: Tensor) => {
    item = add(item, target());
  };
  t.div = (target: Tensor) => {
    item = div(item, target());
  };
  t.sub = (target: Tensor) => {
    item = sub(item, target());
  };
  t.mul = (target: Tensor) => {
    item = mul(item, target());
  };
  t.dot = (target: Tensor) => {
    return dot(item, target());
  };
  const handler = {
    get: function (target: { [x: string]: any }, prop: any, receiver: any) {
      if (!isNaN(Number(prop))) {
        return item[prop];
      }
      return target[prop];
    },
  };
  return new Proxy(t, handler) as Tensor;
}
