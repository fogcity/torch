import { add, getSize } from "./array";
class Tensor {
  item: number[];
  size: number[];
  requires_grad: boolean;
  grad: number;
  readonly name: string = "Tensor";
  private index: number = 0;
  [Symbol.iterator]() {
    return this;
  }

  constructor(source: number[], size?: number[], requires_grad = false) {
    this.item = source;
    source.map((v, i) => ((this as unknown as Array<number>)[i] = v));
    this.size = size || getSize(source);
    this.requires_grad = requires_grad;
    this.grad = 0;
  }
  private next() {
    if (this.index < this.item.length) {
      return { done: false, value: this.item[this.index++] };
    } else {
      this.index = 0;
      return { done: true };
    }
  }
  map(callbackfn: (value: number, index: number, array: number[]) => any) {
    return this.item.map(callbackfn);
  }
  reduce(
    callbackfn: (
      previousValue: number,
      currentValue: number,
      currentIndex: number,
      array: number[]
    ) => any,
    initialValue?: any
  ) {
    return this.item.reduce(callbackfn, initialValue);
  }
}

// export function tensor(arr: number[], size?: number[], requires_grad = true) {
//    return new Tensor(arr, size, requires_grad);
// }
export function tensor(
  source: number[],
  size?: number[],
  requires_grad = true
) {
  let item = source;
  const fakeArray: any = () => {};
  // item.map((v, i) => (fakeArray[i] = v));
  fakeArray["add"] = (t: any) => {
    item = add(item, t());
  };
  const handler = {
    [Symbol.iterator]: function () {},
    apply: function (target: any, thisArg: any, argumentsList: any) {
      return item;
    },
    get: function (target: { [x: string]: any }, prop: any, receiver: any) {
      if (!isNaN(Number(prop))) {
        return item[prop];
      }
      return target[prop];
    },
  };
  return new Proxy(fakeArray, handler);
}
