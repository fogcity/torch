/**
 * vector element-wise binary operators fn.
 */

export function getSize(t: number[] | number[][] | number[][][]) {
  const dim = [];
  for (;;) {
    dim.push(t.length);
    if (Array.isArray(t[0])) {
      t = t[0];
    } else {
      break;
    }
  }
  return dim;
}

export function depth(
  t: number[] | number[][] | number[][][] | number
): number {
  return (
    1 +
    (t instanceof Array
      ? (t as number[]).reduce(function (max, item) {
          return Math.max(max, depth(item));
        }, 0)
      : -1)
  );
}
/**
 * 查找数组中出现次数最多的项
 * @param t
 * @returns
 */
export function most(t: number[]) {
  return [...t]
    .sort(
      (a, b) =>
        t.filter((v) => v === a).length - t.filter((v) => v === b).length
    )
    .pop();
}
/**
 * 打乱数组顺序
 * @param array
 * @returns
 */
export function shuffle(array: number[]) {
  let resultArray = array;
  let currentIndex = resultArray.length,
    randomIndex;
  while (currentIndex != 0) {
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;
    [resultArray[currentIndex], resultArray[randomIndex]] = [
      resultArray[randomIndex],
      resultArray[currentIndex],
    ];
  }
  return resultArray;
}

export function abs(t: number[]) {
  return t.map((v) => Math.abs(v));
}
export function exp(t: number[]) {
  return t.map((v) => Math.exp(v));
}

export function argMin(t: number[]) {
  return t.indexOf(Math.min(...t));
}

export function transpose(input: number[][]) {
  return input[0].map((_, colIndex) => input.map((row) => row[colIndex]));
}

export function square(a: number[]) {
  return a.map((v) => v ** 2);
}

export function argMax(t: number[]) {
  return t.indexOf(Math.max(...t));
}
export function sum(t: number[]) {
  return t.reduce((pre, v) => (pre += v), 0);
}
export function dot(a: number[], b: number[]) {
  return a.reduce((k, v, i) => {
    k = k + v * b[i];
    return k;
  }, 0);
}
/**
 * Subtracts other, scaled by alpha, from input.
 * out_i = input _i − alpha×other_i
 * @param a
 * @param b
 * @param alpha
 * @returns
 */
export function sub(a: number[], b: number[], alpha = 1) {
  return a.map((v, i) => v - alpha * b[i]);
}
export function div(a: number[], b: number[], alpha = 1) {
  return a.map((v, i) => (v / alpha) * b[i]);
}
export function add(a: number[], b: number[], alpha = 1) {
  return a.map((v, i) => v + alpha * b[i]);
}
export function mul(a: number[], b: number[]) {
  return hadamard(a, b);
}

export function mean(t: number[]) {
  return sum(t) / t.length;
}

export function range(stop: number, start: number = 0, step: number = 1) {
  const ra: number[] = [];
  for (let i = start; i < stop; i += step) {
    ra.push(i);
  }
  return ra;
}
export function softmax(t: number[]) {
  const st = sum(exp(t));
  return t.map((v) => Math.exp(v) / st);
}
export function randZeroOne() {
  return Math.round(Math.random());
}
// 获得高斯分布随机值
export function gusValue() {
  let u = 0,
    v = 0;
  while (u === 0) u = Math.random(); //Converting [0,1) to (0,1)
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function randn(size: number[], valueFn = gusValue) {
  const result = [];
  switch (size.length) {
    case 1:
      for (let i = 0; i < size[0]; i++) {
        result.push(valueFn());
      }
      break;
    case 2:
      for (let i = 0; i < size[0]; i++) {
        const subResult = [];
        for (let j = 0; j < size[1]; j++) {
          subResult.push(valueFn());
        }
        result.push(subResult);
      }
      break;
    case 3:
      for (let i = 0; i < size[0]; i++) {
        const subResult = [];
        for (let j = 0; j < size[1]; j++) {
          const subSubResult = [];
          for (let k = 0; k < size[2]; k++) {
            subSubResult.push(valueFn());
          }
          subResult.push(subSubResult);
        }
        result.push(subResult);
      }
      break;
    default:
      break;
  }

  return result;
}

export function hadamard(a: number[], b: number[]) {
  const r = [];
  for (let i = 0; i < a.length; i++) {
    r.push(a[i] * b[i]);
  }
  return r;
}

// 将两个向量转置重组
export function zip(a: number[], b: number[]) {
  const r = [];
  for (let i = 0; i < a.length; i++) {
    r.push([a[i], b[i]]);
  }
  return r;
}
