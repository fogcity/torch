export function getAverage(t: number[]) {
  return (
    (t as number[]).reduce((previous, current) => (current += previous)) /
    t.length
  );
}
export function getShape(t: number[] | number[][] | number[][][]) {
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
/**
 * 查找数组深度
 * @param t
 * @returns
 */
export function getDepth(
  t: number[] | number[][] | number[][][] | number
): number {
  return (
    1 +
    (t instanceof Array
      ? (t as number[]).reduce(function (max, item) {
          return Math.max(max, getDepth(item));
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
export function sub(a: number[], b: number[]) {
  return a.map((v, i) => v - b[i]);
}
export function add(a: number[], b: number[]) {
  return a.map((v, i) => v + b[i]);
}
/**
 * 求加权平均值
 * @param t
 * @returns
 */
export function mean(t: number[]) {
  return sum(t) / t.length;
}
/**
 * 生成数组
 * @param stop 直到哪个数
 * @param start 从哪个数开始
 * @param step 步长
 * @returns
 */
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
export function randn() {
  let u = 0,
    v = 0;
  while (u === 0) u = Math.random(); //Converting [0,1) to (0,1)
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * 计算数组的hadamard乘积
 * @param a 左边的数组
 * @param b 右边的数组
 * @returns 新的数组
 */
export function hadamard(a: number[], b: number[]) {
  return a.reduce((r: number[], v, i) => {
    r.push(b[i] * v);
    return r;
  }, []);
}

// 将两个向量转置重组
export function zip(a: number[], b: number[]) {
  return a.reduce((s: number[][], v, i) => {
    s.push([v, b[i]]);
    return s;
  }, []);
}
