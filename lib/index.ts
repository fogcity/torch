import { Loss } from "./loss";
import { backProp } from "./optim";
import { randZeroOne, randn, softmax } from "./array";
const nn = {
  MSELoss: Loss.MSE,
};
const optim = {
  SGD: backProp,
};
export { nn, optim, randZeroOne, randn, softmax };
