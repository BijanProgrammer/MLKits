const tf = require("@tensorflow/tfjs-node");

const k = 2;

const features = tf.tensor([
  [-121, 47],
  [-121.2, 46.5],
  [-122, 46.4],
  [-120.9, 46.7],
]);

const labels = tf.tensor([[200], [250], [215], [240]]);

const predictionPoint = tf.tensor([-121, 47]);

const result =
  features
    .sub(predictionPoint)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1)
    .concat(labels, 1)
    .unstack()
    .sort((a, b) => a.arraySync()[0] - b.arraySync()[0])
    .slice(0, k)
    .reduce((sum, t) => sum + t.arraySync()[1], 0) / k;

console.log(result);
