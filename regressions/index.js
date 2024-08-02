const tf = require("@tensorflow/tfjs-node");
const loadCsv = require("./load-csv");
const LinearRegression = require("./linear-regression");

let { features, labels, testFeatures, testLabels } = loadCsv("cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower"],
  labelColumns: ["mpg"],
});

/*
features
[
  [ 90 ],  [ 150 ], [ 152 ], [ 75 ],  [ 110 ], [ 97 ],  [ 67 ]
]
 */

/*
labels
[
  [ 22.5 ], [ 16 ],   [ 14.5 ], [ 26 ],   [ 18.6 ], [ 23.9 ], [ 30 ]
]
 */

const regression = new LinearRegression(features, labels, {
  learningRate: 0.001,
  iterations: 1,
});

regression.train();

console.log(`m: ${regression.weights.arraySync()[1][0]}`);
console.log(`b: ${regression.weights.arraySync()[0][0]}`);
