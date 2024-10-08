const loadCsv = require("../load-csv");
const LinearRegression = require("./linear-regression");

let { features, labels, testFeatures, testLabels } = loadCsv("data/cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "displacement", "weight"],
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
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10,
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);

console.log(`R2: ${r2}`);

regression.predict([[150, 400, 1.8805]]).print();
