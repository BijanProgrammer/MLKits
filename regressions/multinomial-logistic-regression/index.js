const tf = require("@tensorflow/tfjs-node");
const loadCsv = require("../load-csv");
const LogisticRegression = require("./logistic-regression");

let { features, labels, testFeatures, testLabels } = loadCsv("data/cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "displacement", "weight"],
  labelColumns: ["mpg"],
  converters: {
    mpg: (value) => {
      const mpg = parseFloat(value);

      if (mpg < 15) {
        return [1, 0, 0];
      } else if (mpg < 30) {
        return [0, 1, 0];
      } else {
        return [0, 0, 1];
      }
    },
  },
});

labels = labels.map((x) => x[0]);
testLabels = testLabels.map((x) => x[0]);

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
});

regression.train();

console.log(regression.test(testFeatures, testLabels));
