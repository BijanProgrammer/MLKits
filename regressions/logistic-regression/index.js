const tf = require("@tensorflow/tfjs-node");
const loadCsv = require("../load-csv");
const LogisticRegression = require("./logistic-regression");

let { features, labels, testFeatures, testLabels } = loadCsv("data/cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "displacement", "weight"],
  labelColumns: ["passedemissions"],
  converters: {
    passedemissions: (value) => {
      return value === "TRUE" ? 1 : 0;
    },
  },
});

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 50,
  decisionBoundary: 0.5,
});

regression.train();

console.log(regression.test(testFeatures, testLabels));
