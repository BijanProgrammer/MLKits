const fs = require("fs/promises");
const mnist = require("mnist-data");

const LogisticRegression = require("./logistic-regression");

function loadTrainData() {
  const trainData = mnist.training(0, 60000);

  const features = trainData.images.values.map((image) => image.flat());

  const labels = trainData.labels.values.map((label) => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
  });

  return {
    features,
    labels,
  };
}

const { features, labels } = loadTrainData();

const testData = mnist.testing(0, 100);

const testFeatures = testData.images.values.map((image) => image.flat());

const testLabels = testData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

const regression = new LogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 20,
  batchSize: 100,
});

regression.train();

console.log(regression.test(testFeatures, testLabels));

(async function () {
  // await fs.writeFile("./features.json", JSON.stringify(features[0]));

  await fs.writeFile(
    "./weights.json",
    JSON.stringify(regression.weights.arraySync()),
  );

  await fs.writeFile(
    "./mean.json",
    JSON.stringify(regression.mean.arraySync()),
  );

  await fs.writeFile(
    "./variance.json",
    JSON.stringify(regression.variance.arraySync()),
  );
})();
