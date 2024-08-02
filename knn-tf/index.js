const tf = require("@tensorflow/tfjs-node");
const loadCsv = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
  const { mean, variance } = tf.moments(features, 0);

  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

  return (
    features
      .sub(mean)
      .div(variance.pow(0.5))
      .sub(scaledPrediction)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => a.arraySync()[0] - b.arraySync()[0])
      .slice(0, k)
      .reduce((sum, t) => sum + t.arraySync()[1], 0) / k
  );
}

function main() {
  const k = 10;

  let { features, labels, testFeatures, testLabels } = loadCsv(
    "kc_house_data.csv",
    {
      shuffle: true,
      splitTest: 10,
      dataColumns: [
        "lat",
        "long",
        "sqft_lot",
        "sqft_living",
        "yr_built",
        "condition",
        "grade",
      ],
      labelColumns: ["price"],
    },
  );

  features = tf.tensor(features);
  labels = tf.tensor(labels);

  testFeatures.forEach((testPoint, testIndex) => {
    const predictionPoint = tf.tensor(testPoint);
    const result = knn(features, labels, predictionPoint, k);

    const error =
      Math.floor(
        (Math.abs(testLabels[testIndex][0] - result) /
          testLabels[testIndex][0]) *
          100 *
          100,
      ) / 100;

    console.log({
      calculated: result,
      actual: testLabels[testIndex][0],
      accuracy: `${100 - error}%`,
    });
  });
}

main();
