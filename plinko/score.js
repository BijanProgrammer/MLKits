import _ from "lodash";

const outputs = [];
const testSize = 50;

export function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

export function runAnalysis() {
  // Write code here to analyze stuff
  const [testSet, trainingSet] = splitDataset(outputs, testSize);

  _.range(1, 15, 1).forEach((k) => {
    const accuracy = _.chain(testSet)
      .filter(
        (testPoint) =>
          knn(trainingSet, _.initial(testPoint), k) === testPoint[3],
      )
      .size()
      .divide(testSize)
      .value();

    console.log(`Accuracy (k=${k}): ${accuracy}`);
  });
}

function knn(data, point, k) {
  return _.chain(data)
    .map((row) => {
      return [distance(_.initial(row), point), _.last(row)];
    })
    .sortBy((row) => row[0])
    .slice(0, k)
    .countBy((row) => row[1])
    .toPairs()
    .sortBy((row) => row[1])
    .last()
    .first()
    .parseInt()
    .value();
}

function distance(pointA, pointB) {
  return (
    _.chain(pointA)
      .zip(pointB)
      .map(([a, b]) => (a - b) ** 2)
      .sum()
      .value() ** 0.5
  );
}

function splitDataset(data, testSize) {
  const shuffled = _.shuffle(data);

  const testSet = _.slice(shuffled, 0, testSize);
  const trainingSet = _.slice(shuffled, testSize);

  return [testSet, trainingSet];
}
