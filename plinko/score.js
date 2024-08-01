import _ from "lodash";

const outputs = [];
const k = 10;
const testSize = 50;

export function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

export function runAnalysis() {
  // Write code here to analyze stuff

  _.range(0, 3).forEach((feature) => {
    const data = _.map(outputs, (row) => [row[feature], _.last(row)]);
    const [testSet, trainingSet] = splitDataset(minmax(data, 1), testSize);

    const accuracy = _.chain(testSet)
      .filter(
        (testPoint) =>
          knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint),
      )
      .size()
      .divide(testSize)
      .value();

    console.log(`Accuracy (feature=${feature}): ${accuracy}`);
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

function minmax(data, featureCount) {
  const clonedData = _.cloneDeep(data);

  for (let i = 0; i < featureCount; i++) {
    const column = clonedData.map((row) => row[i]);

    const min = _.min(column);
    const max = _.max(column);

    for (let j = 0; j < clonedData.length; j++) {
      clonedData[j][i] = (clonedData[j][i] - min) / (max - min);
    }
  }

  return clonedData;
}
