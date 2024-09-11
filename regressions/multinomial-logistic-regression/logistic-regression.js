const tf = require("@tensorflow/tfjs-node");

class LogisticRegression {
  /**
   * @param {number[][]} features
   * @param {number[][]} labels
   * @param {{learningRate: number, iterations: number, batchSize: number }} options
   */
  constructor(features, labels, options) {
    /**
     * @type {Tensor<number>}
     */
    this.features = this.#processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = [];

    this.options = {
      learningRate: 0.1,
      iterations: 1000,
      batchSize: 1,
      ...options,
    };

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  }

  train() {
    const { batchSize } = this.options;

    const batchCount = Math.floor(this.features.shape[0] / batchSize);

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchCount; j++) {
        const startIndex = j * batchSize;

        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1],
        );

        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        this.#gradientDescent(featureSlice, labelSlice);
      }

      this.#recordCost();
      this.#updateLearningRate();
    }
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);

    const incorrect = predictions.notEqual(testLabels).sum().dataSync();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  predict(observations) {
    return this.#processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1);
  }

  #gradientDescent(features, labels) {
    const guesses = features.matMul(this.weights).softmax();
    const diffs = guesses.sub(labels);

    const slopes = features.transpose().matMul(diffs).div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  /**
   * @param {number[][]} features
   * @return {Tensor}
   */
  #processFeatures(features) {
    let result = tf.tensor(features);

    if (this.mean && this.variance) {
      result = result.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      result = this.#standardize(result);
    }

    result = tf.ones([result.shape[0], 1]).concat(result, 1);

    return result;
  }

  #standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  #recordCost() {
    const guesses = this.features.matMul(this.weights).softmax();

    const termOne = this.labels.transpose().matMul(guesses.log());

    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(guesses.mul(-1).add(1).log());

    const cost = termOne
      .add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .dataSync();

    this.costHistory.unshift(cost);
  }

  #updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    const lastCost = this.costHistory[0];
    const secondToLastCost = this.costHistory[1];

    if (lastCost > secondToLastCost) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
