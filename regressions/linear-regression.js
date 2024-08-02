const tf = require("@tensorflow/tfjs-node");

class LinearRegression {
  /**
   * @param {number[][]} features
   * @param {number[][]} labels
   * @param {{learningRate: number, iterations: number}} options
   */
  constructor(features, labels, options) {
    this.features = this.#processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];

    this.options = {
      learningRate: 0.1,
      iterations: 1000,
      batchSize: 1,
      ...options,
    };

    this.weights = tf.zeros([this.features.shape[1], 1]);
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

      this.#recordMse();
      this.#updateLearningRate();
    }
  }

  test(testFeatures, testLabels) {
    testFeatures = this.#processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);

    const ssTot = testLabels.sub(testLabels.mean()).pow(2).sum().dataSync();
    const ssRes = testLabels.sub(predictions).pow(2).sum().dataSync();

    return 1 - ssRes / ssTot;
  }

  predict(observations) {
    return this.#processFeatures(observations).matMul(this.weights);
  }

  #gradientDescent(features, labels) {
    const guesses = features.matMul(this.weights);
    const diffs = guesses.sub(labels);

    const slopes = features.transpose().matMul(diffs).div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  #processFeatures(features) {
    features = tf.tensor(features);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.#standardize(features);
    }

    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  #standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  #recordMse() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .dataSync();

    this.mseHistory.unshift(mse);
  }

  #updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }

    const lastMse = this.mseHistory[0];
    const secondToLastMse = this.mseHistory[1];

    if (lastMse > secondToLastMse) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LinearRegression;
