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
      ...options,
    };

    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.#gradientDescent();
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

  #gradientDescent() {
    const guesses = this.features.matMul(this.weights);
    const diffs = guesses.sub(this.labels);

    const slopes = this.features
      .transpose()
      .matMul(diffs)
      .div(this.features.shape[0]);

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
