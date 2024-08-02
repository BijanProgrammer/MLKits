const tf = require("@tensorflow/tfjs-node");

class LinearRegression {
  /**
   * @param {number[][]} features
   * @param {number[][]} labels
   * @param {{learningRate: number, iterations: number}} options
   */
  constructor(features, labels, options) {
    this.features = tf.tensor(features);
    this.labels = tf.tensor(labels);

    this.features = tf
      .ones([this.features.shape[0], 1])
      .concat(this.features, 1);

    this.options = {
      learningRate: 0.1,
      iterations: 1000,
      ...options,
    };

    this.weights = tf.zeros([2, 1]);
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.#gradientDescent();
    }
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
}

module.exports = LinearRegression;
