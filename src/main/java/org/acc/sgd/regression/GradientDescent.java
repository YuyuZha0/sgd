package org.acc.sgd.regression;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import org.acc.sgd.model.LabeledPoint;
import org.acc.sgd.model.ResultModel;
import org.acc.sgd.regression.learning.LearningRateUpdater;
import org.acc.sgd.regression.noise.NoiseUpdater;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by zhaoyy on 2016/11/23.
 */
public final class GradientDescent {

    protected static final Logger logger = LoggerFactory.getLogger(GradientDescent.class);

    private final double learningRate;
    private final int iterations;
    private final Sampler sampler;
    private final LearningRateUpdater learningRateUpdater;
    private final Hypothesis hypothesis;
    private final NoiseUpdater noiseUpdater;
    private final ImmutableList<LabeledPoint> samples;
    private final int dimension;
    private final int batchSize;
    private final double epsilon;

    private GradientDescent(Builder builder, ImmutableList<LabeledPoint> samples) {
        logger.info("init gradient descent model with params:{}", builder);
        this.learningRate = builder.learningRate;
        this.iterations = builder.iterations;
        this.batchSize = builder.batchSize;
        this.sampler = builder.sampler;
        this.hypothesis = builder.hypothesis;
        this.learningRateUpdater = builder.learningRateUpdater;
        this.noiseUpdater = builder.noiseUpdater;
        this.epsilon = builder.epsilon;
        this.samples = samples;
        this.dimension = getDimension(samples);
    }

    public static Builder newGradientDescent() {
        return new Builder();
    }


    private static int getDimension(List<LabeledPoint> points) {
        int dimension = 0;
        for (LabeledPoint point : points) {
            //index is 1-based
            int index = point.maxIndex();
            dimension = dimension < index ? index : dimension;
        }
        return dimension;
    }

    private static DoubleMatrix getX(LabeledPoint[] points, int dimension) {
        int rows = points.length;
        double[][] data = new double[rows][];
        for (int i = 0; i < data.length; i++) {
            data[i] = points[i].featureArray(1, dimension + 1);
            data[i][0] = 1;
        }
        return new DoubleMatrix(data);
    }


    //DoubleMatrix 一维数组默认创建的是列向量
    private static DoubleMatrix getLabel(LabeledPoint[] points) {
        int rows = points.length;
        double[] label = new double[rows];
        for (int i = 0; i < rows; i++)
            label[i] = points[i].label();
        return new DoubleMatrix(label);
    }

    private static double epsilon(DoubleMatrix error, DoubleMatrix lastError) {
        if (lastError == null)
            return 9999;
        double delta = error
                .sub(lastError)
                .normmax();
        return delta;
    }

    private LabeledPoint[] sample() {
        int[] indices = sampler.sample(batchSize, samples.size());
        LabeledPoint[] points = new LabeledPoint[indices.length];
        for (int i = 0; i < indices.length; i++)
            points[i] = samples.get(indices[i]);
        return points;
    }

    public ResultModel run() {

        logger.info("start running gradient descent...");

        long st = System.currentTimeMillis();
        DoubleMatrix theta = new DoubleMatrix(dimension + 1, 1);
        DoubleMatrix lastError = null;
        double eta = learningRate;
        double epsilon = this.epsilon;
        for (int i = 0; i < iterations; i++) {
            LabeledPoint[] samples = sample();
            DoubleMatrix x = getX(samples, dimension);
            DoubleMatrix label = getLabel(samples);
            DoubleMatrix error = hypothesis
                    .apply(x, theta)
                    .sub(label);
            if ((epsilon = epsilon(error, lastError)) < this.epsilon) {
                logger.info("iteration ended beforehand,total iteration [{}]", i);
                break;
            } else {
                lastError = error;
            }
            DoubleMatrix delta = x
                    .transpose()
                    .mmul(error)
                    .add(noiseUpdater.noiseOf(theta));
            theta = theta.sub(delta.mmul(eta / x.getRows()));
            eta = learningRateUpdater.nextRate(learningRate, theta);
        }
        logger.info("training finished,current epsilon:[{}],duration:[{}]ms", epsilon, System.currentTimeMillis() - st);

        return new ResultModel(theta.toArray());
    }

    public static class Builder {

        private double learningRate = 0;
        private int iterations = 0;
        private int batchSize = 10;
        private double epsilon = 0.1;
        private Sampler sampler = null;
        private Hypothesis hypothesis = null;
        private LearningRateUpdater learningRateUpdater = null;
        private NoiseUpdater noiseUpdater = null;

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder epsilon(double epsilon) {
            this.epsilon = epsilon;
            return this;
        }

        public Builder iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder sampler(Sampler sampler) {
            this.sampler = sampler;
            return this;
        }

        public Builder hypothesis(Hypothesis hypothesis) {
            this.hypothesis = hypothesis;
            return this;
        }

        public Builder learningRateUpdater(LearningRateUpdater learningRateUpdater) {
            this.learningRateUpdater = learningRateUpdater;
            return this;
        }

        public Builder noiseUpdater(NoiseUpdater noiseUpdater) {
            this.noiseUpdater = noiseUpdater;
            return this;
        }


        private void check() {
            Preconditions.checkArgument(learningRate > 0, "learning rate must be a positive number");
            Preconditions.checkArgument(iterations > 0, "iterations must be a positive number");
            Preconditions.checkArgument(batchSize > 0, "batchSize must be a positive number");
            Preconditions.checkArgument(epsilon > 0, "epsilon must be a positive number");
            Preconditions.checkNotNull(sampler, "sampler can't be null");
            Preconditions.checkNotNull(hypothesis, "hypothesis can't be null");
            Preconditions.checkNotNull(learningRateUpdater, "learning rate noise can't be null");
            Preconditions.checkNotNull(noiseUpdater, "noise noise can't be null");
        }

        @Override
        public String toString() {
            Map<String, Object> map = new HashMap<String, Object>();
            map.put("learningRate", learningRate);
            map.put("iterations", iterations);
            map.put("batchSize", batchSize);
            map.put("epsilon", epsilon);
            map.put("sampler", sampler);
            map.put("hypothesis", hypothesis);
            map.put("learningRateUpdater", learningRateUpdater);
            map.put("noiseUpdater", noiseUpdater);
            StringBuilder builder = new StringBuilder();
            map.forEach((k, v) -> builder
                    .append("\n\t")
                    .append(k)
                    .append(':')
                    .append(v));
            return builder.toString();
        }

        public GradientDescent build(ImmutableList<LabeledPoint> data) {
            check();
            return new GradientDescent(this, data);
        }
    }
}
