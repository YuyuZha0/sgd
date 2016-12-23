package org.acc.sgd.model;

import java.util.Arrays;

/**
 * Created by zhaoyy on 2016/12/23.
 */
public final class ResultModel {

    private final ImmutableVector weights;
    private final double intercept;

    public ResultModel(ImmutableVector weights, double intercept) {
        this.weights = weights;
        this.intercept = intercept;
    }

    public ResultModel(double[] a) {
        this(ImmutableVector.valueOf(Arrays.copyOfRange(a, 1, a.length)), a[0]);
    }

    public ImmutableVector getWeights() {
        return weights;
    }

    public double getIntercept() {
        return intercept;
    }

    @Override
    public String toString() {
        return new StringBuilder()
                .append("{weights:")
                .append(weights)
                .append(',')
                .append("intercept:")
                .append(intercept)
                .append("}")
                .toString();
    }
}
