package org.acc.sgd.regression.noise;

import org.jblas.DoubleMatrix;

/**
 * Created by zhaoyy on 2016/12/26.
 */
public final class NoneNoiseUpdater implements NoiseUpdater {
    @Override
    public DoubleMatrix noiseOf(DoubleMatrix theta) {
        theta.checkColumns(1);
        return new DoubleMatrix(theta.getRows(), 1);
    }

    @Override
    public String toString() {
        return "None";
    }
}
