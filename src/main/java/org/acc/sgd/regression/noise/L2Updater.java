package org.acc.sgd.regression.noise;

import org.jblas.DoubleMatrix;

/**
 * Created by zhaoyy on 2016/12/26.
 */
public final class L2Updater implements NoiseUpdater {
    @Override
    public DoubleMatrix noiseOf(DoubleMatrix theta) {
        theta.checkColumns(1);
        return new DoubleMatrix(theta.toArray());
    }

    @Override
    public String toString() {
        return "L2";
    }
}
