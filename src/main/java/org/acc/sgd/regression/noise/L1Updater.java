package org.acc.sgd.regression.noise;

import org.jblas.DoubleMatrix;

/**
 * Created by zhaoyy on 2016/12/26.
 */
public final class L1Updater implements NoiseUpdater {
    @Override
    public DoubleMatrix noiseOf(DoubleMatrix theta) {
        theta.checkColumns(1);
        double[] temp = theta.toArray();
        for (int i = 0; i < temp.length; i++)
            temp[i] = Math.signum(temp[i]);
        return new DoubleMatrix(temp);
    }

    @Override
    public String toString() {
        return "L1";
    }
}
