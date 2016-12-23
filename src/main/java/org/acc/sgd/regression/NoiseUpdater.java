package org.acc.sgd.regression;

import org.jblas.DoubleMatrix;

/**
 * Created by zhaoyy on 2016/11/23.
 */
public enum NoiseUpdater {

    NONE {
        @Override
        public DoubleMatrix update(DoubleMatrix theta, double alpha) {
            theta.checkColumns(1);
            return new DoubleMatrix(theta.getRows(), 1);
        }
    },
    ELASTIC_NET {
        @Override
        public DoubleMatrix update(DoubleMatrix theta, double alpha) {
            theta.checkColumns(1);
            double[] temp = theta.toArray();
            for (int i = 0; i < temp.length; i++)
                temp[i] = alpha * temp[i] + (1 - alpha) * Math.signum(temp[i]);
            return new DoubleMatrix(temp);
        }
    };

    public abstract DoubleMatrix update(DoubleMatrix theta, double alpha);
}
