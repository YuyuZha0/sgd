package org.acc.sgd.regression;

import org.jblas.DoubleMatrix;

/**
 * Created by zhaoyy on 2016/11/23.
 */
public enum LearningRateUpdater {

    CONST {
        @Override
        public double update(double initValue, DoubleMatrix weights) {
            return initValue;
        }
    };

    public abstract double update(double initValue, DoubleMatrix weights);
}
