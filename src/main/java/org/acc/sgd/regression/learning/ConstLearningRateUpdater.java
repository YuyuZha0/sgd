package org.acc.sgd.regression.learning;

import org.jblas.DoubleMatrix;

/**
 * Created by zhaoyy on 2016/12/26.
 */
public final class ConstLearningRateUpdater implements LearningRateUpdater {
    @Override
    public double nextRate(double initValue, DoubleMatrix theta) {
        return initValue;
    }

    @Override
    public String toString() {
        String name = ConstLearningRateUpdater.class.getName();
        return name;
    }
}
