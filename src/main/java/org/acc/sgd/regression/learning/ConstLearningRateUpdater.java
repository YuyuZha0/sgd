package org.acc.sgd.regression.learning;

import com.google.common.base.Preconditions;
import org.jblas.DoubleMatrix;

/**
 * Created by zhaoyy on 2016/12/26.
 */
public final class ConstLearningRateUpdater implements LearningRateUpdater {

    private final double initValue;

    public ConstLearningRateUpdater(double initValue) {
        Preconditions.checkArgument(initValue > 0, "initValue>0");
        this.initValue = initValue;
    }

    @Override
    public double nextLearningRate(int iteration, DoubleMatrix gradient) {
        return initValue;
    }

    @Override
    public String toString() {
        return "Const[" + initValue + "]";
    }
}
