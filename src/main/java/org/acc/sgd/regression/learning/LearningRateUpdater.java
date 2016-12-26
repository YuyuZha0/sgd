package org.acc.sgd.regression.learning;

import org.jblas.DoubleMatrix;

/**
 * Created by zhaoyy on 2016/12/26.
 */
public interface LearningRateUpdater {

    /**
     * @param iteration iteration is 0-based
     * @param gradient
     * @return
     */
    double nextLearningRate(int iteration, DoubleMatrix gradient);
}
