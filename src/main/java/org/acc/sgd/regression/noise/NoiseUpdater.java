package org.acc.sgd.regression.noise;

import org.jblas.DoubleMatrix;

/**
 * Created by zhaoyy on 2016/12/26.
 */
public interface NoiseUpdater {

    DoubleMatrix noiseOf(DoubleMatrix theta);
}
