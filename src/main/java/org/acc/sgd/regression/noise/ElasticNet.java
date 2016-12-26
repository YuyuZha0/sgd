package org.acc.sgd.regression.noise;

import com.google.common.base.Preconditions;
import org.jblas.DoubleMatrix;

/**
 * Created by zhaoyy on 2016/12/26.
 */
public final class ElasticNet implements NoiseUpdater {

    private final double alpha;

    /**
     * @param alpha the weight of L2,alpha∈[0,1]
     */
    public ElasticNet(double alpha) {
        Preconditions.checkArgument(alpha >= 0 && alpha <= 1, "alpha belongs to [0,1]");
        this.alpha = alpha;
    }

    @Override
    public DoubleMatrix noiseOf(DoubleMatrix theta) {
        theta.checkColumns(1);
        double[] temp = theta.toArray();
        for (int i = 0; i < temp.length; i++)
            temp[i] = alpha * temp[i] + (1 - alpha) * Math.signum(temp[i]);
        return new DoubleMatrix(temp);
    }

    @Override
    public String toString() {
        String name = "ElasticNet";
        return name + "[" + alpha + "]";
    }
}
