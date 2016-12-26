package org.acc.sgd.regression.learning;

import com.google.common.base.Preconditions;
import org.jblas.DoubleMatrix;

/**
 * Created by zhaoyy on 2016/12/26.
 */
public final class AdaDelta implements LearningRateUpdater {

    private final double initValue;
    private final double rho;
    private final double epsilon;

    private volatile double n = 0;

    public AdaDelta(double initValue, double rho, double epsilon) {
        Preconditions.checkArgument(initValue > 0, "initValue>0");
        Preconditions.checkArgument(rho > 0 && rho < 1, "rho∈(0,1)");
        Preconditions.checkArgument(epsilon > 0, "epsilon>0");
        this.initValue = initValue;
        this.rho = rho;
        this.epsilon = epsilon;
    }

    public AdaDelta(double initValue) {
        Preconditions.checkArgument(initValue > 0, "initValue>0");
        this.initValue = initValue;
        this.rho = 0.95;
        this.epsilon = 1e-6;
    }

    @Override
    public double nextLearningRate(int iteration, DoubleMatrix gradient) {
        gradient.checkColumns(1);
        n = rho * n + (1 - rho) * gradient.dot(gradient);
        return initValue / Math.sqrt(n + epsilon);
    }

    @Override
    public String toString() {
        String name = "AdaDelta";
        return name + "[" + rho + "," + epsilon + "]";
    }
}
