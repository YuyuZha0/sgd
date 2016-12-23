package org.acc.sgd.regression;

import org.jblas.DoubleMatrix;

/**
 * Created by zhaoyy on 2016/12/22.
 */
public enum Hypothesis {

    LINEAR {
        @Override
        public DoubleMatrix apply(DoubleMatrix x, DoubleMatrix theta) {
            return x.mmul(theta);
        }
    },
    LOGISTIC {
        @Override
        public DoubleMatrix apply(DoubleMatrix x, DoubleMatrix theta) {
            theta.checkColumns(1);
            return sigmoid(x.mmul(theta));
        }
    };

    private static double sigmoid(double d) {
        return 1.0 / (1 + Math.exp(-d));
    }

    private static DoubleMatrix sigmoid(DoubleMatrix m) {
        m.checkColumns(1);
        double[] temp = m.toArray();
        for (int i = 0; i < temp.length; i++)
            temp[i] = sigmoid(temp[i]);
        return new DoubleMatrix(temp);
    }

    public static void main(String[] args) {
        DoubleMatrix x = new DoubleMatrix(new double[][]{{1, 2}});
        System.out.println(x);
        DoubleMatrix theta = new DoubleMatrix(new double[]{1, 1});
        System.out.println(theta);
        System.out.println(LOGISTIC.apply(x, theta));
    }

    public abstract DoubleMatrix apply(DoubleMatrix x, DoubleMatrix theta);
}
