package org.acc.sgd.regression;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by zhaoyy on 2016/11/23.
 */
public enum Sampler {

    RANDOM {
        @Override
        public int[] sample(int batch, int bound) {
            if (batch >= bound) {
                int[] indices = new int[bound];
                for (int i = 0; i < indices.length; i++)
                    indices[i] = i;
                return indices;
            }
            int[] indices = new int[batch];
            randomSample(batch, bound, indices);
            for (int i = 0; i < indices.length; i++)
                indices[i]--;
            return indices;
        }
    };


    /**
     * RANDOM-SAMPLE(m,n)
     * if m == 0
     * return ∅
     * else
     * S = RANDOM-SAMPLE(m-1, n-1)
     * i = RANDOM(1,n)
     * if i ∈ S
     * S = S ∪ {n}
     * else
     * S = S ∪ {i}
     * return S
     *
     * @param m
     * @param n
     * @param a
     */
    private static void randomSample(int m, int n, int[] a) {
        if (m == 0)
            return;
        randomSample(m - 1, n - 1, a);
        int i = ThreadLocalRandom.current().nextInt(1, n);
        if (contains(a, m - 1, i))
            a[m - 1] = n;
        else a[m - 1] = i;

    }

    private static boolean contains(int[] a, int m, int n) {
        for (int i = 0; i < m; i++)
            if (a[i] == n)
                return true;
        return false;
    }

    /**
     * @param batch
     * @param bound(excluded)
     * @return
     */
    public abstract int[] sample(int batch, int bound);
}
