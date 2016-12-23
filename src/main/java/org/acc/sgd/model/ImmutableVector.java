package org.acc.sgd.model;

import com.google.common.base.Preconditions;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Created by zhaoyy on 2016/12/12.
 */
public final class ImmutableVector implements Serializable {

    private final double[] value;


    private ImmutableVector(double[] value) {
        this.value = value;
    }

    public static ImmutableVector valueOf(double[] doubles) {
        Preconditions.checkNotNull(doubles);
        return new ImmutableVector(Arrays.copyOf(doubles, doubles.length));
    }

    public double get(int index) {
        Preconditions.checkPositionIndex(index, value.length);
        return value[index];
    }

    public int dimension() {
        return value.length;
    }

    public double[] toArray() {
        return Arrays.copyOf(value, value.length);
    }

    public int size() {
        return value.length;
    }

    @Override
    public String toString() {
        return Arrays.toString(value);
    }
}
