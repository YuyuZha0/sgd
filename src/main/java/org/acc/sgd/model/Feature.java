package org.acc.sgd.model;

import com.google.common.base.Preconditions;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hashing;
import com.google.common.primitives.Floats;
import com.google.common.primitives.Ints;

import java.io.Serializable;

/**
 * Created by zhaoyy on 2016/11/11.
 */
public final class Feature implements Comparable<Feature>, Serializable {

    public static final Feature UNDEFINED = new Feature(0, 0);
    private static final HashFunction hashFunction = Hashing.murmur3_32();

    private final int index;
    private final float weight;


    private Feature(int index, float weight) {
        this.index = index;
        this.weight = weight;
    }

    public static Feature newInstance(int index, float weight) {
        Preconditions.checkArgument(index > 0, "index is 1-based");
        return new Feature(index, weight);
    }

    public static Feature fromString(String input) {
        int split = input.indexOf(":");
        int index = Ints.tryParse(input.substring(0, split));
        float weight = Floats.tryParse(input.substring(split + 1, input.length()));
        return new Feature(index, weight);
    }

    public static void main(String[] args) {
        System.out.println(Feature.UNDEFINED);
    }

    public int getIndex() {
        return index;
    }

    public float getWeight() {
        return weight;
    }

    public Feature combine(Feature feature) {
        if (feature == null || feature.index == 0)
            return this;
        Preconditions.checkArgument(feature.index == index, "index mismatch");
        return new Feature(index, weight + feature.weight);
    }

    @Override
    public int hashCode() {
        int result = hashFunction
                .newHasher()
                .putInt(index)
                .putDouble(weight)
                .hash()
                .asInt();
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null)
            return false;
        if (!(obj instanceof Feature))
            return false;
        Feature feature = (Feature) obj;
        return index == feature.index && weight == feature.weight;
    }

    @Override
    public String toString() {
        return new StringBuilder()
                .append(index)
                .append(':')
                .append(weight)
                .toString();
    }

    @Override
    public int compareTo(Feature o) {
        if (index != o.index)
            return Integer.compare(index, o.index);
        return Float.compare(weight, o.weight);
    }
}
