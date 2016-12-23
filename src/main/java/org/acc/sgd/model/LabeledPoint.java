package org.acc.sgd.model;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by zhaoyy on 2016/11/11.
 */
public final class LabeledPoint implements Serializable {

    private static final Pattern pattern = Pattern.compile("[^\\s]+");
    private final float label;
    private final Feature[] features;

    private LabeledPoint(float label, Feature[] features) {
        this.label = label;
        this.features = features;
    }

    public static LabeledPoint fromString(String input) {
        Preconditions.checkArgument(!Strings.isNullOrEmpty(input), "null input");

        Matcher matcher = pattern.matcher(input.trim());
        float label = 0;
        if (matcher.find())
            label = Float.parseFloat(matcher.group());

        List<Feature> features = new ArrayList<Feature>();
        while (matcher.find())
            features.add(Feature.fromString(matcher.group()));

        return new LabeledPoint(label, features.toArray(new Feature[features.size()]));
    }

    public static LabeledPoint newInstance(float label, Collection<Feature> c) {
        Preconditions.checkArgument(c != null && c.size() > 0, "empty collection");
        Feature[] temp = c.toArray(new Feature[c.size()]);
        Arrays.sort(temp);
        List<Feature> list = new ArrayList<Feature>();
        int i = 0, j = 0;
        list.add(temp[j]);
        while (++j < temp.length) {
            Feature feature = temp[j];
            boolean repeat = list.get(i).getIndex() == feature.getIndex();
            if (repeat) {
                feature = list.get(i).combine(feature);
                list.set(i, feature);
            } else {
                list.add(feature);
                i++;
            }
        }
        return new LabeledPoint(label, list.toArray(new Feature[list.size()]));
    }

    public static void main(String[] args) {
        List<Feature> features = Arrays.asList(Feature.newInstance(1, 2), Feature.newInstance(2, 1.5f), Feature.newInstance(1, 3));
        System.out.println(LabeledPoint.newInstance(1, features));
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(String.valueOf(label));
        for (Feature feature : features) {
            builder.append(' ');
            builder.append(feature.toString());
        }
        return builder.toString();
    }

    public float label() {
        return label;
    }

    public List<Feature> features() {
        return Arrays.asList(features);
    }

    /**
     * index is 1-based
     *
     * @return
     */
    public int maxIndex() {
        return features[features.length - 1].getIndex();
    }

    /**
     * convert the features to a feature array,if the feature index out of the given bound,it will be ignored.
     *
     * @param bound the size of the array
     * @return
     */
    public double[] featureArray(int bound) {
        double[] d = new double[bound];
        for (Feature feature : features) {
            int index = feature.getIndex();
            if (index > bound)
                continue;
            d[index - 1] = feature.getWeight();
        }
        return d;
    }

    public double[] featureArray(int offset, int bound) {
        Preconditions.checkArgument(offset > 0, "offset must be a positive number");
        double[] d = new double[bound];
        for (Feature feature : features) {
            int index = feature.getIndex() + offset;
            if (index > bound)
                continue;
            d[index - 1] = feature.getWeight();
        }
        return d;
    }

    /**
     * @param vector
     * @return the dot product with the vector,over bound dimensions will be ignored
     */
    public double dot(ImmutableVector vector) {
        Preconditions.checkNotNull(vector);
        double product = 0;
        int len = vector.size();
        for (Feature feature : features) {
            int index = feature.getIndex() - 1;
            if (index >= len)
                continue;
            product += vector.get(index) * feature.getWeight();
        }
        return product;
    }

}
