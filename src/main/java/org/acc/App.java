package org.acc;

import com.google.common.collect.ImmutableList;
import org.acc.sgd.model.Feature;
import org.acc.sgd.model.LabeledPoint;
import org.acc.sgd.model.ResultModel;
import org.acc.sgd.regression.GradientDescent;
import org.acc.sgd.regression.Hypothesis;
import org.acc.sgd.regression.Sampler;
import org.acc.sgd.regression.learning.ConstLearningRateUpdater;
import org.acc.sgd.regression.noise.ConstNoiseUpdater;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Hello world!
 */
public class App {

    public static void main(String[] args) {
        ImmutableList<LabeledPoint> samples = generateSamples(1000);

        GradientDescent gradientDescent = GradientDescent
                .newGradientDescent()
                .iterations(500)
                .batchSize(100)
                .epsilon(0.01)
                .learningRateUpdater(new ConstLearningRateUpdater(0.01))
                .noiseUpdater(new ConstNoiseUpdater())
                .sampler(Sampler.RANDOM)
                .hypothesis(Hypothesis.LOGISTIC)
                .build(samples);

        ResultModel model = gradientDescent.run();

        System.out.println(model);
    }

    private static ImmutableList<LabeledPoint> generateSamples(int size) {
        List<LabeledPoint> points = new ArrayList<LabeledPoint>();
        for (int i = 0; i < size; i++) {
            LabeledPoint point = randomLabel();
            points.add(point);
        }
        return ImmutableList.copyOf(points);
    }

    private static LabeledPoint randomLabel() {
        double x = Math.random();
        double y = Math.random();
        double label = 1 * x - 2 * y > 0 ? 1 : 0;
        return LabeledPoint.newInstance((float) label, Arrays.asList(Feature.of(1, (float) x), Feature.of(2, (float) y)));
    }
}
