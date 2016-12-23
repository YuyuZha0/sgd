package org.acc;

import com.google.common.collect.ImmutableList;
import org.acc.sgd.model.Feature;
import org.acc.sgd.model.LabeledPoint;
import org.acc.sgd.model.ResultModel;
import org.acc.sgd.regression.*;

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
                .learningRate(0.1)
                .iterations(1000)
                .batchSize(100)
                .epsilon(0.01)
                .learningRateUpdater(LearningRateUpdater.CONST)
                .noiseUpdater(NoiseUpdater.ELASTIC_NET, 0.8)
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
        double x = (Math.random() - 0.5) * 20;
        double y = (Math.random() - 0.5) * 20;
        double label = 10 * x - 5 * y + 7 > 0 ? 1 : 0;
        return LabeledPoint.newInstance((float) label, Arrays.asList(Feature.of(1, (float) x), Feature.of(2, (float) y)));
    }
}
