package dev.vulkanchunk;

import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.synth.SimplexNoise;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.Map;

/** Immutable simplex permutation plus the End's two-dimensional corner coordinates. */
final class EndIslandBatch {
    private static final Map<DensityFunction, int[]> TEMPLATES = new IdentityHashMap<>();

    private EndIslandBatch() {}

    static synchronized int[] assemble(DensityFunction function, NoiseSampleGrid grid) {
        int vertical = grid.height() / grid.cellHeight() + 1;
        int horizontalSamples = grid.sampleCount() / vertical;
        int[] template = TEMPLATES.computeIfAbsent(function, EndIslandBatch::template);
        int[] words = Arrays.copyOf(template, template.length + horizontalSamples * 2);
        words[0] = horizontalSamples;
        words[1] = vertical;
        words[2] = template.length;
        int[] xyz = grid.coordinates();
        for (int i = 0; i < horizontalSamples; i++) {
            int at = i * vertical * 3;
            words[template.length + i * 2] = xyz[at];
            words[template.length + i * 2 + 1] = xyz[at + 2];
        }
        return words;
    }

    static synchronized void clearCache() { TEMPLATES.clear(); }

    private static int[] template(DensityFunction function) {
        try {
            Field noiseField = function.getClass().getDeclaredField("islandNoise");
            noiseField.setAccessible(true);
            SimplexNoise noise = (SimplexNoise)noiseField.get(function);
            Field permutationField = SimplexNoise.class.getDeclaredField("p");
            permutationField.setAccessible(true);
            int[] permutation = (int[])permutationField.get(noise);
            int[] words = new int[74];
            words[3] = 10;
            pair(words, 4, (Math.sqrt(3.0) - 1.0) * 0.5);
            pair(words, 6, (3.0 - Math.sqrt(3.0)) / 6.0);
            pair(words, 8, (double)-0.9f);
            for (int i = 0; i < 256; i++) words[10 + i / 4] |= permutation[i] << ((i & 3) * 8);
            return words;
        } catch (ReflectiveOperationException failure) {
            throw new UnsupportedOperationException("Cannot capture exact End island simplex permutation", failure);
        }
    }

    private static void pair(int[] words, int index, double value) {
        float hi = (float)value;
        words[index] = Float.floatToRawIntBits(hi);
        words[index + 1] = Float.floatToRawIntBits((float)(value - hi));
    }
}
