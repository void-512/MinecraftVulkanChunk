package dev.vulkanchunk;

import dev.vulkanchunk.mixin.ImprovedNoiseAccessor;
import net.minecraft.world.level.levelgen.synth.BlendedNoise;
import net.minecraft.world.level.levelgen.synth.ImprovedNoise;
import net.minecraft.world.level.levelgen.synth.PerlinNoise;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.Map;

final class BlendedNoiseBatch {
    private static final int HEADER = 16, STRIDE = 74;
    private static final Map<BlendedNoise, int[]> GPU_TEMPLATES = new IdentityHashMap<>();
    private static final NoiseSampleGrid EMPTY_GRID = new NoiseSampleGrid(new int[0], 0, 0,
            0, 0, 0, 0, 0);

    static synchronized void clearCache() { GPU_TEMPLATES.clear(); }

    static int[] assemble(BlendedNoise noise, NoiseSampleGrid grid) {
        return assemble(noise, grid, false);
    }

    static synchronized int[] assembleGpuCoordinates(BlendedNoise noise, NoiseSampleGrid grid) {
        int[] template = GPU_TEMPLATES.computeIfAbsent(noise, key -> assemble(key, EMPTY_GRID, true));
        int[] words = Arrays.copyOf(template, template.length + grid.coordinates().length);
        words[0] = grid.sampleCount();
        System.arraycopy(grid.coordinates(), 0, words, template.length, grid.coordinates().length);
        return words;
    }

    private static int[] assemble(BlendedNoise noise, NoiseSampleGrid grid, boolean gpuCoordinates) {
        PerlinNoise min = (PerlinNoise)field(noise, "minLimitNoise");
        PerlinNoise max = (PerlinNoise)field(noise, "maxLimitNoise");
        PerlinNoise main = (PerlinNoise)field(noise, "mainNoise");
        double xz = (double)field(noise, "xzMultiplier");
        double y = (double)field(noise, "yMultiplier");
        double xzFactor = (double)field(noise, "xzFactor");
        double yFactor = (double)field(noise, "yFactor");
        double smear = (double)field(noise, "smearScaleMultiplier");
        int header = gpuCoordinates ? 20 : HEADER;
        int sampleOffset = header + 40 * STRIDE;
        int[] words = new int[sampleOffset + grid.sampleCount() * (gpuCoordinates ? 3 : 12)];
        words[0] = grid.sampleCount();
        words[1] = 8;
        words[2] = 16;
        words[3] = sampleOffset;
        words[4] = header;
        words[5] = STRIDE;
        pair(words, 6, y * smear);
        pair(words, 8, y * smear / yFactor);
        if (gpuCoordinates) {
            pair(words, 10, xz);
            pair(words, 12, y);
            pair(words, 14, 1.0 / xzFactor);
            pair(words, 16, 1.0 / yFactor);
        }
        for (int i = 0; i < 8; i++) octave(words, header + i * STRIDE,
                main.getOctaveNoise(i), 1.0 / ((y * smear / yFactor) * Math.scalb(1.0, -i)));
        for (int i = 0; i < 16; i++) octave(words, header + (8 + i) * STRIDE,
                min.getOctaveNoise(i), 1.0 / ((y * smear) * Math.scalb(1.0, -i)));
        for (int i = 0; i < 16; i++) octave(words, header + (24 + i) * STRIDE,
                max.getOctaveNoise(i), 1.0 / ((y * smear) * Math.scalb(1.0, -i)));
        int[] coords = grid.coordinates();
        for (int i = 0; i < grid.sampleCount(); i++) {
            if (gpuCoordinates) {
                int at = sampleOffset + i * 3;
                words[at] = coords[i * 3];
                words[at + 1] = coords[i * 3 + 1];
                words[at + 2] = coords[i * 3 + 2];
                continue;
            }
            double d0 = (double)coords[i * 3] * xz;
            double d1 = (double)coords[i * 3 + 1] * y;
            double d2 = (double)coords[i * 3 + 2] * xz;
            int at = sampleOffset + i * 12;
            pair(words, at, d0);
            pair(words, at + 2, d1);
            pair(words, at + 4, d2);
            pair(words, at + 6, d0 / xzFactor);
            pair(words, at + 8, d1 / yFactor);
            pair(words, at + 10, d2 / xzFactor);
        }
        return words;
    }

    private static void octave(int[] words, int offset, ImprovedNoise noise, double inverseSmear) {
        if (noise == null) return;
        words[offset] = 1;
        pair(words, offset + 2, inverseSmear);
        pair(words, offset + 4, noise.xo);
        pair(words, offset + 6, noise.yo);
        pair(words, offset + 8, noise.zo);
        byte[] permutation = ((ImprovedNoiseAccessor)(Object)noise).vulkanchunk$permutation();
        for (int i = 0; i < 64; i++) {
            int at = i * 4;
            words[offset + 10 + i] = (permutation[at] & 255)
                    | (permutation[at + 1] & 255) << 8
                    | (permutation[at + 2] & 255) << 16
                    | (permutation[at + 3] & 255) << 24;
        }
    }

    private static Object field(BlendedNoise noise, String name) {
        try {
            Field field = BlendedNoise.class.getDeclaredField(name);
            field.setAccessible(true);
            return field.get(noise);
        } catch (ReflectiveOperationException failure) {
            throw new IllegalStateException(failure);
        }
    }

    private static void pair(int[] words, int at, double value) {
        float hi = (float)value;
        words[at] = Float.floatToRawIntBits(hi);
        words[at + 1] = Float.floatToRawIntBits((float)(value - hi));
    }
}
