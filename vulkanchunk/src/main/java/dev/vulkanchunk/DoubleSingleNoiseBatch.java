package dev.vulkanchunk;

import java.util.List;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/** Packed selective double-single noise constants and dynamic coordinates. */
final class DoubleSingleNoiseBatch {
    private static final int HEADER = 14;
    private static final int STRIDE = 74;
    private static final Map<Key, int[]> TEMPLATES = new HashMap<>();

    static synchronized void clearCache() { TEMPLATES.clear(); }

    static synchronized int[] assembleWeird(NormalNoiseSnapshot snapshot, NoiseSampleGrid grid,
                               double[] input, boolean type2) {
        int[] prefix = template(snapshot, true, type2, 0, 0);
        int sampleOffset = prefix.length;
        int[] words = Arrays.copyOf(prefix, sampleOffset + grid.sampleCount() * 5);
        words[0] = grid.sampleCount();
        int[] coords = grid.coordinates();
        for (int i = 0; i < grid.sampleCount(); i++) {
            int at = sampleOffset + i * 5;
            words[at] = coords[i * 3];
            words[at + 1] = coords[i * 3 + 1];
            words[at + 2] = coords[i * 3 + 2];
            putPair(words, at + 3, input[i]);
        }
        return words;
    }

    static synchronized int[] assembleCoordinateNoise(NormalNoiseSnapshot snapshot, NoiseSampleGrid grid,
                                         double[] shifts, double xzScale, double yScale) {
        int[] prefix = template(snapshot, false, false, xzScale, yScale);
        int sampleOffset = prefix.length;
        int[] words = Arrays.copyOf(prefix, sampleOffset + grid.sampleCount() * 9);
        words[0] = grid.sampleCount();
        int[] coords = grid.coordinates();
        for (int i = 0; i < grid.sampleCount(); i++) {
            int at = sampleOffset + i * 9;
            words[at] = coords[i * 3];
            words[at + 1] = coords[i * 3 + 1];
            words[at + 2] = coords[i * 3 + 2];
            for (int axis = 0; axis < 3; axis++) putPair(words, at + 3 + axis * 2, shifts[i * 3 + axis]);
        }
        return words;
    }

    private static int[] template(NormalNoiseSnapshot snapshot, boolean weird, boolean type2,
                                  double xzScale, double yScale) {
        Key key = new Key(snapshot, weird, type2, xzScale, yScale);
        return TEMPLATES.computeIfAbsent(key, ignored -> {
            List<NormalNoiseSnapshot.Octave> first = snapshot.firstOctaves();
            List<NormalNoiseSnapshot.Octave> second = snapshot.secondOctaves();
            int header = weird ? 24 : HEADER;
            int[] words = new int[header + (first.size() + second.size()) * STRIDE];
            words[1] = first.size();
            words[2] = second.size();
            words[3] = words.length;
            words[4] = header;
            words[5] = STRIDE;
            putPair(words, 6, snapshot.valueFactor());
            if (weird) {
                words[8] = type2 ? 2 : 1;
                double[] rarities = type2 ? new double[]{0.5, 0.75, 1, 2, 3}
                        : new double[]{0.75, 1, 1.5, 2, 2};
                for (int i = 0; i < rarities.length; i++) putPair(words, 14 + i * 2, 1.0 / rarities[i]);
            } else {
                putPair(words, 8, xzScale);
                putPair(words, 10, yScale);
            }
            putPair(words, 12, NormalNoiseSnapshot.INPUT_FACTOR);
            int record = header;
            for (NormalNoiseSnapshot.Octave octave : first) {
                putOctave(words, record, octave);
                record += STRIDE;
            }
            for (NormalNoiseSnapshot.Octave octave : second) {
                putOctave(words, record, octave);
                record += STRIDE;
            }
            return words;
        });
    }

    private record Key(NormalNoiseSnapshot snapshot, boolean weird, boolean type2,
                       double xzScale, double yScale) {}

    private static void putPair(int[] words, int offset, double value) {
        float hi = (float) value;
        words[offset] = Float.floatToRawIntBits(hi);
        words[offset + 1] = Float.floatToRawIntBits((float) (value - hi));
    }

    private static void putOctave(int[] words, int offset, NormalNoiseSnapshot.Octave octave) {
        putPair(words, offset, octave.inputFrequency());
        putPair(words, offset + 2, octave.valueScale());
        putPair(words, offset + 4, octave.xOffset());
        putPair(words, offset + 6, octave.yOffset());
        putPair(words, offset + 8, octave.zOffset());
        byte[] permutation = octave.permutation();
        for (int packed = 0; packed < 64; packed++) {
            int base = packed * 4;
            words[offset + 10 + packed] = (permutation[base] & 255)
                    | (permutation[base + 1] & 255) << 8
                    | (permutation[base + 2] & 255) << 16
                    | (permutation[base + 3] & 255) << 24;
        }
    }
}
