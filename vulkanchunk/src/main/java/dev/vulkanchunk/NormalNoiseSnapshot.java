package dev.vulkanchunk;

import dev.vulkanchunk.mixin.ImprovedNoiseAccessor;
import dev.vulkanchunk.mixin.NormalNoiseAccessor;
import dev.vulkanchunk.mixin.PerlinNoiseAccessor;
import it.unimi.dsi.fastutil.doubles.DoubleList;
import net.minecraft.world.level.levelgen.synth.ImprovedNoise;
import net.minecraft.world.level.levelgen.synth.NormalNoise;
import net.minecraft.world.level.levelgen.synth.PerlinNoise;

import java.util.ArrayList;
import java.util.List;
import java.util.IdentityHashMap;
import java.util.Map;

/** Immutable numeric form of Minecraft's two-Perlin NormalNoise implementation. */
public final class NormalNoiseSnapshot {
    public static final double INPUT_FACTOR = 1.0181268882175227;
    private static final Map<NormalNoise, NormalNoiseSnapshot> CACHE = new IdentityHashMap<>();
    private static final double WRAP = 33_554_432.0;
    private static final int[][] GRADIENTS = {
            {1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0},
            {1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1},
            {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1},
            {1, 1, 0}, {0, -1, 1}, {-1, 1, 0}, {0, -1, -1}
    };

    private final List<Octave> first;
    private final List<Octave> second;
    private final double valueFactor;

    private NormalNoiseSnapshot(List<Octave> first, List<Octave> second, double valueFactor) {
        this.first = List.copyOf(first);
        this.second = List.copyOf(second);
        this.valueFactor = valueFactor;
    }

    public static synchronized NormalNoiseSnapshot capture(NormalNoise noise) {
        return CACHE.computeIfAbsent(noise, NormalNoiseSnapshot::captureFresh);
    }

    public static synchronized void clearCache() { CACHE.clear(); }

    private static NormalNoiseSnapshot captureFresh(NormalNoise noise) {
        NormalNoiseAccessor normal = (NormalNoiseAccessor) (Object) noise;
        return new NormalNoiseSnapshot(capturePerlin(normal.vulkanchunk$first()),
                capturePerlin(normal.vulkanchunk$second()), normal.vulkanchunk$valueFactor());
    }

    private static List<Octave> capturePerlin(PerlinNoise noise) {
        PerlinNoiseAccessor perlin = (PerlinNoiseAccessor) (Object) noise;
        ImprovedNoise[] levels = perlin.vulkanchunk$noiseLevels();
        DoubleList amplitudes = perlin.vulkanchunk$amplitudes();
        List<Octave> result = new ArrayList<>();
        double inputFrequency = perlin.vulkanchunk$lowestFreqInputFactor();
        double valueScale = perlin.vulkanchunk$lowestFreqValueFactor();
        for (int index = 0; index < levels.length; index++) {
            ImprovedNoise level = levels[index];
            if (level != null) {
                byte[] permutation = ((ImprovedNoiseAccessor) (Object) level).vulkanchunk$permutation().clone();
                result.add(new Octave(inputFrequency, amplitudes.getDouble(index) * valueScale,
                        level.xo, level.yo, level.zo, permutation));
            }
            inputFrequency *= 2.0;
            valueScale /= 2.0;
        }
        return result;
    }

    /** Exact Java-double reconstruction used to validate the exported constants. */
    public double compute(double x, double y, double z) {
        double shiftedX = x * INPUT_FACTOR;
        double shiftedY = y * INPUT_FACTOR;
        double shiftedZ = z * INPUT_FACTOR;
        return (samplePerlin(first, x, y, z) + samplePerlin(second, shiftedX, shiftedY, shiftedZ)) * valueFactor;
    }

    /** Reference lattice fingerprint for the isolated shader precision check. */
    public int latticeHash(double x, double y, double z) {
        int hash = latticeHash(first, x, y, z, 0x811c9dc5);
        return latticeHash(second, x * INPUT_FACTOR, y * INPUT_FACTOR, z * INPUT_FACTOR, hash);
    }

    private static int latticeHash(List<Octave> octaves, double x, double y, double z, int hash) {
        for (Octave octave : octaves) {
            int ix = floor(wrap(x * octave.inputFrequency) + octave.xOffset);
            int iy = floor(wrap(y * octave.inputFrequency) + octave.yOffset);
            int iz = floor(wrap(z * octave.inputFrequency) + octave.zOffset);
            hash = (hash ^ ix) * 16777619;
            hash = (hash ^ iy) * 16777619;
            hash = (hash ^ iz) * 16777619;
        }
        return hash;
    }

    private static double samplePerlin(List<Octave> octaves, double x, double y, double z) {
        double result = 0.0;
        for (Octave octave : octaves) {
            result += octave.valueScale * improvedNoise(octave,
                    wrap(x * octave.inputFrequency), wrap(y * octave.inputFrequency), wrap(z * octave.inputFrequency));
        }
        return result;
    }

    private static double improvedNoise(Octave octave, double x, double y, double z) {
        double offsetX = x + octave.xOffset;
        double offsetY = y + octave.yOffset;
        double offsetZ = z + octave.zOffset;
        int floorX = floor(offsetX);
        int floorY = floor(offsetY);
        int floorZ = floor(offsetZ);
        double localX = offsetX - floorX;
        double localY = offsetY - floorY;
        double localZ = offsetZ - floorZ;

        int px0 = permutation(octave, floorX);
        int px1 = permutation(octave, floorX + 1);
        int pxy00 = permutation(octave, px0 + floorY);
        int pxy01 = permutation(octave, px0 + floorY + 1);
        int pxy10 = permutation(octave, px1 + floorY);
        int pxy11 = permutation(octave, px1 + floorY + 1);

        double n000 = gradient(permutation(octave, pxy00 + floorZ), localX, localY, localZ);
        double n100 = gradient(permutation(octave, pxy10 + floorZ), localX - 1.0, localY, localZ);
        double n010 = gradient(permutation(octave, pxy01 + floorZ), localX, localY - 1.0, localZ);
        double n110 = gradient(permutation(octave, pxy11 + floorZ), localX - 1.0, localY - 1.0, localZ);
        double n001 = gradient(permutation(octave, pxy00 + floorZ + 1), localX, localY, localZ - 1.0);
        double n101 = gradient(permutation(octave, pxy10 + floorZ + 1), localX - 1.0, localY, localZ - 1.0);
        double n011 = gradient(permutation(octave, pxy01 + floorZ + 1), localX, localY - 1.0, localZ - 1.0);
        double n111 = gradient(permutation(octave, pxy11 + floorZ + 1), localX - 1.0, localY - 1.0, localZ - 1.0);
        return lerp3(smoothstep(localX), smoothstep(localY), smoothstep(localZ),
                n000, n100, n010, n110, n001, n101, n011, n111);
    }

    private static int permutation(Octave octave, int index) {
        return octave.permutation[index & 0xFF] & 0xFF;
    }

    private static double gradient(int hash, double x, double y, double z) {
        int[] gradient = GRADIENTS[hash & 15];
        return gradient[0] * x + gradient[1] * y + gradient[2] * z;
    }

    private static int floor(double value) {
        int integer = (int) value;
        return value < integer ? integer - 1 : integer;
    }

    private static long longFloor(double value) {
        long integer = (long) value;
        return value < integer ? integer - 1L : integer;
    }

    private static double wrap(double value) {
        return value - longFloor(value / WRAP + 0.5) * WRAP;
    }

    private static double smoothstep(double value) {
        return value * value * value * (value * (value * 6.0 - 15.0) + 10.0);
    }

    private static double lerp(double delta, double start, double end) {
        return start + delta * (end - start);
    }

    private static double lerp2(double x, double y, double v00, double v10, double v01, double v11) {
        return lerp(y, lerp(x, v00, v10), lerp(x, v01, v11));
    }

    private static double lerp3(double x, double y, double z,
                                double v000, double v100, double v010, double v110,
                                double v001, double v101, double v011, double v111) {
        return lerp(z, lerp2(x, y, v000, v100, v010, v110),
                lerp2(x, y, v001, v101, v011, v111));
    }

    public List<Octave> firstOctaves() { return first; }
    public List<Octave> secondOctaves() { return second; }
    public double valueFactor() { return valueFactor; }
    public int activeOctaves() { return first.size() + second.size(); }

    public record Octave(double inputFrequency, double valueScale,
                         double xOffset, double yOffset, double zOffset,
                         byte[] permutation) {
    }
}
