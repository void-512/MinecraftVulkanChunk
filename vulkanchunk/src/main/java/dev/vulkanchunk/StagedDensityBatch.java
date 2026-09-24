package dev.vulkanchunk;

import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.DensityFunctions;
import net.minecraft.world.level.levelgen.synth.BlendedNoise;
import net.minecraft.world.level.levelgen.synth.NormalNoise;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

record StagedDensityBatch(int[] words, List<Stage> stages, int blocks, int dispatches,
                          BitSet scratch, List<EndIslandCheck> endIslandChecks) {
    private static final AtomicInteger CHECK_END_ISLANDS = new AtomicInteger(
            Integer.getInteger("vulkanchunk.checkEndIslands", 0));
    private static final Map<DensityFunction, DensityGraphProgram> STATIC_GRAPHS = new IdentityHashMap<>();
    private static final Map<DensityFunctions.Spline, SplineProgram> STATIC_SPLINES = new IdentityHashMap<>();

    static void clearStaticPrograms() {
        STATIC_GRAPHS.clear();
        STATIC_SPLINES.clear();
    }

    private static DensityGraphProgram graphFor(DensityFunction function) {
        return STATIC_GRAPHS.computeIfAbsent(function, key -> new DensityGraphProgram(key, true));
    }

    private static SplineProgram splineFor(DensityFunctions.Spline spline) {
        return STATIC_SPLINES.computeIfAbsent(spline, SplineProgram::new);
    }
    record Stage(int kind, int table, int jobs, int samples) {}
    record EndIslandCheck(int offset, double expected, int x, int z) {}
    record Case(DensityGraphProgram outer, DensityGraphProgram[] inner, double[] beard,
                int cellWidth, int cellHeight, int height, net.minecraft.world.level.ChunkPos pos, int minY) {}
    private record Source(int offset, int stride) {}
    private record Job(int input, int output, int type, int samples) {}

    static StagedDensityBatch assembleProduction(List<Case> cases) {
        DensityBatchLayout base = DensityBatchLayout.assemble(cases);
        Builder builder = new Builder(base.words());
        builder.scratch.set(base.cornerOffsets()[0],
                base.cornerOffsets()[0] + cases.getFirst().inner().length * base.cornerSamples() * 3);
        for (int chunk = 0; chunk < cases.size(); chunk++) builder.chunk(cases.get(chunk), chunk, base);
        List<Stage> stages = builder.stages();
        int dispatches = stages.size() + 1;
        return new StagedDensityBatch(builder.finish(), stages, base.blocks(), dispatches,
                builder.scratch, List.copyOf(builder.endIslandChecks));
    }

    private static final class Builder {
        final int[] words = new int[VulkanComputeBackend.MAX_INPUT_INTS];
        final BitSet scratch = new BitSet();
        final List<EndIslandCheck> endIslandChecks = new ArrayList<>();
        final List<Job>[] stageJobs;
        int cursor;

        @SuppressWarnings("unchecked")
        Builder(int[] base) {
            System.arraycopy(base, 0, words, 0, base.length);
            cursor = base.length;
            stageJobs = new List[7];
            for (int i = 0; i < stageJobs.length; i++) stageJobs[i] = new ArrayList<>();
        }

        int put(int[] data) {
            int at = cursor;
            System.arraycopy(data, 0, words, at, data.length);
            cursor += data.length;
            return at;
        }

        int reserve(int count) {
            int at = cursor;
            cursor += count;
            scratch.set(at, cursor);
            return at;
        }

        Source noise(int stage, int[] input, int type, int count) {
            int inputAt = put(input), outputAt = reserve(count * 3);
            stageJobs[stage].add(new Job(inputAt, outputAt, type, count));
            return new Source(outputAt, 3);
        }

        Source graph(int stage, DensityGraphProgram program, double[][] samples,
                     Map<Integer, Source> sources, int outputAt) {
            int[] batch = program.batch(samples, 0, samples.length);
            int[] packed = Arrays.copyOf(batch, batch.length + program.leafCount() * 2);
            for (Map.Entry<Integer, Source> entry : sources.entrySet()) {
                int at = batch.length + entry.getKey() * 2;
                packed[at] = entry.getValue().offset() + 1;
                packed[at + 1] = entry.getValue().stride();
            }
            int inputAt = put(packed);
            if (outputAt < 0) outputAt = reserve(samples.length * 3);
            stageJobs[stage].add(new Job(inputAt, outputAt, 3, samples.length));
            return new Source(outputAt, 3);
        }

        Source spline(SplineProgram program, int count, Map<Integer, Source> sources) {
            int[] batch = program.batch(new float[count][program.coordinateCount()]);
            int[] packed = Arrays.copyOf(batch, batch.length + program.coordinateCount() * 2);
            for (Map.Entry<Integer, Source> entry : sources.entrySet()) {
                int at = batch.length + entry.getKey() * 2;
                packed[at] = entry.getValue().offset() + 1;
                packed[at + 1] = entry.getValue().stride();
            }
            int inputAt = put(packed), outputAt = reserve(count * 3);
            stageJobs[5].add(new Job(inputAt, outputAt, 0, count));
            return new Source(outputAt, 3);
        }

        void chunk(Case candidate, int chunk, DensityBatchLayout base) {
            NoiseSampleGrid grid = NoiseSampleGrid.create(candidate.pos().x, candidate.pos().z, 1,
                    candidate.minY(), candidate.height(), candidate.cellWidth(), candidate.cellHeight());
            int count = grid.sampleCount();
            Map<DensityFunction, Source> primitives = new HashMap<>();
            Map<DensityFunction, Source> coordinates = new HashMap<>();
            Map<DensityFunction, Source> splines = new HashMap<>();
            Map<DensityFunction, Source> rarity = new HashMap<>();
            for (DensityGraphProgram inner : candidate.inner()) {
                for (DensityFunction leaf : inner.leaves()) {
                    if (leaf == null) continue;
                    String type = leaf.getClass().getSimpleName();
                    if (type.equals("Noise") || type.equals("ShiftedNoise") || leaf instanceof BlendedNoise)
                        primitive(leaf, grid, primitives);
                    else if (type.equals("WeirdScaledSampler")) {
                        DensityFunction input = (DensityFunction)DensityGraphProgram.part(leaf, "input");
                        DensityGraphProgram program = graphFor(input);
                        for (DensityFunction sourceLeaf : program.leaves())
                            if (sourceLeaf != null) primitive(sourceLeaf, grid, primitives);
                    } else if (leaf instanceof DensityFunctions.Spline spline) {
                        SplineProgram program = splineFor(spline);
                        for (int i = 0; i < program.coordinateCount(); i++) {
                            DensityFunction function = program.coordinateFunction(i);
                            DensityGraphProgram coordinate = graphFor(function);
                            for (DensityFunction sourceLeaf : coordinate.leaves())
                                if (sourceLeaf != null) primitive(sourceLeaf, grid, primitives);
                        }
                    } else if (type.equals("EndIslandDensityFunction")) {
                        primitives.put(leaf, endIslands(leaf, grid));
                    } else throw new UnsupportedOperationException("Unsupported density leaf: " + type);
                }
            }
            for (DensityGraphProgram inner : candidate.inner()) {
                for (DensityFunction leaf : inner.leaves()) {
                    if (leaf == null || !leaf.getClass().getSimpleName().equals("WeirdScaledSampler")) continue;
                    DensityFunction input = (DensityFunction)DensityGraphProgram.part(leaf, "input");
                    Source raw = rarity.computeIfAbsent(input, key -> {
                        DensityGraphProgram program = graphFor(key);
                        return graph(2, program, inlineSamples(program, grid), leafSources(program, primitives), -1);
                    });
                    DensityFunction.NoiseHolder holder = (DensityFunction.NoiseHolder)DensityGraphProgram.part(leaf, "noise");
                    boolean type2 = DensityGraphProgram.part(leaf, "rarityValueMapper").toString().equals("TYPE2");
                    int[] original = DoubleSingleNoiseBatch.assembleWeird(
                            NormalNoiseSnapshot.capture(holder.noise()), grid, new double[count], type2);
                    int[] packed = Arrays.copyOf(original, original.length + 1);
                    packed[original.length] = raw.offset() + 1;
                    primitives.put(leaf, noise(3, packed, 8, count));
                }
            }
            for (DensityGraphProgram inner : candidate.inner()) {
                for (DensityFunction leaf : inner.leaves()) {
                    if (!(leaf instanceof DensityFunctions.Spline spline) || splines.containsKey(leaf)) continue;
                    SplineProgram program = splineFor(spline);
                    Map<Integer, Source> sourceSlots = new HashMap<>();
                    for (int i = 0; i < program.coordinateCount(); i++) {
                        DensityFunction function = program.coordinateFunction(i);
                        Source source = coordinates.computeIfAbsent(function, key -> {
                            DensityGraphProgram coordinate = graphFor(key);
                            return graph(4, coordinate, inlineSamples(coordinate, grid),
                                    leafSources(coordinate, primitives), -1);
                        });
                        sourceSlots.put(i, source);
                    }
                    splines.put(leaf, spline(program, count, sourceSlots));
                }
            }
            for (int leaf = 0; leaf < candidate.inner().length; leaf++) {
                DensityGraphProgram program = candidate.inner()[leaf];
                Map<Integer, Source> sourceSlots = new HashMap<>();
                for (int i = 0; i < program.leafCount(); i++) {
                    DensityFunction function = program.leaves().get(i);
                    if (function == null) continue;
                    Source source = function instanceof DensityFunctions.Spline
                            ? splines.get(function) : primitives.get(function);
                    if (source == null) throw new IllegalStateException("Missing staged source for " + function);
                    sourceSlots.put(i, source);
                }
                int outputAt = base.cornerOffsets()[leaf] + chunk * count * 3;
                graph(6, program, inlineSamples(program, grid), sourceSlots, outputAt);
            }
        }

        Source primitive(DensityFunction function, NoiseSampleGrid grid, Map<DensityFunction, Source> sources) {
            Source cached = sources.get(function);
            if (cached != null) return cached;
            String type = function.getClass().getSimpleName();
            int count = grid.sampleCount();
            Source result;
            if (function instanceof BlendedNoise blended) {
                result = noise(1, BlendedNoiseBatch.assembleGpuCoordinates(blended, grid), 6, count);
            } else if (type.equals("Noise") || type.equals("ShiftedNoise")) {
                DensityFunction.NoiseHolder holder = (DensityFunction.NoiseHolder)DensityGraphProgram.part(function, "noise");
                double xz = (double)DensityGraphProgram.part(function, "xzScale");
                double y = (double)DensityGraphProgram.part(function, "yScale");
                int[] raw = DoubleSingleNoiseBatch.assembleCoordinateNoise(
                        NormalNoiseSnapshot.capture(holder.noise()), grid, new double[count * 3], xz, y);
                int shaderType = 4;
                if (type.equals("ShiftedNoise")) {
                    int[] packed = Arrays.copyOf(raw, raw.length + 3);
                    for (int axis = 0; axis < 3; axis++) {
                        String field = axis == 0 ? "shiftX" : axis == 1 ? "shiftY" : "shiftZ";
                        DensityFunction shift = (DensityFunction)DensityGraphProgram.part(function, field);
                        DensityFunction unwrapped = unwrap(shift);
                        if (unwrapped.getClass().getSimpleName().equals("Constant")) {
                            double value = unwrapped.compute(new DensityFunction.SinglePointContext(0, 0, 0));
                            int sampleBase = packed[3];
                            for (int i = 0; i < count; i++) pair(packed, sampleBase + i * 9 + 3 + axis * 2, value);
                        } else packed[raw.length + axis] = shift(shift, grid, sources).offset() + 1;
                    }
                    raw = packed;
                    shaderType = 7;
                }
                result = noise(1, raw, shaderType, count);
            } else throw new UnsupportedOperationException("Unsupported noise leaf: " + type);
            sources.put(function, result);
            return result;
        }

        Source endIslands(DensityFunction function, NoiseSampleGrid grid) {
            int[] coordinates = grid.coordinates();
            int[] input = EndIslandBatch.assemble(function, grid);
            int inputAt = put(input), outputAt = reserve(grid.sampleCount() * 3);
            stageJobs[1].add(new Job(inputAt, outputAt, 9, input[0]));
            if (CHECK_END_ISLANDS.get() > 0) {
                int vertical = grid.height() / grid.cellHeight() + 1;
                for (int sample = 0; sample < input[0]; sample++) {
                    if (CHECK_END_ISLANDS.getAndUpdate(value -> Math.max(0, value - 1)) == 0) break;
                    int at = sample * vertical * 3;
                    int x = coordinates[at], z = coordinates[at + 2];
                    double expected = function.compute(new DensityFunction.SinglePointContext(x, 0, z));
                    endIslandChecks.add(new EndIslandCheck(outputAt + sample * vertical * 3,
                            expected, x, z));
                }
            }
            return new Source(outputAt, 3);
        }

        Source shift(DensityFunction original, NoiseSampleGrid grid, Map<DensityFunction, Source> sources) {
            Source cached = sources.get(original);
            if (cached != null) return cached;
            DensityFunction function = unwrap(original);
            String type = function.getClass().getSimpleName();
            if (!type.equals("ShiftA") && !type.equals("ShiftB") && !type.equals("Shift"))
                throw new UnsupportedOperationException("Unsupported shifted-noise coordinate: " + type);
            DensityFunction.NoiseHolder holder = (DensityFunction.NoiseHolder)DensityGraphProgram.part(function, "offsetNoise");
            int[] xyz = grid.coordinates(), transformed = new int[xyz.length];
            for (int i = 0; i < grid.sampleCount(); i++) {
                int at = i * 3;
                transformed[at] = type.equals("ShiftB") ? xyz[at + 2] : xyz[at];
                transformed[at + 1] = type.equals("ShiftA") ? 0
                        : type.equals("ShiftB") ? xyz[at] : xyz[at + 1];
                transformed[at + 2] = type.equals("ShiftB") ? 0 : xyz[at + 2];
            }
            NoiseSampleGrid shifted = new NoiseSampleGrid(transformed, 1, grid.sampleCount(),
                    grid.cellWidth(), grid.cellHeight(), grid.minY(), grid.height(), 0);
            int[] input = DoubleSingleNoiseBatch.assembleCoordinateNoise(
                    NormalNoiseSnapshot.capture(holder.noise()), shifted,
                    new double[transformed.length], 0.25, 0.25);
            Source result = noise(0, input, 5, grid.sampleCount());
            sources.put(original, result);
            return result;
        }

        private DensityFunction unwrap(DensityFunction original) {
            DensityFunction function = original;
            while (function instanceof DensityFunctions.HolderHolder
                    || function instanceof DensityFunctions.MarkerOrMarked) {
                if (function instanceof DensityFunctions.HolderHolder holder) function = holder.function().value();
                else function = ((DensityFunctions.MarkerOrMarked)function).wrapped();
            }
            return function;
        }

        private void pair(int[] words, int at, double value) {
            float hi = (float)value;
            words[at] = Float.floatToRawIntBits(hi);
            words[at + 1] = Float.floatToRawIntBits((float)(value - hi));
        }

        private Map<Integer, Source> leafSources(DensityGraphProgram program,
                                                  Map<DensityFunction, Source> primitives) {
            Map<Integer, Source> result = new HashMap<>();
            for (int i = 0; i < program.leafCount(); i++) {
                DensityFunction function = program.leaves().get(i);
                if (function == null) continue;
                Source source = primitives.get(function);
                if (source == null) throw new IllegalStateException("Missing coordinate noise: " + function);
                result.put(i, source);
            }
            return result;
        }

        private double[][] inlineSamples(DensityGraphProgram program, NoiseSampleGrid grid) {
            double[][] samples = new double[grid.sampleCount()][program.leafCount()];
            int[] xyz = grid.coordinates();
            for (int leaf = 0; leaf < program.leafCount(); leaf++) {
                if (program.leaves().get(leaf) == null)
                    for (int i = 0; i < samples.length; i++) samples[i][leaf] = xyz[i * 3 + 1];
            }
            return samples;
        }

        List<Stage> stages() {
            List<Stage> result = new ArrayList<>();
            for (int stage = 0; stage < stageJobs.length; stage++) {
                List<Job> jobs = stageJobs[stage];
                if (jobs.isEmpty()) continue;
                int[] table = new int[jobs.size() * 4];
                int samples = 0;
                for (int i = 0; i < jobs.size(); i++) {
                    Job job = jobs.get(i);
                    int at = i * 4;
                    table[at] = job.input();
                    table[at + 1] = job.output();
                    table[at + 2] = job.type();
                    table[at + 3] = job.samples();
                    samples = Math.max(samples, job.samples());
                }
                result.add(new Stage(stage == 0 || stage == 1 || stage == 3 ? 0
                        : stage == 5 ? 2 : 1, put(table), jobs.size(), samples));
            }
            return result;
        }

        int[] finish() { return Arrays.copyOf(words, cursor); }
    }
}
