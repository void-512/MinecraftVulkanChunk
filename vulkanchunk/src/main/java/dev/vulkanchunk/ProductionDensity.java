package dev.vulkanchunk;

import dev.vulkanchunk.mixin.NoiseBasedChunkGeneratorAccessor;
import dev.vulkanchunk.mixin.NoiseChunkAccessor;
import net.minecraft.util.KeyDispatchDataCodec;
import net.minecraft.server.MinecraftServer;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.StructureManager;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.DensityFunctions;
import net.minecraft.world.level.levelgen.NoiseBasedChunkGenerator;
import net.minecraft.world.level.levelgen.NoiseChunk;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;

import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLongArray;

/** Supplies final density to the existing NoiseChunk material rule during a real noise fill. */
public final class ProductionDensity {
    private static final ThreadLocal<Active> ACTIVE = new ThreadLocal<>();
    private static final Map<RandomState, Graphs> GRAPHS = new IdentityHashMap<>();
    private static final Map<RandomState, String> DIMENSIONS = new IdentityHashMap<>();
    private static volatile VanillaPlanSignatures SIGNATURES;
    private static final Map<NoiseGeneratorSettings, Optional<Plan>> PLAN_CACHE = new IdentityHashMap<>();
    private static final AtomicLong USED_CHUNKS = new AtomicLong();
    private static final ConcurrentHashMap<String, AtomicLong> NOISE_FILLS = new ConcurrentHashMap<>();
    private static final boolean MEASURE_FILL = Boolean.getBoolean("vulkanchunk.measureFill");
    private static final ThreadLocal<Long> FILL_START = new ThreadLocal<>();
    private static final ThreadLocal<Long> POST_START = new ThreadLocal<>();
    private static final ConcurrentHashMap<String, FillTiming> FILL_TIMINGS = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, AtomicLongArray> BEGIN_PHASES = new ConcurrentHashMap<>();
    private static final AtomicInteger VALIDATE_REMAINING = new AtomicInteger(
            Integer.getInteger("vulkanchunk.validateChunks", 0));
    private static final org.slf4j.Logger LOGGER = com.mojang.logging.LogUtils.getLogger();

    private ProductionDensity() {}

    private enum Plan { OVERWORLD, NETHER, END }

    /** Bundled vanilla resources anchor graph signatures; dimension identity never selects a plan. */
    public static void registerLevels(MinecraftServer server) {
        SIGNATURES = new VanillaPlanSignatures(server.registryAccess());
        synchronized (PLAN_CACHE) { PLAN_CACHE.clear(); }
        synchronized (DIMENSIONS) {
            DIMENSIONS.clear();
            for (var level : server.getAllLevels()) {
                var source = level.getChunkSource();
                var generator = source.getGenerator();
                String dimension = level.dimension().location().toString();
                DIMENSIONS.put(source.randomState(), dimension);
                if (!(generator instanceof NoiseBasedChunkGenerator))
                    VulkanChunkService.notApplicable(dimension, generator.getClass().getName());
                else LOGGER.info("[vulkanchunk] Generator for {}: {}", dimension, generator.getClass().getName());
            }
        }
    }

    public static void begin(NoiseBasedChunkGenerator generator, Blender blender,
                             StructureManager structures, RandomState state, ChunkAccess chunk,
                             int minCellY, int cellCountY) {
        if (MEASURE_FILL) FILL_START.set(System.nanoTime());
        ACTIVE.remove();
        if (!VulkanChunkService.active()) {
            SchedulingTrace.discarded(chunk);
            return;
        }
        String dimension;
        synchronized (DIMENSIONS) { dimension = DIMENSIONS.getOrDefault(state, "unknown dimension"); }
        SchedulingTrace.started(chunk, dimension);
        var settings = ((NoiseBasedChunkGeneratorAccessor)(Object)generator).vulkanchunk$settings().value();
        if (SIGNATURES == null) {
            VulkanChunkService.unsupported(dimension, "world graph registry is still initializing");
            return;
        }
        if (generator.getClass() != NoiseBasedChunkGenerator.class) {
            VulkanChunkService.unsupported(dimension, "generator subclass " + generator.getClass().getName()
                    + " has unvalidated fill behavior");
            return;
        }
        long mark = MEASURE_FILL ? System.nanoTime() : 0;
        Plan plan = matchingPlan(settings);
        phase(dimension, 0, mark);
        if (plan == null) {
            VulkanChunkService.unsupported(dimension, graphReason(state, settings));
            return;
        }
        if (blender != Blender.empty()) {
            VulkanChunkService.unsupported(dimension, "old-chunk blending is active");
            return;
        }
        try {
            mark = MEASURE_FILL ? System.nanoTime() : 0;
            NoiseChunk noise = chunk.getOrCreateNoiseChunk(source ->
                    ((NoiseBasedChunkGeneratorAccessor)(Object)generator)
                            .vulkanchunk$createNoiseChunk(source, structures, blender, state));
            Graphs graphs;
            synchronized (GRAPHS) {
                graphs = GRAPHS.computeIfAbsent(state, key -> compileGraphs(key, noise));
            }
            phase(dimension, 1, mark);
            mark = MEASURE_FILL ? System.nanoTime() : 0;
            int cellWidth = settings.noiseSettings().getCellWidth();
            int cellHeight = settings.noiseSettings().getCellHeight();
            int minY = minCellY * cellHeight;
            int height = cellCountY * cellHeight;
            double[] beard = new double[16 * 16 * height];
            long beardNonzero = 0;
            boolean countBeard = VALIDATE_REMAINING.get() > 0;
            DensityFunction beardifier = ((NoiseChunkAccessor)(Object)noise).vulkanchunk$beardifier();
            for (int y = minY; y < minY + height; y++) {
                for (int z = chunk.getPos().getMinBlockZ(); z < chunk.getPos().getMinBlockZ() + 16; z++) {
                    for (int x = chunk.getPos().getMinBlockX(); x < chunk.getPos().getMinBlockX() + 16; x++) {
                        int index = (y - minY) * 256 + (z - chunk.getPos().getMinBlockZ()) * 16
                                + x - chunk.getPos().getMinBlockX();
                        beard[index] = beardifier.compute(new DensityFunction.SinglePointContext(x, y, z));
                        if (countBeard && beard[index] != 0.0) beardNonzero++;
                    }
                }
            }
            StagedDensityBatch.Case request = new StagedDensityBatch.Case(graphs.outer(), graphs.inner(),
                    beard, cellWidth, cellHeight, height, chunk.getPos(), minY);
            SchedulingTrace.created();
            phase(dimension, 2, mark);
            mark = MEASURE_FILL ? System.nanoTime() : 0;
            double[] density = VulkanChunkService.submit(request, dimension);
            phase(dimension, 3, mark);
            if (density != null) {
                ACTIVE.set(new Active(noise, minY, height, chunk.getPos().getMinBlockX(),
                        chunk.getPos().getMinBlockZ(), density, beardNonzero));
                VulkanChunkService.activeFor(dimension, plan.name().toLowerCase(java.util.Locale.ROOT));
            }
        } catch (UnsupportedOperationException failure) {
            VulkanChunkService.unsupported(dimension, failure.getMessage());
        } catch (Throwable failure) {
            VulkanChunkService.failed(failure);
        }
    }

    private static void phase(String dimension, int index, long start) {
        if (MEASURE_FILL) BEGIN_PHASES.computeIfAbsent(dimension, key -> new AtomicLongArray(5))
                .addAndGet(index, System.nanoTime() - start);
    }

    public static void afterBegin() {
        if (MEASURE_FILL) POST_START.set(System.nanoTime());
    }

    private static Plan matchingPlan(NoiseGeneratorSettings settings) {
        synchronized (PLAN_CACHE) {
            return PLAN_CACHE.computeIfAbsent(settings, ProductionDensity::findPlan).orElse(null);
        }
    }

    private static Optional<Plan> findPlan(NoiseGeneratorSettings settings) {
        String fingerprint = SIGNATURES.fingerprint(settings);
        for (Plan plan : Plan.values())
            if (SIGNATURES.matches(plan.name().toLowerCase(java.util.Locale.ROOT), fingerprint)) return Optional.of(plan);
        return Optional.empty();
    }

    private static String graphReason(RandomState state, NoiseGeneratorSettings settings) {
        DensityFunction root = state.router().finalDensity();
        String reason = "unvalidated final-density graph (runtime root " + root.getClass().getName() + ")";
        reason += "; bundled vanilla graph/layout signature differs";
        return reason;
    }

    private static Graphs compileGraphs(RandomState state, NoiseChunk noise) {
        NoiseChunkAccessor accessor = (NoiseChunkAccessor)(Object)noise;
        DensityFunction root = DensityFunctions.add(state.router().finalDensity(), accessor.vulkanchunk$beardifier());
        DensityFunction wrapped = root.mapAll(accessor::vulkanchunk$wrap);
        DensityGraphProgram outer = new DensityGraphProgram(wrapped);
        if (outer.leafCount() < 2 || outer.leafCount() > 6) throw new UnsupportedOperationException(
                "final-density graph has unsupported leaf count " + outer.leafCount());
        int interpolators = outer.leafCount() - 1;
        if (!outer.leaves().get(interpolators).getClass().getSimpleName().equals("Beardifier"))
            throw new UnsupportedOperationException("final-density graph has a non-Beardifier external leaf");
        DensityGraphProgram[] inner = new DensityGraphProgram[interpolators];
        for (int i = 0; i < interpolators; i++) {
            DensityFunction leaf = outer.leaves().get(i);
            if (!(leaf instanceof DensityFunctions.MarkerOrMarked marker)
                    || !leaf.getClass().getSimpleName().equals("NoiseInterpolator"))
                throw new UnsupportedOperationException("unsupported final-density leaf " + i + ": "
                        + leaf.getClass().getSimpleName());
            inner[i] = new DensityGraphProgram(marker.wrapped(), true);
        }
        return new Graphs(outer, inner);
    }

    public static void end(NoiseBasedChunkGenerator generator, Blender blender,
                           StructureManager structures, RandomState state, ChunkAccess chunk) {
        SchedulingTrace.ended();
        if (MEASURE_FILL) {
            String dimension;
            synchronized (DIMENSIONS) { dimension = DIMENSIONS.getOrDefault(state, "unknown dimension"); }
            NOISE_FILLS.computeIfAbsent(dimension, key -> new AtomicLong()).incrementAndGet();
            Long start = FILL_START.get();
            FILL_START.remove();
            if (start != null) FILL_TIMINGS.computeIfAbsent(dimension, key -> new FillTiming())
                    .record(start, System.nanoTime());
            Long postStart = POST_START.get();
            POST_START.remove();
            if (postStart != null) phase(dimension, 4, postStart);
        }
        Active active = ACTIVE.get();
        if (active != null && active.used) USED_CHUNKS.incrementAndGet();
        ACTIVE.remove();
        if (active != null && active.used && VALIDATE_REMAINING.getAndUpdate(value -> Math.max(0, value - 1)) > 0) {
            try { validateActualChunk(generator, blender, structures, state, chunk, active); }
            catch (Throwable failure) { LOGGER.error("[vulkanchunk] Actual-chunk validation failed for {}", chunk.getPos(), failure); }
        }
    }

    public static long usedChunks() { return USED_CHUNKS.get(); }

    public static Map<String, Long> noiseFills() {
        Map<String, Long> counts = new TreeMap<>();
        NOISE_FILLS.forEach((dimension, count) -> counts.put(dimension, count.get()));
        return counts;
    }

    public static Map<String, String> fillTimings() {
        Map<String, String> timings = new TreeMap<>();
        FILL_TIMINGS.forEach((dimension, timing) -> timings.put(dimension,
                "count=" + timing.count.get() + " span ns="
                        + (timing.lastEnd.get() - timing.firstStart.get())
                        + " sum ns=" + timing.sumNanos.get()));
        return timings;
    }

    public static Map<String, String> beginPhases() {
        Map<String, String> timings = new TreeMap<>();
        BEGIN_PHASES.forEach((dimension, phases) -> timings.put(dimension,
                "plan=" + phases.get(0) + " noise=" + phases.get(1)
                        + " beard=" + phases.get(2) + " submitWait=" + phases.get(3)
                        + " minecraftAfter=" + phases.get(4)));
        return timings;
    }

    private static void validateActualChunk(NoiseBasedChunkGenerator generator, Blender blender,
                                            StructureManager structures, RandomState state, ChunkAccess chunk,
                                            Active active) {
        NoiseChunk control = ((NoiseBasedChunkGeneratorAccessor)(Object)generator)
                .vulkanchunk$createNoiseChunk(chunk, structures, blender, state);
        DensityFunction controlDensity = DensityFunctions.add(state.router().finalDensity(),
                ((NoiseChunkAccessor)(Object)control).vulkanchunk$beardifier())
                .mapAll(((NoiseChunkAccessor)(Object)control)::vulkanchunk$wrap);
        int width = ((NoiseBasedChunkGeneratorAccessor)(Object)generator)
                .vulkanchunk$settings().value().noiseSettings().getCellWidth();
        int cellHeight = ((NoiseBasedChunkGeneratorAccessor)(Object)generator)
                .vulkanchunk$settings().value().noiseSettings().getCellHeight();
        BlockState defaultBlock = ((NoiseBasedChunkGeneratorAccessor)(Object)generator)
                .vulkanchunk$settings().value().defaultBlock();
        BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
        long checked = 0, mismatches = 0, water = 0, lava = 0, caveAir = 0;
        String first = "none";
        control.initializeForFirstCellX();
        for (int cellX = 0; cellX < 16 / width; cellX++) {
            control.advanceCellX(cellX);
            for (int cellZ = 0; cellZ < 16 / width; cellZ++) {
                for (int cellY = active.height / cellHeight - 1; cellY >= 0; cellY--) {
                    control.selectCellYZ(cellY, cellZ);
                    for (int localY = cellHeight - 1; localY >= 0; localY--) {
                        int y = active.minY + cellY * cellHeight + localY;
                        control.updateForY(y, (double)localY / cellHeight);
                        for (int localX = 0; localX < width; localX++) {
                            int x = active.minX + cellX * width + localX;
                            control.updateForX(x, (double)localX / width);
                            for (int localZ = 0; localZ < width; localZ++) {
                                int z = active.minZ + cellZ * width + localZ;
                                control.updateForZ(z, (double)localZ / width);
                                BlockState expected = ((NoiseChunkAccessor)(Object)control)
                                        .vulkanchunk$getInterpolatedState();
                                if (expected == null) expected = defaultBlock;
                                BlockState actual = chunk.getBlockState(pos.set(x, y, z));
                                checked++;
                                if (actual.is(Blocks.WATER)) water++;
                                if (actual.is(Blocks.LAVA)) lava++;
                                if (y < 0 && actual.isAir()) caveAir++;
                                if (expected != actual) {
                                    mismatches++;
                                    if (first.equals("none")) {
                                        int index = (y - active.minY) * 256 + (z - active.minZ) * 16 + x - active.minX;
                                        first = pos + " expected=" + expected + " actual=" + actual
                                                + " cpuDensity=" + controlDensity.compute(control)
                                                + " gpuDensity=" + active.density[index];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            control.swapSlices();
        }
        control.stopInterpolation();
        LOGGER.info("[vulkanchunk] Actual noise-fill validation chunk {}: checked={} blockMismatches={} water={} lava={} caveAir={} beardNonzero={} first={}",
                chunk.getPos(), checked, mismatches, water, lava, caveAir, active.beardNonzero, first);
    }

    public static void clearWorlds() {
        synchronized (GRAPHS) { GRAPHS.clear(); }
        synchronized (DIMENSIONS) { DIMENSIONS.clear(); }
        SIGNATURES = null;
        synchronized (PLAN_CACHE) { PLAN_CACHE.clear(); }
        NormalNoiseSnapshot.clearCache();
        DoubleSingleNoiseBatch.clearCache();
        StagedDensityBatch.clearStaticPrograms();
        BlendedNoiseBatch.clearCache();
        EndIslandBatch.clearCache();
        NOISE_FILLS.clear();
        FILL_TIMINGS.clear();
        BEGIN_PHASES.clear();
        FILL_START.remove();
        POST_START.remove();
        ACTIVE.remove();
    }

    public static DensityFunction override(NoiseChunk owner, DensityFunction original) {
        return new OverrideFunction(owner, original);
    }

    private record Graphs(DensityGraphProgram outer, DensityGraphProgram[] inner) {}

    private static final class FillTiming {
        final AtomicLong count = new AtomicLong();
        final AtomicLong firstStart = new AtomicLong(Long.MAX_VALUE);
        final AtomicLong lastEnd = new AtomicLong(Long.MIN_VALUE);
        final AtomicLong sumNanos = new AtomicLong();

        void record(long start, long end) {
            firstStart.accumulateAndGet(start, Math::min);
            lastEnd.accumulateAndGet(end, Math::max);
            sumNanos.addAndGet(end - start);
            count.incrementAndGet();
        }
    }

    private static final class Active {
        final NoiseChunk noise;
        final int minY, height, minX, minZ;
        final double[] density;
        final long beardNonzero;
        boolean used;

        Active(NoiseChunk noise, int minY, int height, int minX, int minZ, double[] density,
               long beardNonzero) {
            this.noise = noise;
            this.minY = minY;
            this.height = height;
            this.minX = minX;
            this.minZ = minZ;
            this.density = density;
            this.beardNonzero = beardNonzero;
        }

        Double get(NoiseChunk owner, DensityFunction.FunctionContext point) {
            if (owner != noise) return null;
            int dx = point.blockX() - minX, dz = point.blockZ() - minZ, dy = point.blockY() - minY;
            if (dx < 0 || dx >= 16 || dz < 0 || dz >= 16 || dy < 0 || dy >= height) return null;
            used = true;
            return density[dy * 256 + dz * 16 + dx];
        }
    }

    private record OverrideFunction(NoiseChunk owner, DensityFunction original) implements DensityFunction {
        @Override public double compute(FunctionContext point) {
            Active active = ACTIVE.get();
            Double replacement = active == null ? null : active.get(owner, point);
            return replacement == null ? original.compute(point) : replacement;
        }
        @Override public void fillArray(double[] values, ContextProvider context) {
            context.fillAllDirectly(values, this);
        }
        @Override public DensityFunction mapAll(Visitor visitor) {
            return visitor.apply(new OverrideFunction(owner, original.mapAll(visitor)));
        }
        @Override public double minValue() { return original.minValue(); }
        @Override public double maxValue() { return original.maxValue(); }
        @Override public KeyDispatchDataCodec<? extends DensityFunction> codec() { return original.codec(); }
    }
}
