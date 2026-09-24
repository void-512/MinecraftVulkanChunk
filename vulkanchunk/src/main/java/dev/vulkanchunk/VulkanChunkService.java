package dev.vulkanchunk;

import com.mojang.logging.LogUtils;
import org.slf4j.Logger;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CompletionException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;
import java.util.TreeMap;

/** Owns the production density backend and its four-chunk worker. */
public final class VulkanChunkService {
    private static final Logger LOGGER = LogUtils.getLogger();
    private static final int BATCH_CAPACITY = 4;
    private static final boolean MEASURE_FILL = Boolean.getBoolean("vulkanchunk.measureFill");
    private static final LinkedBlockingQueue<Job> QUEUE = new LinkedBlockingQueue<>();
    private static final Set<String> WARNED_UNSUPPORTED = new HashSet<>();
    private static final AtomicLong SUBMITTED = new AtomicLong();
    private static final AtomicLong COMPLETED = new AtomicLong();
    private static final AtomicLong BATCHES = new AtomicLong();
    private static final AtomicLong BACKEND_NANOS = new AtomicLong();
    private static final AtomicLong GPU_NANOS = new AtomicLong();
    private static final AtomicLong GPU_TIMED_BATCHES = new AtomicLong();
    private static final AtomicLong UPLOAD_NANOS = new AtomicLong();
    private static final AtomicLong COMMAND_NANOS = new AtomicLong();
    private static final AtomicLong SUBMIT_NANOS = new AtomicLong();
    private static final AtomicLong WAIT_NANOS = new AtomicLong();
    private static final AtomicLong READBACK_NANOS = new AtomicLong();
    private static final AtomicLong GPU_COPY_NANOS = new AtomicLong();
    private static final AtomicLong INVALIDATE_NANOS = new AtomicLong();
    private static final AtomicLong ALLOCATION_NANOS = new AtomicLong();
    private static final AtomicLong NATIVE_COPY_NANOS = new AtomicLong();
    private static final AtomicLong DECODE_NANOS = new AtomicLong();
    private static final AtomicLong OUTPUT_BYTES = new AtomicLong();
    private static final AtomicLong ASSEMBLY_NANOS = new AtomicLong();
    private static final AtomicLong BOTH_SLOT_BATCHES = new AtomicLong();
    private static final AtomicLong SLOT_STALL_NANOS = new AtomicLong();
    private static final AtomicLong FENCE_IDLE_NANOS = new AtomicLong();
    private static final AtomicLong ASSEMBLY_OVERLAP_NANOS = new AtomicLong();
    private static final AtomicLongArray GPU_STAGES = new AtomicLongArray(8);
    private static final ConcurrentHashMap<String, WorkerTiming> WORKER_TIMINGS = new ConcurrentHashMap<>();
    private static volatile State state = State.INITIALIZING;
    private static volatile String reason = "initializing";
    private static volatile VulkanComputeBackend backend;
    private static volatile Thread worker;
    private static volatile String device = "unavailable";
    private static volatile String driver = "unavailable";
    private static volatile String lastUnsupported;
    private static final java.util.Map<String, String> DIMENSION_STATUS = new java.util.LinkedHashMap<>();

    public enum State { INITIALIZING, VULKAN_ACTIVE, CPU_FALLBACK, VULKAN_FAILED }

    private VulkanChunkService() {}

    public static synchronized void initialize() {
        if (backend != null || state == State.VULKAN_FAILED) return;
        if (Boolean.getBoolean("vulkanchunk.forceCpu")) {
            state = State.CPU_FALLBACK;
            reason = "CPU mode requested by vulkanchunk.forceCpu";
            LOGGER.warn("[vulkanchunk] Vulkan backend disabled: {}. Using original CPU chunk generation.", reason);
            return;
        }
        state = State.INITIALIZING;
        VulkanComputeBackend created = null;
        try {
            created = new VulkanComputeBackend();
            created.prepareStagedDensity();
            backend = created;
            device = created.deviceName();
            driver = created.driverName() + " / " + created.driverInfo();
            reason = "none";
            state = State.VULKAN_ACTIVE;
            Thread thread = new Thread(VulkanChunkService::runWorker, "vulkanchunk-density");
            thread.setDaemon(true);
            worker = thread;
            thread.start();
            LOGGER.info("[vulkanchunk] Vulkan chunk-density backend active: {} ({})", device, driver);
        } catch (Throwable failure) {
            state = State.CPU_FALLBACK;
            reason = failure.toString();
            VulkanComputeBackend old = backend;
            backend = null;
            if (old != null) old.close();
            else if (created != null) created.close();
            LOGGER.warn("[vulkanchunk] Vulkan backend unavailable: {}. Using original CPU chunk generation.", reason);
        }
    }

    public static boolean active() { return state == State.VULKAN_ACTIVE; }

    public static double[] submit(StagedDensityBatch.Case request, String dimension) {
        Job job;
        synchronized (VulkanChunkService.class) {
            if (!active()) return null;
            job = new Job(request, dimension,
                    MEASURE_FILL || SchedulingTrace.ENABLED ? System.nanoTime() : 0,
                    new CompletableFuture<>(), SchedulingTrace.enqueued(QUEUE.size() + 1));
            SUBMITTED.incrementAndGet();
            QUEUE.add(job);
        }
        try { return job.result().join(); }
        catch (CompletionException failure) {
            if (failure.getCause() instanceof UnsupportedOperationException unsupported) {
                unsupported(dimension, unsupported.getMessage());
                return null;
            }
            if (active()) failed(failure);
            return null;
        }
        catch (RuntimeException failure) {
            if (active()) failed(failure);
            return null;
        }
    }

    private static void runWorker() {
        VulkanComputeBackend activeBackend = backend;
        List<Job> group = new ArrayList<>(BATCH_CAPACITY);
        Pending[] pending = new Pending[2];
        String traceDimension = null;
        long stateAt = System.nanoTime();
        int stateMask = 0;
        while (state == State.VULKAN_ACTIVE) {
            try {
                if (SchedulingTrace.ENABLED && traceDimension != null) {
                    long now = System.nanoTime();
                    traceOccupancy(traceDimension, stateMask, now - stateAt);
                    stateAt = now;
                    stateMask = (pending[0] == null ? 0 : 1) | (pending[1] == null ? 0 : 2);
                }
                int free = pending[0] == null ? 0 : pending[1] == null ? 1 : -1;
                if (free >= 0) {
                    long pollStart = SchedulingTrace.ENABLED ? System.nanoTime() : 0;
                    Job first = QUEUE.poll(pending[1 - free] == null ? 100 : 1, TimeUnit.MILLISECONDS);
                    if (SchedulingTrace.ENABLED && pending[0] == null && pending[1] == null && traceDimension != null)
                        SchedulingTrace.idle(traceDimension, System.nanoTime() - pollStart, 0, 0, 0, 0, 0, 0);
                    if (first != null) {
                        long firstSeenAt = SchedulingTrace.ENABLED ? System.nanoTime() : 0;
                        if (SchedulingTrace.ENABLED && !first.dimension().equals(traceDimension)) {
                            stateAt = firstSeenAt;
                            stateMask = (pending[0] == null ? 0 : 1) | (pending[1] == null ? 0 : 2);
                        }
                        traceDimension = first.dimension();
                        SchedulingTrace.seen(first.trace());
                        long batchStart = MEASURE_FILL ? System.nanoTime() : 0;
                        group.add(first);
                        long until = System.nanoTime() + SchedulingTrace.WINDOW_MICROS * 1000L;
                        boolean incompatible = false;
                        while (group.size() < BATCH_CAPACITY) {
                            Job next = QUEUE.poll(Math.max(0, until - System.nanoTime()), TimeUnit.NANOSECONDS);
                            if (next == null) break;
                            if (!compatible(first.request(), next.request())) {
                                QUEUE.add(next);
                                incompatible = true;
                                break;
                            }
                            SchedulingTrace.seen(next.trace());
                            group.add(next);
                        }
                        SchedulingTrace.closed(first.dimension(), group.size(), incompatible, QUEUE.size(), first.enqueued());
                        Pending other = pending[1 - free];
                        boolean overlap = MEASURE_FILL && other != null && !activeBackend.completed(other.slot);
                        long assemblyStart = MEASURE_FILL ? System.nanoTime() : 0;
                        VulkanComputeBackend.Slot slot = activeBackend.slot(free);
                        StagedDensityBatch batch = StagedDensityBatch.assembleProduction(
                                group.stream().map(Job::request).toList(), slot.workspace());
                        long assemblyEnd = MEASURE_FILL ? System.nanoTime() : 0;
                        activeBackend.submitStagedDensity(slot, batch);
                        long submitEnd = MEASURE_FILL ? System.nanoTime() : 0;
                        if (group.size() < BATCH_CAPACITY) SchedulingTrace.partialSubmitted(first.dimension());
                        for (Job job : group) SchedulingTrace.submitted(job.trace());
                        pending[free] = new Pending(slot, List.copyOf(group), first.dimension());
                        if (SchedulingTrace.ENABLED) {
                            long now = System.nanoTime();
                            traceOccupancy(first.dimension(), stateMask, now - stateAt);
                            if (pending[1 - free] == null)
                                SchedulingTrace.idle(first.dimension(), 0, now - firstSeenAt, 0, 0, 0, 0, 0);
                            stateMask = (pending[0] == null ? 0 : 1) | (pending[1] == null ? 0 : 2);
                            stateAt = now;
                        }
                        if (MEASURE_FILL) {
                            ASSEMBLY_NANOS.addAndGet(assemblyEnd - assemblyStart);
                            if (overlap && !activeBackend.completed(other.slot))
                                ASSEMBLY_OVERLAP_NANOS.addAndGet(assemblyEnd - assemblyStart);
                            if (other != null) BOTH_SLOT_BATCHES.incrementAndGet();
                            WorkerTiming timing = WORKER_TIMINGS.computeIfAbsent(first.dimension(), key -> new WorkerTiming());
                            timing.first.accumulateAndGet(batchStart, Math::min);
                            timing.active.addAndGet(submitEnd - batchStart);
                            for (Job job : group) timing.queueWait.addAndGet(Math.max(0, batchStart - job.enqueued()));
                        }
                        continue;
                    }
                }
                int ready = pending[0] != null && activeBackend.completed(pending[0].slot) ? 0
                        : pending[1] != null && activeBackend.completed(pending[1].slot) ? 1 : -1;
                if (ready < 0) {
                    int waiting = pending[0] != null ? 0 : pending[1] != null ? 1 : -1;
                    if (waiting < 0) continue;
                    long waitStart = MEASURE_FILL ? System.nanoTime() : 0;
                    boolean signaled = activeBackend.waitForCompletion(pending[waiting].slot, 1_000_000);
                    if (MEASURE_FILL) {
                        long elapsed = System.nanoTime() - waitStart;
                        pending[waiting].waitNanos += elapsed;
                        FENCE_IDLE_NANOS.addAndGet(elapsed);
                        if (free < 0) SLOT_STALL_NANOS.addAndGet(elapsed);
                    }
                    if (!signaled) continue;
                    ready = waiting;
                }
                Pending done = pending[ready];
                long completionStart = MEASURE_FILL ? System.nanoTime() : 0;
                ComputeResult result = activeBackend.finishStagedDensity(done.slot, Long.MAX_VALUE);
                recordResult(result, done.waitNanos);
                long decodeStart = MEASURE_FILL ? System.nanoTime() : 0;
                int offset = 0;
                for (Job job : done.jobs) {
                    int count = job.request().beard().length;
                    double[] values = new double[count];
                    for (int i = 0; i < count; i++) {
                        int at = (offset + i) * 3;
                        values[i] = (double)Float.intBitsToFloat(result.values().get(at))
                                + Float.intBitsToFloat(result.values().get(at + 1));
                    }
                    job.result().complete(values);
                    SchedulingTrace.completed(job.trace());
                    COMPLETED.incrementAndGet();
                    offset += count;
                }
                long completionEnd = MEASURE_FILL ? System.nanoTime() : 0;
                if (MEASURE_FILL) {
                    DECODE_NANOS.addAndGet(completionEnd - decodeStart);
                    WorkerTiming timing = WORKER_TIMINGS.get(done.dimension);
                    timing.last.accumulateAndGet(completionEnd, Math::max);
                    timing.active.addAndGet(completionEnd - completionStart);
                    timing.batches.incrementAndGet();
                }
                BATCHES.incrementAndGet();
                if (SchedulingTrace.ENABLED) {
                    long now = System.nanoTime();
                    traceOccupancy(done.dimension, stateMask, now - stateAt);
                    stateAt = now;
                }
                pending[ready] = null;
                if (SchedulingTrace.ENABLED) {
                    stateMask = (pending[0] == null ? 0 : 1) | (pending[1] == null ? 0 : 2);
                }
            } catch (UnsupportedOperationException unsupported) {
                for (Job job : group) job.result().completeExceptionally(unsupported);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                break;
            } catch (Throwable failure) {
                failed(failure);
                for (Job job : group) job.result().completeExceptionally(failure);
                for (Pending batch : pending) if (batch != null)
                    for (Job job : batch.jobs) job.result().completeExceptionally(failure);
                break;
            } finally {
                group.clear();
            }
        }
        for (Pending batch : pending) if (batch != null)
            for (Job job : batch.jobs)
                job.result().completeExceptionally(new IllegalStateException("Vulkan worker stopped"));
    }

    private static void traceOccupancy(String dimension, int mask, long elapsed) {
        SchedulingTrace.idle(dimension, 0, 0,
                (mask & 1) != 0 ? elapsed : 0,
                (mask & 2) != 0 ? elapsed : 0,
                mask == 3 ? elapsed : 0,
                mask == 0 ? elapsed : 0,
                mask == 3 && !QUEUE.isEmpty() ? elapsed : 0);
    }

    private static void recordResult(ComputeResult result, long earlierFenceWait) {
        if (!MEASURE_FILL) return;
        for (int i = 0; i < 8; i++) GPU_STAGES.addAndGet(i, result.gpuStageNanos()[i]);
        BACKEND_NANOS.addAndGet(result.totalNanos());
        if (result.gpuNanos() >= 0) {
            GPU_NANOS.addAndGet(result.gpuNanos());
            GPU_TIMED_BATCHES.incrementAndGet();
        }
        UPLOAD_NANOS.addAndGet(result.uploadNanos());
        COMMAND_NANOS.addAndGet(result.commandRecordNanos());
        SUBMIT_NANOS.addAndGet(result.queueSubmitNanos());
        WAIT_NANOS.addAndGet(result.fenceWaitNanos() + earlierFenceWait);
        READBACK_NANOS.addAndGet(result.readbackNanos());
        if (result.gpuCopyNanos() >= 0) GPU_COPY_NANOS.addAndGet(result.gpuCopyNanos());
        INVALIDATE_NANOS.addAndGet(result.invalidateNanos());
        ALLOCATION_NANOS.addAndGet(result.allocationNanos());
        NATIVE_COPY_NANOS.addAndGet(result.nativeCopyNanos());
        OUTPUT_BYTES.addAndGet(result.outputBytes());
    }

    private static boolean compatible(StagedDensityBatch.Case a, StagedDensityBatch.Case b) {
        return a.outer() == b.outer() && a.inner() == b.inner()
                && a.cellWidth() == b.cellWidth() && a.cellHeight() == b.cellHeight()
                && a.height() == b.height() && a.minY() == b.minY();
    }

    public static synchronized void unsupported(String message) {
        lastUnsupported = message;
        if (message != null && WARNED_UNSUPPORTED.add(message))
            LOGGER.warn("[vulkanchunk] Vulkan chunk generation unsupported: {}. Using original CPU chunk generation for affected chunks.", message);
    }

    public static synchronized void unsupported(String dimension, String message) {
        DIMENSION_STATUS.put(dimension, "CPU: " + message);
        unsupported(dimension + ": " + message);
    }

    public static synchronized void notApplicable(String dimension, String generator) {
        String message = "unsupported chunk generator " + generator + "; original mod/vanilla generation applies";
        DIMENSION_STATUS.put(dimension, "not applicable: " + generator);
        if (WARNED_UNSUPPORTED.add(dimension + ": " + message))
            LOGGER.warn("[vulkanchunk] Vulkan density offload unavailable for {}: {}", dimension, message);
    }

    public static synchronized void activeFor(String dimension, String plan) {
        String status = "Vulkan active (" + plan + " plan)";
        if (!status.equals(DIMENSION_STATUS.put(dimension, status)))
            LOGGER.info("[vulkanchunk] Vulkan density plan active: {} for {}", plan, dimension);
    }

    public static synchronized void failed(Throwable failure) {
        if (state == State.VULKAN_FAILED || (state == State.CPU_FALLBACK && backend == null)) return;
        reason = failure.toString();
        state = State.VULKAN_FAILED;
        LOGGER.error("[vulkanchunk] Vulkan density backend failed: {}. Using original CPU chunk generation.", reason, failure);
        Job queued;
        while ((queued = QUEUE.poll()) != null) queued.result().completeExceptionally(failure);
    }

    public static synchronized String status() {
        return "vulkanchunk backend: " + (active() ? "Vulkan" : "CPU fallback")
                + ", state=" + state + (active() ? ", device=" + device + ", driver=" + driver
                + ", density pipeline=active, batching=active (max " + BATCH_CAPACITY + ")"
                : ", reason=" + reason)
                + ", submitted=" + SUBMITTED.get() + ", completed=" + COMPLETED.get()
                + ", used=" + ProductionDensity.usedChunks()
                + (MEASURE_FILL
                ? ", noise fills=" + ProductionDensity.noiseFills()
                + ", fill timing=" + ProductionDensity.fillTimings()
                + ", fill phases ns=" + ProductionDensity.beginPhases() : "")
                + ", batches=" + BATCHES.get()
                + ", average batch=" + (BATCHES.get() == 0 ? 0.0
                : (double)COMPLETED.get() / BATCHES.get())
                + (backend == null ? "" : ", output policy=" + backend.outputPolicy())
                + (MEASURE_FILL
                ? ", backend ns=" + BACKEND_NANOS.get()
                + ", gpu ns=" + GPU_NANOS.get() + " (timed batches=" + GPU_TIMED_BATCHES.get() + ")"
                + ", upload ns=" + UPLOAD_NANOS.get()
                + ", command ns=" + COMMAND_NANOS.get()
                + ", submit ns=" + SUBMIT_NANOS.get()
                + ", wait ns=" + WAIT_NANOS.get()
                + ", readback ns=" + READBACK_NANOS.get() : "")
                + (MEASURE_FILL
                ? ", gpu copy ns=" + GPU_COPY_NANOS.get()
                + ", invalidate ns=" + INVALIDATE_NANOS.get()
                + ", array alloc ns=" + ALLOCATION_NANOS.get()
                + ", native copy ns=" + NATIVE_COPY_NANOS.get()
                + ", decode ns=" + DECODE_NANOS.get()
                + ", assembly ns=" + ASSEMBLY_NANOS.get()
                + ", assembly phases ns=" + StagedDensityBatch.assemblyPhases()
                + ", gpu stages ns=" + gpuStages()
                + ", worker timing ns=" + workerTimings()
                + ", both-slot batches=" + BOTH_SLOT_BATCHES.get()
                + ", slot stall ns=" + SLOT_STALL_NANOS.get()
                + ", idle fence wait ns=" + FENCE_IDLE_NANOS.get()
                + ", confirmed assembly overlap ns=" + ASSEMBLY_OVERLAP_NANOS.get()
                + ", output bytes=" + OUTPUT_BYTES.get() : "")
                + (SchedulingTrace.ENABLED ? ", scheduling trace=" + SchedulingTrace.report() : "")
                + (backend == null ? "" : ", uploaded input bytes=" + backend.uploadedInputBytes())
                + (lastUnsupported == null ? "" : ", unsupported input=" + lastUnsupported)
                + ", dimensions=" + DIMENSION_STATUS;
    }

    private static String gpuStages() {
        String[] names = {"shift", "noise", "rarityGraph", "rarityNoise", "coordinateGraph",
                "spline", "latticeGraph", "finalDensity"};
        StringBuilder report = new StringBuilder("{");
        for (int i = 0; i < names.length; i++) {
            if (i > 0) report.append(", ");
            report.append(names[i]).append('=').append(GPU_STAGES.get(i));
        }
        return report.append('}').toString();
    }

    private static Map<String, String> workerTimings() {
        Map<String, String> report = new TreeMap<>();
        WORKER_TIMINGS.forEach((dimension, timing) -> report.put(dimension,
                "span=" + (timing.last.get() - timing.first.get())
                        + " active=" + timing.active.get() + " batches=" + timing.batches.get()
                        + " queueWait=" + timing.queueWait.get()));
        return report;
    }

    public static void shutdown() {
        Thread oldWorker;
        VulkanComputeBackend old;
        synchronized (VulkanChunkService.class) {
            state = State.CPU_FALLBACK;
            reason = "server stopped";
            oldWorker = worker;
            worker = null;
            old = backend;
            backend = null;
            Job queued;
            while ((queued = QUEUE.poll()) != null)
                queued.result().completeExceptionally(new IllegalStateException("server stopped"));
        }
        if (oldWorker != null) {
            oldWorker.interrupt();
            try { oldWorker.join(30_000); }
            catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
        }
        if (old != null) old.close();
        ProductionDensity.clearWorlds();
        synchronized (VulkanChunkService.class) { DIMENSION_STATUS.clear(); WARNED_UNSUPPORTED.clear(); }
    }

    private static final class WorkerTiming {
        final AtomicLong first = new AtomicLong(Long.MAX_VALUE);
        final AtomicLong last = new AtomicLong(Long.MIN_VALUE);
        final AtomicLong active = new AtomicLong();
        final AtomicLong batches = new AtomicLong();
        final AtomicLong queueWait = new AtomicLong();
    }

    private static final class Pending {
        final VulkanComputeBackend.Slot slot;
        final List<Job> jobs;
        final String dimension;
        long waitNanos;

        Pending(VulkanComputeBackend.Slot slot, List<Job> jobs, String dimension) {
            this.slot = slot;
            this.jobs = jobs;
            this.dimension = dimension;
        }
    }

    private record Job(StagedDensityBatch.Case request, String dimension, long enqueued,
                       CompletableFuture<double[]> result, SchedulingTrace.Request trace) {}
}
