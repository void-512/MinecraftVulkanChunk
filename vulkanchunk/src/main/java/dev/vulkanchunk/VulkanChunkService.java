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

/** Owns the production density backend and its four-chunk worker. */
public final class VulkanChunkService {
    private static final Logger LOGGER = LogUtils.getLogger();
    private static final int BATCH_CAPACITY = 4;
    private static final LinkedBlockingQueue<Job> QUEUE = new LinkedBlockingQueue<>();
    private static final Set<String> WARNED_UNSUPPORTED = new HashSet<>();
    private static final AtomicLong SUBMITTED = new AtomicLong();
    private static final AtomicLong COMPLETED = new AtomicLong();
    private static final AtomicLong BATCHES = new AtomicLong();
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
        Job job = new Job(request, new CompletableFuture<>());
        synchronized (VulkanChunkService.class) {
            if (!active()) return null;
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
        List<Job> group = new ArrayList<>(BATCH_CAPACITY);
        while (state == State.VULKAN_ACTIVE) {
            try {
                Job first = QUEUE.poll(100, TimeUnit.MILLISECONDS);
                if (first == null) continue;
                group.add(first);
                long until = System.nanoTime() + 1_000_000;
                while (group.size() < BATCH_CAPACITY) {
                    Job next = QUEUE.poll(Math.max(0, until - System.nanoTime()), TimeUnit.NANOSECONDS);
                    if (next == null) break;
                    if (!compatible(first.request(), next.request())) {
                        QUEUE.add(next);
                        break;
                    }
                    group.add(next);
                }
                StagedDensityBatch batch = StagedDensityBatch.assembleProduction(
                        group.stream().map(Job::request).toList());
                ComputeResult result = backend.evaluateStagedDensity(batch);
                int offset = 0;
                for (Job job : group) {
                    int count = job.request().beard().length;
                    double[] values = new double[count];
                    for (int i = 0; i < count; i++) {
                        int at = (offset + i) * 3;
                        values[i] = (double)Float.intBitsToFloat(result.values()[at])
                                + Float.intBitsToFloat(result.values()[at + 1]);
                    }
                    job.result().complete(values);
                    COMPLETED.incrementAndGet();
                    offset += count;
                }
                BATCHES.incrementAndGet();
            } catch (UnsupportedOperationException unsupported) {
                for (Job job : group) job.result().completeExceptionally(unsupported);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                break;
            } catch (Throwable failure) {
                failed(failure);
                for (Job job : group) job.result().completeExceptionally(failure);
                break;
            } finally {
                group.clear();
            }
        }
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
                + ", batches=" + BATCHES.get()
                + ", average batch=" + (BATCHES.get() == 0 ? 0.0
                : (double)COMPLETED.get() / BATCHES.get())
                + (backend == null ? "" : ", uploaded input bytes=" + backend.uploadedInputBytes())
                + (lastUnsupported == null ? "" : ", unsupported input=" + lastUnsupported)
                + ", dimensions=" + DIMENSION_STATUS;
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

    private record Job(StagedDensityBatch.Case request, CompletableFuture<double[]> result) {}
}
