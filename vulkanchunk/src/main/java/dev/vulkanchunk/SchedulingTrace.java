package dev.vulkanchunk;

import net.minecraft.Util;
import net.minecraft.world.level.chunk.ChunkAccess;

import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;

/** Small opt-in timing probe for the Minecraft-to-Vulkan request boundary. */
public final class SchedulingTrace {
    static final boolean ENABLED = Boolean.getBoolean("vulkanchunk.scheduleTrace");
    static final int WINDOW_MICROS = Math.max(0, Integer.getInteger("vulkanchunk.batchWindowMicros", 1000));
    private static final int TARGET = Integer.getInteger("vulkanchunk.scheduleTraceTarget", 0);
    private static final ConcurrentHashMap<ChunkAccess, Long> INVOKED = new ConcurrentHashMap<>();
    private static final ThreadLocal<Request> CURRENT = new ThreadLocal<>();
    private static final ConcurrentHashMap<String, Stats> BY_DIMENSION = new ConcurrentHashMap<>();
    private static final AtomicInteger READY = new AtomicInteger();
    private static final AtomicInteger PREPARING = new AtomicInteger();
    private static final AtomicInteger WAITING = new AtomicInteger();
    private static final AtomicInteger MAX_READY = new AtomicInteger();
    private static final AtomicInteger MAX_WAITING = new AtomicInteger();
    private static final AtomicInteger MAX_POOL_SIZE = new AtomicInteger();
    private static final AtomicInteger MAX_POOL_ACTIVE = new AtomicInteger();
    private static final AtomicInteger MAX_POOL_QUEUED = new AtomicInteger();

    static final class Request {
        final String dimension;
        final long invocation, start;
        long created, enqueued, seen, submitted;
        Request(String dimension, long invocation, long start) {
            this.dimension = dimension;
            this.invocation = invocation;
            this.start = start;
        }
    }

    private static final class Stats {
        long requests, completions, batches, full, timeout, incompatible, shortAfter, partialQueueNonzero;
        long partialReadyElsewhere, partialPreparing, partialNoLatent;
        long invokeToStart, startToCreate, createToEnqueue, enqueueToSeen;
        long seenToSubmit, submitToComplete, totalLatency, interArrival;
        long enqueueSamples, lastEnqueue, lastPartialSubmit;
        long queueAtClose, maxQueue, oldestBatchWait, maxOldestBatchWait;
        long batchWait, batchWaitSamples;
        long idleNoRequest, idleWithRequest;
        long slot0, slot1, both, noSlotWork, slotBlockedWithRequest;
        long[] sizes = new long[5];
        long maxReady, maxWaiting, maxDoFill, activeDoFill;
        long poolActive, poolSize, poolQueued, poolSamples;
        long workerThreadRequests;
        boolean finished;
    }

    private SchedulingTrace() {}

    public static void invoked(ChunkAccess chunk) {
        if (!ENABLED) return;
        INVOKED.put(chunk, System.nanoTime());
        MAX_READY.accumulateAndGet(READY.incrementAndGet(), Math::max);
    }

    static void discarded(ChunkAccess chunk) {
        if (ENABLED && INVOKED.remove(chunk) != null) READY.decrementAndGet();
    }

    static void started(ChunkAccess chunk, String dimension) {
        if (!ENABLED) return;
        long now = System.nanoTime();
        Long invocation = INVOKED.remove(chunk);
        if (invocation != null) READY.decrementAndGet();
        Request request = new Request(dimension, invocation == null ? now : invocation, now);
        CURRENT.set(request);
        PREPARING.incrementAndGet();
        Stats stats = stats(dimension);
        synchronized (stats) {
            stats.invokeToStart += now - request.invocation;
            stats.maxReady = Math.max(stats.maxReady, READY.get());
            stats.maxDoFill = Math.max(stats.maxDoFill, ++stats.activeDoFill);
        }
        samplePool(stats);
    }

    static void created() {
        if (!ENABLED) return;
        Request request = CURRENT.get();
        if (request == null) return;
        request.created = System.nanoTime();
        Stats stats = stats(request.dimension);
        synchronized (stats) { stats.startToCreate += request.created - request.start; }
    }

    static Request enqueued(int depth) {
        if (!ENABLED) return null;
        long now = System.nanoTime();
        Request request = CURRENT.get();
        if (request == null) return null;
        request.enqueued = now;
        PREPARING.decrementAndGet();
        int waiting = WAITING.incrementAndGet();
        MAX_WAITING.accumulateAndGet(waiting, Math::max);
        Stats stats = stats(request.dimension);
        synchronized (stats) {
            stats.requests++;
            stats.createToEnqueue += request.created == 0 ? 0 : now - request.created;
            if (stats.lastEnqueue != 0) { stats.interArrival += now - stats.lastEnqueue; stats.enqueueSamples++; }
            stats.lastEnqueue = now;
            if (stats.lastPartialSubmit != 0 && now - stats.lastPartialSubmit <= 1_000_000L) stats.shortAfter++;
            stats.maxQueue = Math.max(stats.maxQueue, depth);
            stats.maxWaiting = Math.max(stats.maxWaiting, waiting);
            if (Thread.currentThread().getName().startsWith("Worker-Main-")) stats.workerThreadRequests++;
        }
        samplePool(stats);
        return request;
    }

    static void seen(Request request) {
        if (!ENABLED || request == null) return;
        request.seen = System.nanoTime();
        Stats stats = stats(request.dimension);
        synchronized (stats) { stats.enqueueToSeen += request.seen - request.enqueued; }
    }

    static void closed(String dimension, int size, boolean incompatible, int queueDepth, long firstEnqueue) {
        if (!ENABLED) return;
        long now = System.nanoTime();
        Stats stats = stats(dimension);
        synchronized (stats) {
            stats.batches++;
            stats.sizes[size]++;
            if (size == 4) stats.full++;
            else {
                if (incompatible) stats.incompatible++; else stats.timeout++;
                if (queueDepth > 0) stats.partialQueueNonzero++;
                int ready = READY.get(), preparing = PREPARING.get();
                if (ready > 0) stats.partialReadyElsewhere++;
                if (preparing > 0) stats.partialPreparing++;
                if (ready == 0 && preparing == 0) stats.partialNoLatent++;
            }
            stats.queueAtClose += queueDepth;
            stats.maxQueue = Math.max(stats.maxQueue, queueDepth);
            long wait = now - firstEnqueue;
            stats.oldestBatchWait += wait;
            stats.maxOldestBatchWait = Math.max(stats.maxOldestBatchWait, wait);
        }
    }

    static void partialSubmitted(String dimension) {
        if (!ENABLED) return;
        Stats stats = stats(dimension);
        synchronized (stats) { stats.lastPartialSubmit = System.nanoTime(); }
    }

    static void submitted(Request request) {
        if (!ENABLED || request == null) return;
        request.submitted = System.nanoTime();
        Stats stats = stats(request.dimension);
        synchronized (stats) {
            stats.seenToSubmit += request.submitted - request.seen;
            stats.batchWait += request.submitted - request.enqueued;
            stats.batchWaitSamples++;
        }
    }

    static void completed(Request request) {
        if (!ENABLED || request == null) return;
        long now = System.nanoTime();
        WAITING.decrementAndGet();
        Stats stats = stats(request.dimension);
        synchronized (stats) {
            stats.submitToComplete += now - request.submitted;
            stats.totalLatency += now - request.enqueued;
            stats.completions++;
            if (TARGET > 0 && stats.completions >= TARGET) stats.finished = true;
        }
    }

    static void ended() {
        if (!ENABLED) return;
        Request request = CURRENT.get();
        CURRENT.remove();
        if (request == null) return;
        if (request.enqueued == 0) PREPARING.decrementAndGet();
        Stats stats = stats(request.dimension);
        synchronized (stats) { stats.activeDoFill--; }
    }

    static void idle(String dimension, long noRequest, long withRequest, long slot0, long slot1,
                     long both, long noSlotWork, long slotBlockedWithRequest) {
        if (!ENABLED || dimension == null) return;
        Stats stats = stats(dimension);
        synchronized (stats) {
            if (stats.finished) return;
            stats.idleNoRequest += noRequest;
            stats.idleWithRequest += withRequest;
            stats.slot0 += slot0;
            stats.slot1 += slot1;
            stats.both += both;
            stats.noSlotWork += noSlotWork;
            stats.slotBlockedWithRequest += slotBlockedWithRequest;
        }
    }

    static String report() {
        if (!ENABLED) return "disabled";
        Map<String, String> result = new TreeMap<>();
        BY_DIMENSION.forEach((name, s) -> {
            synchronized (s) {
                long n = Math.max(1, s.requests), b = Math.max(1, s.batches);
                result.put(name, "requests=" + s.requests + " batches=" + s.batches
                        + " sizes=" + java.util.Arrays.toString(s.sizes)
                        + " partialTimeout=" + s.timeout + " partialIncompatible=" + s.incompatible
                        + " partialQueueNonzero=" + s.partialQueueNonzero
                        + " partialReadyElsewhere=" + s.partialReadyElsewhere
                        + " partialPreparing=" + s.partialPreparing
                        + " partialNoLatent=" + s.partialNoLatent
                        + " arrivalWithin1msAfterPartial=" + s.shortAfter
                        + " interArrivalAvgUs=" + micros(s.interArrival, s.enqueueSamples)
                        + " queueAtCloseAvg=" + ((double)s.queueAtClose / b) + " maxQueue=" + s.maxQueue
                        + " oldestBatchWaitAvgUs=" + micros(s.oldestBatchWait, b)
                        + " batchWaitAvgUs=" + micros(s.batchWait, s.batchWaitSamples)
                        + " requestLatencyAvgUs=" + micros(s.totalLatency, n)
                        + " invokeToStartAvgUs=" + micros(s.invokeToStart, n)
                        + " startToCreateAvgUs=" + micros(s.startToCreate, n)
                        + " createToEnqueueAvgUs=" + micros(s.createToEnqueue, n)
                        + " enqueueToSeenAvgUs=" + micros(s.enqueueToSeen, n)
                        + " seenToSubmitAvgUs=" + micros(s.seenToSubmit, n)
                        + " submitToCompleteAvgUs=" + micros(s.submitToComplete, n)
                        + " idleNoRequestMs=" + millis(s.idleNoRequest)
                        + " idleWithRequestMs=" + millis(s.idleWithRequest)
                        + " slot0Ms=" + millis(s.slot0) + " slot1Ms=" + millis(s.slot1)
                        + " bothMs=" + millis(s.both) + " noSlotWorkMs=" + millis(s.noSlotWork)
                        + " slotBlockedWithRequestMs=" + millis(s.slotBlockedWithRequest)
                        + " maxReady=" + s.maxReady + " maxDoFill=" + s.maxDoFill
                        + " maxWaiting=" + s.maxWaiting + " workerThreadRequests=" + s.workerThreadRequests
                        + " poolActiveAvg=" + ((double)s.poolActive / Math.max(1,s.poolSamples))
                        + " poolSizeMax=" + s.poolSize + " poolQueuedMax=" + s.poolQueued);
            }
        });
        return "windowUs=" + WINDOW_MICROS + " target=" + TARGET + " poolParallelism="
                + (Util.backgroundExecutor() instanceof ForkJoinPool pool ? pool.getParallelism() : -1)
                + " globalMaxReady=" + MAX_READY.get()
                + " globalMaxWaiting=" + MAX_WAITING.get() + " poolMaxSize=" + MAX_POOL_SIZE.get()
                + " poolMaxActive=" + MAX_POOL_ACTIVE.get() + " poolMaxQueued=" + MAX_POOL_QUEUED.get()
                + " dimensions=" + result;
    }

    private static double micros(long nanos, long count) { return (double)nanos / Math.max(1, count) / 1000.0; }
    private static double millis(long nanos) { return nanos / 1_000_000.0; }
    private static Stats stats(String dimension) { return BY_DIMENSION.computeIfAbsent(dimension, ignored -> new Stats()); }
    private static void samplePool(Stats stats) {
        if (!(Util.backgroundExecutor() instanceof ForkJoinPool pool)) return;
        int active = pool.getActiveThreadCount(), size = pool.getPoolSize();
        int queued = (int)Math.min(Integer.MAX_VALUE, pool.getQueuedSubmissionCount());
        MAX_POOL_ACTIVE.accumulateAndGet(active, Math::max);
        MAX_POOL_SIZE.accumulateAndGet(size, Math::max);
        MAX_POOL_QUEUED.accumulateAndGet(queued, Math::max);
        synchronized (stats) {
            stats.poolActive += active;
            stats.poolSamples++;
            stats.poolSize = Math.max(stats.poolSize, size);
            stats.poolQueued = Math.max(stats.poolQueued, queued);
        }
    }
}
