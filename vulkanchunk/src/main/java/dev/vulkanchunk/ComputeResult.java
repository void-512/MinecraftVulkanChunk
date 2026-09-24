package dev.vulkanchunk;

public record ComputeResult(
        int[] values,
        long uploadNanos,
        long commandRecordNanos,
        long queueSubmitNanos,
        long fenceWaitNanos,
        long gpuNanos,
        long readbackNanos,
        long totalNanos) {

    public long submitAndWaitNanos() {
        return queueSubmitNanos + fenceWaitNanos;
    }
}
