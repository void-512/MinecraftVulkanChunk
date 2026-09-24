package dev.vulkanchunk;

public record ComputeResult(
        java.nio.IntBuffer values,
        long uploadNanos,
        long commandRecordNanos,
        long queueSubmitNanos,
        long fenceWaitNanos,
        long gpuNanos,
        long gpuCopyNanos,
        long[] gpuStageNanos,
        long invalidateNanos,
        long allocationNanos,
        long nativeCopyNanos,
        long outputBytes,
        long readbackNanos,
        long totalNanos) {

    public long submitAndWaitNanos() {
        return queueSubmitNanos + fenceWaitNanos;
    }
}
