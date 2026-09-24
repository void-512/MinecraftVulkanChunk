package dev.vulkanchunk;

import java.nio.IntBuffer;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

record DensityBatchLayout(int[] cornerOffsets, int cornerSamples, int blocks, int end) {
    private static final Map<DensityGraphProgram, int[]> OUTER_PROGRAMS = new IdentityHashMap<>();

    static void clearCache() { OUTER_PROGRAMS.clear(); }

    static DensityBatchLayout assemble(List<StagedDensityBatch.Case> cases, IntBuffer words) {
        StagedDensityBatch.Case first = cases.getFirst();
        int chunks = cases.size();
        int cornersPerChunk = (16 / first.cellWidth() + 1) * (16 / first.cellWidth() + 1)
                * (first.height() / first.cellHeight() + 1);
        int cornerSamples = chunks * cornersPerChunk;
        int blocks = chunks * first.beard().length;
        int[] outerProgram = OUTER_PROGRAMS.computeIfAbsent(first.outer(), program -> {
            int[] batch = program.batch(new double[][]{new double[program.leafCount()]}, 0, 1);
            int[] instructions = new int[program.nodeCount() * 10];
            System.arraycopy(batch, 5, instructions, 0, instructions.length);
            return instructions;
        });
        int cornerBase = 16 + outerProgram.length;
        int interpolators = first.inner().length;
        int[] cornerOffsets = new int[interpolators];
        for (int leaf = 0; leaf < interpolators; leaf++)
            cornerOffsets[leaf] = cornerBase + leaf * cornerSamples * 3;
        int beardBase = cornerBase + interpolators * cornerSamples * 3;
        int end = beardBase + blocks * 2;
        words.put(0, chunks);
        words.put(1, first.beard().length);
        words.put(2, cornersPerChunk);
        words.put(3, first.cellWidth());
        words.put(4, first.cellHeight());
        words.put(5, first.height());
        words.put(6, cornerBase);
        words.put(7, beardBase);
        words.put(8, 16);
        words.put(9, first.outer().nodeCount());
        words.put(10, first.outer().leafCount());
        words.put(16, outerProgram, 0, outerProgram.length);
        for (int chunk = 0; chunk < chunks; chunk++) {
            double[] beard = cases.get(chunk).beard();
            for (int i = 0; i < beard.length; i++) {
                float hi = (float)beard[i];
                int at = beardBase + (chunk * beard.length + i) * 2;
                words.put(at, Float.floatToRawIntBits(hi));
                words.put(at + 1, Float.floatToRawIntBits((float)(beard[i] - hi)));
            }
        }
        return new DensityBatchLayout(cornerOffsets, cornerSamples, blocks, end);
    }

}
