package dev.vulkanchunk;

import java.util.List;

record DensityBatchLayout(int[] words, int[] cornerOffsets,
                         int cornerSamples, int blocks) {
    static DensityBatchLayout assemble(List<StagedDensityBatch.Case> cases) {
        StagedDensityBatch.Case first = cases.getFirst();
        int chunks = cases.size();
        int cornersPerChunk = (16 / first.cellWidth() + 1) * (16 / first.cellWidth() + 1)
                * (first.height() / first.cellHeight() + 1);
        int cornerSamples = chunks * cornersPerChunk;
        int blocks = chunks * first.beard().length;
        int[] outerBatch = first.outer().batch(new double[][]{new double[first.outer().leafCount()]}, 0, 1);
        int outerWords = first.outer().nodeCount() * 10;
        int cornerBase = 16 + outerWords;
        int interpolators = first.inner().length;
        int[] cornerOffsets = new int[interpolators];
        for (int leaf = 0; leaf < interpolators; leaf++)
            cornerOffsets[leaf] = cornerBase + leaf * cornerSamples * 3;
        int beardBase = cornerBase + interpolators * cornerSamples * 3;
        int[] words = new int[beardBase + blocks * 2];
        words[0] = chunks;
        words[1] = first.beard().length;
        words[2] = cornersPerChunk;
        words[3] = first.cellWidth();
        words[4] = first.cellHeight();
        words[5] = first.height();
        words[6] = cornerBase;
        words[7] = beardBase;
        words[8] = 16;
        words[9] = first.outer().nodeCount();
        words[10] = first.outer().leafCount();
        System.arraycopy(outerBatch, 5, words, 16, outerWords);
        for (int chunk = 0; chunk < chunks; chunk++) {
            double[] beard = cases.get(chunk).beard();
            for (int i = 0; i < beard.length; i++) {
                float hi = (float)beard[i];
                int at = beardBase + (chunk * beard.length + i) * 2;
                words[at] = Float.floatToRawIntBits(hi);
                words[at + 1] = Float.floatToRawIntBits((float)(beard[i] - hi));
            }
        }
        return new DensityBatchLayout(words, cornerOffsets, cornerSamples, blocks);
    }

}
