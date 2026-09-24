package dev.vulkanchunk;

/** Integer coordinates for the same cell-corner lattice used by NoiseChunk interpolation. */
public record NoiseSampleGrid(int[] coordinates, int chunks, int samplesPerChunk,
                              int cellWidth, int cellHeight, int minY, int height,
                              long preparationNanos) {
    public static NoiseSampleGrid create(int originChunkX, int originChunkZ, int chunks,
                                         int minY, int height, int cellWidth, int cellHeight) {
        long start = System.nanoTime();
        int horizontal = 16 / cellWidth + 1;
        int vertical = height / cellHeight + 1;
        int samplesPerChunk = Math.multiplyExact(horizontal * horizontal, vertical);
        int[] coordinates = new int[Math.multiplyExact(Math.multiplyExact(chunks, samplesPerChunk), 3)];
        int rowWidth = (int) Math.ceil(Math.sqrt(chunks));
        int output = 0;
        for (int chunk = 0; chunk < chunks; chunk++) {
            int chunkX = originChunkX + chunk % rowWidth;
            int chunkZ = originChunkZ + chunk / rowWidth;
            int baseX = Math.multiplyExact(chunkX, 16);
            int baseZ = Math.multiplyExact(chunkZ, 16);
            for (int gridX = 0; gridX < horizontal; gridX++) {
                for (int gridZ = 0; gridZ < horizontal; gridZ++) {
                    for (int gridY = 0; gridY < vertical; gridY++) {
                        coordinates[output++] = baseX + gridX * cellWidth;
                        coordinates[output++] = minY + gridY * cellHeight;
                        coordinates[output++] = baseZ + gridZ * cellWidth;
                    }
                }
            }
        }
        return new NoiseSampleGrid(coordinates, chunks, samplesPerChunk, cellWidth, cellHeight,
                minY, height, System.nanoTime() - start);
    }

    public int sampleCount() { return coordinates.length / 3; }
}
