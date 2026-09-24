package dev.vulkanchunk.mixin;

import dev.vulkanchunk.ProductionDensity;
import dev.vulkanchunk.SchedulingTrace;
import net.minecraft.world.level.StructureManager;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.levelgen.NoiseBasedChunkGenerator;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(NoiseBasedChunkGenerator.class)
public abstract class NoiseBasedChunkGeneratorMixin {
    @Inject(method = "fillFromNoise", at = @At("HEAD"))
    private void vulkanchunk$requested(Blender blender, RandomState state, StructureManager structures,
                                      ChunkAccess chunk, CallbackInfoReturnable<java.util.concurrent.CompletableFuture<ChunkAccess>> callback) {
        SchedulingTrace.invoked(chunk);
    }

    @Inject(method = "doFill", at = @At("HEAD"))
    private void vulkanchunk$begin(Blender blender, StructureManager structures, RandomState state,
                                            ChunkAccess chunk, int minCellY, int cellCountY,
                                            CallbackInfoReturnable<ChunkAccess> callback) {
        ProductionDensity.begin((NoiseBasedChunkGenerator)(Object)this, blender, structures, state,
                chunk, minCellY, cellCountY);
        ProductionDensity.afterBegin();
    }

    @Inject(method = "doFill", at = @At("RETURN"))
    private void vulkanchunk$finish(Blender blender, StructureManager structures, RandomState state,
                                             ChunkAccess chunk, int minCellY, int cellCountY,
                                             CallbackInfoReturnable<ChunkAccess> callback) {
        ProductionDensity.end((NoiseBasedChunkGenerator)(Object)this, blender, structures, state, chunk);
    }
}
