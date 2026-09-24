package dev.vulkanchunk.mixin;

import dev.vulkanchunk.ProductionDensity;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.NoiseChunk;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.ModifyArg;

@Mixin(NoiseChunk.class)
public abstract class NoiseChunkProductionMixin {
    @ModifyArg(method = "<init>", at = @At(value = "INVOKE", target =
            "Lnet/minecraft/world/level/levelgen/DensityFunctions;cacheAllInCell(Lnet/minecraft/world/level/levelgen/DensityFunction;)Lnet/minecraft/world/level/levelgen/DensityFunction;"))
    private DensityFunction vulkanchunk$overrideDensity(DensityFunction original) {
        return ProductionDensity.override((NoiseChunk)(Object)this, original);
    }
}
