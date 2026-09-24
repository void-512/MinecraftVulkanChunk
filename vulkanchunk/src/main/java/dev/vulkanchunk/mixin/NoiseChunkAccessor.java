package dev.vulkanchunk.mixin;

import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.NoiseChunk;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(NoiseChunk.class)
public interface NoiseChunkAccessor {
    @Invoker("getInterpolatedState")
    BlockState vulkanchunk$getInterpolatedState();

    @Invoker("wrap")
    DensityFunction vulkanchunk$wrap(DensityFunction function);

    @org.spongepowered.asm.mixin.gen.Accessor("beardifier")
    net.minecraft.world.level.levelgen.DensityFunctions.BeardifierOrMarker vulkanchunk$beardifier();

}
