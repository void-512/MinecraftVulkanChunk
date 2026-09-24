package dev.vulkanchunk.mixin;

import it.unimi.dsi.fastutil.doubles.DoubleList;
import net.minecraft.world.level.levelgen.synth.ImprovedNoise;
import net.minecraft.world.level.levelgen.synth.PerlinNoise;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(PerlinNoise.class)
public interface PerlinNoiseAccessor {
    @Accessor("noiseLevels")
    ImprovedNoise[] vulkanchunk$noiseLevels();

    @Accessor("amplitudes")
    DoubleList vulkanchunk$amplitudes();

    @Accessor("lowestFreqInputFactor")
    double vulkanchunk$lowestFreqInputFactor();

    @Accessor("lowestFreqValueFactor")
    double vulkanchunk$lowestFreqValueFactor();
}
