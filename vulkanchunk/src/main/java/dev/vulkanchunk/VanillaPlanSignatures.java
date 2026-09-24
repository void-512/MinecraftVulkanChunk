package dev.vulkanchunk;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.mojang.serialization.JsonOps;
import net.minecraft.core.Registry;
import net.minecraft.core.RegistryAccess;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.RegistryOps;
import net.minecraft.resources.ResourceKey;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.NoiseSettings;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.Map;

/** Hashes the fully resolved runtime graph against canonical Minecraft 1.21.1 resources. */
final class VanillaPlanSignatures {
    // SHA-256 of sorted, numeric-normalized JSON {noise, final_density}, recursively expanding
    // density-function references from the 1.21.1-20240808.144430 bundled worldgen resources.
    private static final Map<String, String> EXPECTED = Map.of(
            "overworld", "86e1ac2feaff5ae72838f6b7ed87bb37fb7f19798e26f37eeca2c64871639b0a",
            "nether", "888e9409abfca1274112b3a24b00a4b89ec3621b3540d3b6f6eb2c77695378e3",
            "end", "1f5df10fee4d1ef2c6190490c26e22b700a0bcf0926de98e0b988d6d3f598b7a");

    private final Registry<DensityFunction> densityRegistry;
    private final RegistryOps<JsonElement> registryOps;

    VanillaPlanSignatures(RegistryAccess access) {
        densityRegistry = access.registryOrThrow(Registries.DENSITY_FUNCTION);
        registryOps = RegistryOps.create(JsonOps.INSTANCE, access);
    }

    boolean matches(String plan, String fingerprint) {
        String expected = EXPECTED.get(plan);
        return expected != null && expected.equals(fingerprint);
    }

    String fingerprint(NoiseGeneratorSettings settings) {
        try { return encodeFingerprint(settings); }
        catch (RuntimeException failure) {
            throw new UnsupportedOperationException("cannot inspect runtime final-density graph: "
                    + failure.getMessage(), failure);
        }
    }

    private String encodeFingerprint(NoiseGeneratorSettings settings) {
        JsonElement noise = NoiseSettings.CODEC.encodeStart(JsonOps.INSTANCE, settings.noiseSettings()).result()
                .orElseThrow(() -> new IllegalStateException("Cannot encode noise cell layout"));
        JsonElement root = DensityFunction.HOLDER_HELPER_CODEC
                .encodeStart(registryOps, settings.noiseRouter().finalDensity()).result()
                .orElseThrow(() -> new IllegalStateException("Cannot encode final-density graph"));
        JsonObject tree = new JsonObject();
        tree.add("noise", noise);
        tree.add("final_density", expand(root, new HashMap<>()));
        StringBuilder canonical = new StringBuilder();
        canonical(tree, canonical);
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(canonical.toString().getBytes(StandardCharsets.UTF_8));
            return java.util.HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); }
    }

    private JsonElement expand(JsonElement tree, Map<ResourceLocation, JsonElement> cache) {
        if (tree.isJsonObject()) {
            JsonObject result = new JsonObject();
            for (var entry : tree.getAsJsonObject().entrySet()) result.add(entry.getKey(), expand(entry.getValue(), cache));
            return result;
        }
        if (tree.isJsonArray()) {
            var result = new com.google.gson.JsonArray();
            for (JsonElement item : tree.getAsJsonArray()) result.add(expand(item, cache));
            return result;
        }
        if (tree.isJsonPrimitive() && tree.getAsJsonPrimitive().isString()) {
            ResourceLocation id = ResourceLocation.tryParse(tree.getAsString());
            if (id != null) {
                if (cache.containsKey(id)) return cache.get(id);
                var value = densityRegistry.getOptional(ResourceKey.create(Registries.DENSITY_FUNCTION, id));
                if (value.isPresent()) {
                    JsonElement encoded = DensityFunction.DIRECT_CODEC.encodeStart(registryOps, value.get()).result()
                            .orElseThrow(() -> new IllegalStateException("Cannot encode density function " + id));
                    JsonElement expanded = expand(encoded, cache);
                    cache.put(id, expanded);
                    return expanded;
                }
            }
        }
        return tree;
    }

    private static void canonical(JsonElement tree, StringBuilder result) {
        if (tree.isJsonObject()) {
            result.append('{');
            boolean first = true;
            for (String key : tree.getAsJsonObject().keySet().stream().sorted().toList()) {
                if (!first) result.append(',');
                first = false;
                result.append(new JsonPrimitive(key)).append(':');
                canonical(tree.getAsJsonObject().get(key), result);
            }
            result.append('}');
        } else if (tree.isJsonArray()) {
            result.append('[');
            boolean first = true;
            for (JsonElement item : tree.getAsJsonArray()) {
                if (!first) result.append(',');
                first = false;
                canonical(item, result);
            }
            result.append(']');
        } else if (tree.isJsonPrimitive() && tree.getAsJsonPrimitive().isNumber()) {
            result.append(tree.getAsBigDecimal().stripTrailingZeros().toPlainString());
        } else result.append(tree);
    }
}
