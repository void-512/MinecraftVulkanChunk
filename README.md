# Vulkan Chunk research workspace

This repository contains `vulkanchunk`, a NeoForge 1.21.1 mod that uses Vulkan compute for selected chunk terrain density calculations. The mod source, build instructions, platform notes, and license are in [vulkanchunk](vulkanchunk/README.md).

## Build the mod

Use Java 21 or newer. From `vulkanchunk/`, run `./gradlew build` on Linux or `./gradlew.bat build` on Windows. The output is `vulkanchunk/build/libs/vulkanchunk-0.1.0.jar`. The required SPIR-V shaders are included in the source tree, so a shader compiler is not needed for a normal build.

## Research record

The [prompts](prompts/) directory contains the research and implementation requests. The `prompt*-report.md` files at the repository root document results from the corresponding stages, including GPU validation and remaining limits. [PRECISION.md](vulkanchunk/PRECISION.md) describes the double-single arithmetic used by the shaders.

## Local server files

The Minecraft server, downloaded libraries and mods, world saves, logs, player lists, and local server settings are excluded from Git. To run the mod, set up a separate NeoForge 21.1.219 server for Minecraft 1.21.1 and copy the built mod JAR into its `mods` directory. See the [mod README](vulkanchunk/README.md) for a first server check.
