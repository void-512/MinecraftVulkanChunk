# vulkanchunk

`vulkanchunk` accelerates validated noise-density graphs in Minecraft 1.21.1 chunk terrain generation on NeoForge 21.1.219. It runs grouped noise, coordinate, spline, lattice, interpolation, and final-density stages on a Vulkan compute device. Minecraft still performs aquifer decisions, ore/material selection, block placement, and later world-generation stages.

The mod checks the actual generator, fully resolved final-density graph, and noise cell layout against the validated Overworld, Nether, and End plans. With empty old-chunk blending, a compatible workload activates Vulkan automatically, including AoA3 Precasia's Overworld-compatible graph. Other graphs and custom generators retain their original CPU generation with a dimension-specific warning or status reason. If Vulkan initialization or a runtime job fails, the log says why and subsequent chunks use the original CPU path.

Use `/vulkanchunk status` in game, or `vulkanchunk status` in the server console, to see the active backend, device, driver/API details, batch capacity, submitted/completed/used chunk counts, and each exercised dimension's selected density plan or CPU fallback reason. Up to four ready chunk requests share one staged submission.

## Build

Java 21 or newer is required. The checked-in SPIR-V resources are packaged directly; building the mod does not require a shader compiler. The JAR includes LWJGL core, Vulkan bindings, and LWJGL core natives for Linux ARM64 and Windows x86_64. The operating system's Vulkan loader and driver must be installed. The same `build/libs/vulkanchunk-0.1.0.jar` is used on both platforms.

Linux:

```sh
./gradlew build
```

Windows PowerShell:

```powershell
.\gradlew.bat build
Get-Item .\build\libs\vulkanchunk-0.1.0.jar
```

The GLSL sources are in `src/main/shaders`. When changing a shader, regenerate its matching `src/main/resources/*.spv` with `glslc --target-env=vulkan1.1` and commit both. No shader arithmetic was changed for Windows.

## First Windows server check

Install NeoForge 21.1.219 for Minecraft 1.21.1 in a Windows server directory. From PowerShell, set the project and server paths, then copy the JAR and start the server:

```powershell
$Project = 'C:\path\to\vulkanchunk'
$Server = 'C:\path\to\neoforge-server'
Copy-Item "$Project\build\libs\vulkanchunk-0.1.0.jar" "$Server\mods\" -Force
Set-Location $Server
java -Xmx4G -Dorg.lwjgl.system.stackSize=512 -Dvulkanchunk.validateChunks=4 -Dvulkanchunk.checkEndIslands=4 --enable-native-access=ALL-UNNAMED '@libraries/net/neoforged/neoforge/21.1.219/win_args.txt' nogui
```

The stack-size property is set before NeoForge loads LWJGL and covers drivers that enumerate many Vulkan extensions. After terrain generates, run `vulkanchunk status` in the server console (or `/vulkanchunk status` in game). It should report `backend: Vulkan`, `state=VULKAN_ACTIVE`, the selected GPU, and active density plans. The validation properties compare the next four GPU-filled chunks against CPU block states and check End-island corners when End chunks are among them. Generate fresh Overworld, Nether, and End terrain, including negative and distant coordinates; inspect logs for validator mismatches. Omit both properties for normal operation. `-Dvulkanchunk.forceCpu=true` exercises the visible CPU fallback path.

If Vulkan cannot initialize, status reports `backend: CPU fallback` with the actual native, loader, device, or pipeline failure reason. Hardware correctness on Windows still requires a real server run.

See [PRECISION.md](PRECISION.md) for the selective double-single corrections that must remain intact.