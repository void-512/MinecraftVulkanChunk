package dev.vulkanchunk;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkApplicationInfo;
import org.lwjgl.vulkan.VkBufferCopy;
import org.lwjgl.vulkan.VkBufferCreateInfo;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkCommandPoolCreateInfo;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorBufferInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkDeviceCreateInfo;
import org.lwjgl.vulkan.VkDeviceQueueCreateInfo;
import org.lwjgl.vulkan.VkFenceCreateInfo;
import org.lwjgl.vulkan.VkInstance;
import org.lwjgl.vulkan.VkInstanceCreateInfo;
import org.lwjgl.vulkan.VkMemoryAllocateInfo;
import org.lwjgl.vulkan.VkMemoryBarrier;
import org.lwjgl.vulkan.VkMemoryRequirements;
import org.lwjgl.vulkan.VkMappedMemoryRange;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkPhysicalDeviceFeatures;
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkQueryPoolCreateInfo;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkQueueFamilyProperties;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkSubmitInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.vulkan.VK10.*;
import static org.lwjgl.vulkan.VK11.*;
import static org.lwjgl.vulkan.VK12.VK_API_VERSION_1_2;

/**
 * Minimal headless Vulkan compute context. Buffers, descriptor sets, pipelines,
 * command objects and synchronization primitives are persistent and reused.
 */
public final class VulkanComputeBackend implements AutoCloseable {
    static {
        // LWJGL enumerates all device extensions while constructing VkInstance.
        // Desktop drivers can exceed the default 64 KiB thread-local stack.
        org.lwjgl.system.Configuration.STACK_SIZE.set(512);
    }

    private static final org.slf4j.Logger LOGGER = com.mojang.logging.LogUtils.getLogger();
    public static final int MAX_INPUT_INTS = 4 * 1024 * 1024;
    public static final int MAX_OUTPUT_INTS = 2 * 1024 * 1024;
    private static final long INPUT_BYTES = (long) MAX_INPUT_INTS * Integer.BYTES;
    private static final long OUTPUT_BYTES = (long) MAX_OUTPUT_INTS * Integer.BYTES;
    private static final String OUTPUT_STRATEGY = System.getProperty("vulkanchunk.outputStrategy", "auto");
    private static final boolean MEASURE_FILL = Boolean.getBoolean("vulkanchunk.measureFill");

    private VkInstance instance;
    private VkPhysicalDevice physicalDevice;
    private VkDevice device;
    private VkQueue computeQueue;
    private int queueFamilyIndex = -1;

    private long descriptorSetLayout;
    private long descriptorPool;
    private long pipelineLayout;
    private long doubleSingleNoisePipeline;
    private long densityGraphPipeline;
    private long densitySplinePipeline;
    private long densityFinalFusedPipeline;
    private long commandPool;
    private final Slot[] slots = {new Slot(), new Slot()};
    private boolean stagedOutput;
    private boolean hostCachedOutput;
    private String outputPolicy;
    private long nonCoherentAtomSize;
    private long uploadedInputBytes;
    private float timestampPeriod;
    private int timestampValidBits;

    private String deviceName;
    private String driverName;
    private String driverInfo;
    private boolean shaderFloat64;
    private boolean closed;

    public VulkanComputeBackend() {
        try {
            initialize();
        } catch (Throwable failure) {
            close();
            throw failure instanceof RuntimeException runtime ? runtime : new IllegalStateException(failure);
        }
    }

    Slot slot(int index) { return slots[index]; }

    public synchronized void submitStagedDensity(Slot slot, StagedDensityBatch batch) {
        if (batch.wordCount() > MAX_INPUT_INTS || batch.blocks() * 3 > MAX_OUTPUT_INTS)
            throw new IllegalArgumentException("Staged density batch exceeds Vulkan buffers");
        try {
            if (doubleSingleNoisePipeline == 0) doubleSingleNoisePipeline = createPipeline("/normal_noise_ds.spv");
            if (densityGraphPipeline == 0) densityGraphPipeline = createPipeline("/density_graph.spv");
            if (densitySplinePipeline == 0) densitySplinePipeline = createPipeline("/density_spline.spv");
            if (densityFinalFusedPipeline == 0) densityFinalFusedPipeline = createPipeline("/density_final_fused.spv");
        } catch (IOException failure) {
            throw new IllegalStateException("Could not create staged density pipelines", failure);
        }
        slot.batch = batch;
        slot.totalStart = System.nanoTime();
        long uploadStart = System.nanoTime();
        uploadStaged(slot, batch);
        flushIfNeeded(slot.inputBuffer, (long) batch.wordCount() * Integer.BYTES);
        slot.uploadNanos = System.nanoTime() - uploadStart;
        long commandStart = System.nanoTime();
        recordStagedDensity(slot, batch);
        slot.commandNanos = System.nanoTime() - commandStart;
        long submitStart = System.nanoTime();
        submitCommand(slot);
        slot.submitNanos = System.nanoTime() - submitStart;
    }

    public synchronized boolean completed(Slot slot) {
        int result = vkGetFenceStatus(device, slot.fence);
        if (result == VK_SUCCESS) return true;
        if (result == VK_NOT_READY) return false;
        check(result, "vkGetFenceStatus(staged density)");
        return false;
    }

    public synchronized boolean waitForCompletion(Slot slot, long timeoutNanos) {
        int result = vkWaitForFences(device, slot.fence, true, timeoutNanos);
        if (result == VK_SUCCESS) return true;
        if (result == VK_TIMEOUT) return false;
        check(result, "vkWaitForFences(staged density)");
        return false;
    }

    public synchronized ComputeResult finishStagedDensity(Slot slot, long waitTimeout) {
        StagedDensityBatch batch = slot.batch;
        long waitStart = System.nanoTime();
        check(vkWaitForFences(device, slot.fence, true, waitTimeout), "vkWaitForFences(staged density)");
        long waitNanos = System.nanoTime() - waitStart;
        if (!batch.endIslandChecks().isEmpty()) {
            invalidateIfNeeded(slot.inputBuffer, (long) batch.wordCount() * Integer.BYTES);
            checkEndIslands(slot, batch.endIslandChecks());
        }
        GpuTiming gpuTiming = readGpuTimestamp(slot, batch);
        long readbackStart = System.nanoTime();
        long invalidateStart = System.nanoTime();
        BufferAllocation hostOutput = stagedOutput ? slot.readbackBuffer : slot.outputBuffer;
        long outputBytes = (long) batch.blocks() * 3 * Integer.BYTES;
        invalidateIfNeeded(hostOutput, outputBytes);
        long invalidateNanos = System.nanoTime() - invalidateStart;
        IntBuffer output = hostOutput.mapped.duplicate().order(ByteOrder.nativeOrder()).asIntBuffer();
        long readbackNanos = System.nanoTime() - readbackStart;
        ComputeResult result = new ComputeResult(output, slot.uploadNanos, slot.commandNanos, slot.submitNanos,
                waitNanos, gpuTiming.computeNanos(), gpuTiming.copyNanos(), gpuTiming.stageNanos(), invalidateNanos,
                0, 0, outputBytes, readbackNanos, System.nanoTime() - slot.totalStart);
        slot.batch = null;
        return result;
    }

    private void uploadStaged(Slot slot, StagedDensityBatch batch) {
        int[] words = slot.workspace.words.array();
        IntBuffer mapped = slot.inputBuffer.mapped.duplicate().order(ByteOrder.nativeOrder()).asIntBuffer();
        long changed = 0;
        for (int i = 0; i < batch.wordCount(); i++) {
            if (slot.workspace.scratch.get(i)) {
                slot.residentValid[i] = false;
            } else if (!slot.residentValid[i] || slot.residentInput[i] != words[i]) {
                mapped.put(i, words[i]);
                slot.residentInput[i] = words[i];
                slot.residentValid[i] = true;
                changed++;
            }
        }
        uploadedInputBytes += changed * Integer.BYTES;
    }

    private void checkEndIslands(Slot slot, List<StagedDensityBatch.EndIslandCheck> checks) {
        IntBuffer arena = slot.inputBuffer.mapped.duplicate().order(ByteOrder.nativeOrder()).asIntBuffer();
        int differences = 0, signDifferences = 0;
        double maxError = 0;
        String first = "none";
        for (var check : checks) {
            double actual = (double)Float.intBitsToFloat(arena.get(check.offset()))
                    + Float.intBitsToFloat(arena.get(check.offset() + 1));
            boolean equal = actual == check.expected()
                    || (Double.isNaN(actual) && Double.isNaN(check.expected()));
            double error = equal ? 0.0 : Math.abs(actual - check.expected());
            if (!equal) {
                differences++;
                if (first.equals("none")) first = check.x() + "," + check.z()
                        + " expected=" + check.expected() + " actual=" + actual;
            }
            if ((actual > 0) != (check.expected() > 0)) signDifferences++;
            maxError = Math.max(maxError, error);
        }
        LOGGER.info("[vulkanchunk] End-island GPU/Java corners: checked={} unequal={} signDifferences={} maxAbsError={} first={}",
                checks.size(), differences, signDifferences, maxError, first);
    }

    public synchronized void prepareStagedDensity() {
        try {
            if (doubleSingleNoisePipeline == 0) doubleSingleNoisePipeline = createPipeline("/normal_noise_ds.spv");
            if (densityGraphPipeline == 0) densityGraphPipeline = createPipeline("/density_graph.spv");
            if (densitySplinePipeline == 0) densitySplinePipeline = createPipeline("/density_spline.spv");
            if (densityFinalFusedPipeline == 0) densityFinalFusedPipeline = createPipeline("/density_final_fused.spv");
        } catch (IOException failure) {
            throw new IllegalStateException("Could not create staged density pipelines", failure);
        }
    }

    public String deviceName() {
        return deviceName;
    }

    public String driverName() {
        return driverName;
    }

    public String driverInfo() {
        return driverInfo;
    }

    public boolean shaderFloat64Supported() {
        return shaderFloat64;
    }

    public long uploadedInputBytes() { return uploadedInputBytes; }

    public String outputPolicy() { return outputPolicy; }

    private void initialize() throws IOException {
        createInstance();
        selectComputeDevice();
        createDevice();
        chooseOutputPolicy();
        for (Slot slot : slots) {
            slot.inputBuffer = createBuffer(INPUT_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
            if (stagedOutput) {
                slot.outputBuffer = createBuffer(OUTPUT_BYTES,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, false);
                slot.readbackBuffer = createBuffer(OUTPUT_BYTES, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT
                                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
            } else {
                int preferred = OUTPUT_STRATEGY.equals("direct") ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                slot.outputBuffer = createBuffer(OUTPUT_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                | (hostCachedOutput ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0),
                        preferred, true);
            }
            slot.workspace = new StagedDensityBatch.Workspace(IntBuffer.wrap(new int[MAX_INPUT_INTS]));
        }
        LOGGER.info("[vulkanchunk] output policy={} input type={} output type={} readback type={}",
                outputPolicy, slots[0].inputBuffer.memoryType, slots[0].outputBuffer.memoryType,
                slots[0].readbackBuffer == null ? "none" : slots[0].readbackBuffer.memoryType);
        createDescriptors();
        createPipelines();
        createCommandObjects();
    }

    private void createInstance() {
        try (MemoryStack stack = stackPush()) {
            VkApplicationInfo appInfo = VkApplicationInfo.calloc(stack)
                    .sType$Default()
                    .pApplicationName(stack.UTF8("Minecraft Vulkan Compute"))
                    .applicationVersion(VK_MAKE_VERSION(0, 1, 0))
                    .pEngineName(stack.UTF8("LWJGL"))
                    .engineVersion(VK_MAKE_VERSION(0, 1, 0))
                    .apiVersion(VK_API_VERSION_1_2);
            VkInstanceCreateInfo createInfo = VkInstanceCreateInfo.calloc(stack)
                    .sType$Default()
                    .pApplicationInfo(appInfo);
            PointerBuffer result = stack.mallocPointer(1);
            check(vkCreateInstance(createInfo, null, result), "vkCreateInstance");
            instance = new VkInstance(result.get(0), createInfo);
        }
    }

    private void selectComputeDevice() {
        List<String> rejected = new ArrayList<>();
        String requested = System.getProperty("vulkanchunk.deviceIndex");
        int requestedIndex = requested == null ? -1 : Integer.parseInt(requested);
        if (requestedIndex < -1) throw new IllegalArgumentException("deviceIndex must be nonnegative");
        try (MemoryStack stack = stackPush()) {
            IntBuffer count = stack.ints(0);
            check(vkEnumeratePhysicalDevices(instance, count, null), "vkEnumeratePhysicalDevices(count)");
            if (count.get(0) == 0) {
                throw new IllegalStateException("Vulkan loader returned no physical devices");
            }
            PointerBuffer devices = stack.mallocPointer(count.get(0));
            check(vkEnumeratePhysicalDevices(instance, count, devices), "vkEnumeratePhysicalDevices(list)");

            int selectedType = VK_PHYSICAL_DEVICE_TYPE_OTHER;
            for (int index = 0; index < count.get(0); index++) {
                if (requestedIndex >= 0 && index != requestedIndex) continue;
                VkPhysicalDevice candidate = new VkPhysicalDevice(devices.get(index), instance);
                VkPhysicalDeviceProperties properties = VkPhysicalDeviceProperties.malloc(stack);
                vkGetPhysicalDeviceProperties(candidate, properties);
                String name = properties.deviceNameString();
                int type = properties.deviceType();
                int[] queue = findComputeQueue(candidate, stack);
                var limits = properties.limits();
                String problem = null;
                if (type != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                        && type != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                    problem = "not a hardware GPU";
                } else if (properties.apiVersion() < VK_API_VERSION_1_2) {
                    problem = "Vulkan API below 1.2";
                } else if (queue == null) {
                    problem = "no compute queue";
                } else if (Integer.toUnsignedLong(limits.maxStorageBufferRange()) < INPUT_BYTES
                        || Integer.toUnsignedLong(limits.maxStorageBufferRange()) < OUTPUT_BYTES) {
                    problem = "storage buffer range below 16 MiB";
                } else if (limits.maxComputeWorkGroupInvocations() < 64
                        || limits.maxComputeWorkGroupSize(0) < 64) {
                    problem = "compute workgroup limit below 64";
                } else if (limits.maxPushConstantsSize() < 16
                        || limits.maxPerStageDescriptorStorageBuffers() < 2
                        || limits.maxDescriptorSetStorageBuffers() < 2) {
                    problem = "insufficient push constant or storage descriptor limits";
                }
                if (problem != null) {
                    rejected.add(name + " (" + problem + ")");
                    continue;
                }
                if (physicalDevice != null && (selectedType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                        || type != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) {
                    continue;
                }
                VkPhysicalDeviceFeatures features = VkPhysicalDeviceFeatures.malloc(stack);
                vkGetPhysicalDeviceFeatures(candidate, features);
                physicalDevice = candidate;
                selectedType = type;
                queueFamilyIndex = queue[0];
                timestampValidBits = queue[1];
                timestampPeriod = limits.timestampPeriod();
                nonCoherentAtomSize = limits.nonCoherentAtomSize();
                deviceName = name;
                driverName = "vendor 0x" + Integer.toHexString(properties.vendorID());
                driverInfo = "driver 0x" + Integer.toHexString(properties.driverVersion())
                        + ", Vulkan API " + (properties.apiVersion() >>> 22)
                        + "." + ((properties.apiVersion() >>> 12) & 0x3ff);
                shaderFloat64 = features.shaderFloat64();
            }
        }
        if (physicalDevice == null) {
            throw new IllegalStateException("No suitable hardware Vulkan compute device; rejected: "
                    + String.join(", ", rejected) + (requested == null ? "" : "; deviceIndex=" + requested));
        }
        logMemoryTopology();
    }

    private int[] findComputeQueue(VkPhysicalDevice candidate, MemoryStack stack) {
        IntBuffer count = stack.ints(0);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, count, null);
        VkQueueFamilyProperties.Buffer families = VkQueueFamilyProperties.calloc(count.get(0), stack);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, count, families);
        for (int index = 0; index < families.capacity(); index++) {
            VkQueueFamilyProperties family = families.get(index);
            if (family.queueCount() > 0 && (family.queueFlags() & VK_QUEUE_COMPUTE_BIT) != 0) {
                return new int[] { index, family.timestampValidBits() };
            }
        }
        return null;
    }
    private void createDevice() {
        try (MemoryStack stack = stackPush()) {
            VkDeviceQueueCreateInfo.Buffer queueInfo = VkDeviceQueueCreateInfo.calloc(1, stack);
            queueInfo.get(0)
                    .sType$Default()
                    .queueFamilyIndex(queueFamilyIndex)
                    .pQueuePriorities(stack.floats(1.0f));
            VkDeviceCreateInfo createInfo = VkDeviceCreateInfo.calloc(stack)
                    .sType$Default()
                    .pQueueCreateInfos(queueInfo);
            PointerBuffer result = stack.mallocPointer(1);
            check(vkCreateDevice(physicalDevice, createInfo, null, result), "vkCreateDevice");
            device = new VkDevice(result.get(0), physicalDevice, createInfo);
            PointerBuffer queue = stack.mallocPointer(1);
            vkGetDeviceQueue(device, queueFamilyIndex, 0, queue);
            computeQueue = new VkQueue(queue.get(0), device);
        }
    }

    private void logMemoryTopology() {
        try (MemoryStack stack = stackPush()) {
            VkPhysicalDeviceMemoryProperties properties = VkPhysicalDeviceMemoryProperties.malloc(stack);
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, properties);
            LOGGER.info("[vulkanchunk] Vulkan memory: heaps={} types={} atom={}",
                    properties.memoryHeapCount(), properties.memoryTypeCount(), nonCoherentAtomSize);
            for (int i = 0; i < properties.memoryHeapCount(); i++)
                LOGGER.info("[vulkanchunk] memory heap {}: bytes={} flags=0x{}", i,
                        properties.memoryHeaps(i).size(),
                        Integer.toHexString(properties.memoryHeaps(i).flags()));
            for (int i = 0; i < properties.memoryTypeCount(); i++)
                LOGGER.info("[vulkanchunk] memory type {}: heap={} flags=0x{}", i,
                        properties.memoryTypes(i).heapIndex(),
                        Integer.toHexString(properties.memoryTypes(i).propertyFlags()));
        }
    }

    private void chooseOutputPolicy() {
        if (!OUTPUT_STRATEGY.equals("auto") && !OUTPUT_STRATEGY.equals("direct")
                && !OUTPUT_STRATEGY.equals("cached") && !OUTPUT_STRATEGY.equals("staged"))
            throw new IllegalArgumentException("Invalid vulkanchunk.outputStrategy=" + OUTPUT_STRATEGY);
        int directBits = bufferMemoryTypeBits(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        int deviceBits = bufferMemoryTypeBits(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        int stagingBits = bufferMemoryTypeBits(VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        boolean cachedDirectAvailable = hasMemoryType(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, directBits);
        hostCachedOutput = OUTPUT_STRATEGY.equals("cached")
                || (OUTPUT_STRATEGY.equals("auto") && cachedDirectAvailable);
        stagedOutput = OUTPUT_STRATEGY.equals("staged") || (OUTPUT_STRATEGY.equals("auto")
                && !cachedDirectAvailable
                && hasMemoryType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, deviceBits)
                && hasMemoryType(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                        stagingBits));
        outputPolicy = stagedOutput ? "device-local + host staging"
                : hostCachedOutput ? "direct host-cached" : "direct host-visible";
    }

    private int bufferMemoryTypeBits(int usage) {
        try (MemoryStack stack = stackPush()) {
            VkBufferCreateInfo info = VkBufferCreateInfo.calloc(stack).sType$Default()
                    .size(OUTPUT_BYTES).usage(usage).sharingMode(VK_SHARING_MODE_EXCLUSIVE);
            LongBuffer result = stack.mallocLong(1);
            check(vkCreateBuffer(device, info, null, result), "vkCreateBuffer(memory policy)");
            long buffer = result.get(0);
            try {
                VkMemoryRequirements requirements = VkMemoryRequirements.malloc(stack);
                vkGetBufferMemoryRequirements(device, buffer, requirements);
                return requirements.memoryTypeBits();
            } finally {
                vkDestroyBuffer(device, buffer, null);
            }
        }
    }

    private boolean hasMemoryType(int flags, int allowedBits) {
        try (MemoryStack stack = stackPush()) {
            VkPhysicalDeviceMemoryProperties properties = VkPhysicalDeviceMemoryProperties.malloc(stack);
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, properties);
            for (int i = 0; i < properties.memoryTypeCount(); i++)
                if ((allowedBits & (1 << i)) != 0
                        && (properties.memoryTypes(i).propertyFlags() & flags) == flags) return true;
            return false;
        }
    }

    private BufferAllocation createBuffer(long size, int usage, int requiredFlags,
                                          int preferredFlags, boolean map) {
        try (MemoryStack stack = stackPush()) {
            VkBufferCreateInfo createInfo = VkBufferCreateInfo.calloc(stack)
                    .sType$Default()
                    .size(size)
                    .usage(usage)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
            LongBuffer bufferResult = stack.mallocLong(1);
            check(vkCreateBuffer(device, createInfo, null, bufferResult), "vkCreateBuffer");
            long buffer = bufferResult.get(0);
            long memory = 0;
            boolean mappedMemory = false;
            try {
                VkMemoryRequirements requirements = VkMemoryRequirements.malloc(stack);
                vkGetBufferMemoryRequirements(device, buffer, requirements);
                int memoryType = findMemoryType(requirements.memoryTypeBits(), requiredFlags,
                        preferredFlags, stack);
                VkPhysicalDeviceMemoryProperties properties = VkPhysicalDeviceMemoryProperties.malloc(stack);
                vkGetPhysicalDeviceMemoryProperties(physicalDevice, properties);
                int flags = properties.memoryTypes(memoryType).propertyFlags();
                VkMemoryAllocateInfo allocateInfo = VkMemoryAllocateInfo.calloc(stack)
                        .sType$Default()
                        .allocationSize(requirements.size())
                        .memoryTypeIndex(memoryType);
                LongBuffer memoryResult = stack.mallocLong(1);
                check(vkAllocateMemory(device, allocateInfo, null, memoryResult), "vkAllocateMemory");
                memory = memoryResult.get(0);
                check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
                ByteBuffer mapped = null;
                if (map) {
                    PointerBuffer mappedResult = stack.mallocPointer(1);
                    check(vkMapMemory(device, memory, 0, requirements.size(), 0, mappedResult), "vkMapMemory");
                    mappedMemory = true;
                    mapped = MemoryUtil.memByteBuffer(mappedResult.get(0), Math.toIntExact(size))
                            .order(ByteOrder.nativeOrder());
                }
                return new BufferAllocation(buffer, memory, mapped, memoryType, flags, requirements.size());
            } catch (RuntimeException failure) {
                if (mappedMemory) vkUnmapMemory(device, memory);
                vkDestroyBuffer(device, buffer, null);
                if (memory != 0) vkFreeMemory(device, memory, null);
                throw failure;
            }
        }
    }

    private int findMemoryType(int allowedBits, int requiredFlags, int preferredFlags, MemoryStack stack) {
        VkPhysicalDeviceMemoryProperties properties = VkPhysicalDeviceMemoryProperties.malloc(stack);
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, properties);
        int best = -1, bestScore = -1;
        for (int index = 0; index < properties.memoryTypeCount(); index++) {
            boolean allowed = (allowedBits & (1 << index)) != 0;
            int flags = properties.memoryTypes(index).propertyFlags();
            if (allowed && (flags & requiredFlags) == requiredFlags) {
                int score = Integer.bitCount(flags & preferredFlags);
                if (score > bestScore) { best = index; bestScore = score; }
            }
        }
        if (best < 0) throw new IllegalStateException("No Vulkan memory type with required flags 0x"
                + Integer.toHexString(requiredFlags) + " and allowed bits 0x"
                + Integer.toHexString(allowedBits));
        return best;
    }

    private void flushIfNeeded(BufferAllocation allocation, long bytes) {
        synchronizeMappedMemory(allocation, bytes, true);
    }

    private void invalidateIfNeeded(BufferAllocation allocation, long bytes) {
        synchronizeMappedMemory(allocation, bytes, false);
    }

    private void synchronizeMappedMemory(BufferAllocation allocation, long bytes, boolean flush) {
        if ((allocation.flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0 || bytes == 0) return;
        long atom = Math.max(1, nonCoherentAtomSize);
        long aligned = Math.min(allocation.allocationSize, ((bytes + atom - 1) / atom) * atom);
        try (MemoryStack stack = stackPush()) {
            VkMappedMemoryRange.Buffer range = VkMappedMemoryRange.calloc(1, stack);
            range.get(0).sType$Default().memory(allocation.memory).offset(0).size(aligned);
            check(flush ? vkFlushMappedMemoryRanges(device, range)
                    : vkInvalidateMappedMemoryRanges(device, range),
                    flush ? "vkFlushMappedMemoryRanges" : "vkInvalidateMappedMemoryRanges");
        }
    }

    private void createDescriptors() {
        try (MemoryStack stack = stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(2, stack);
            bindings.get(0).binding(0).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            bindings.get(1).binding(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorSetLayoutCreateInfo layoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(bindings);
            LongBuffer layoutResult = stack.mallocLong(1);
            check(vkCreateDescriptorSetLayout(device, layoutInfo, null, layoutResult), "vkCreateDescriptorSetLayout");
            descriptorSetLayout = layoutResult.get(0);

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(1, stack);
            poolSizes.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(8);
            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(4).pPoolSizes(poolSizes);
            LongBuffer poolResult = stack.mallocLong(1);
            check(vkCreateDescriptorPool(device, poolInfo, null, poolResult), "vkCreateDescriptorPool");
            descriptorPool = poolResult.get(0);

            VkDescriptorSetAllocateInfo allocateInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(descriptorPool)
                    .pSetLayouts(stack.longs(descriptorSetLayout, descriptorSetLayout,
                            descriptorSetLayout, descriptorSetLayout));
            LongBuffer setResult = stack.mallocLong(4);
            check(vkAllocateDescriptorSets(device, allocateInfo, setResult), "vkAllocateDescriptorSets");
            for (int i = 0; i < slots.length; i++) {
                Slot slot = slots[i];
                slot.descriptorSet = setResult.get(i * 2);
                slot.arenaDescriptorSet = setResult.get(i * 2 + 1);
                VkDescriptorBufferInfo.Buffer inputInfo = VkDescriptorBufferInfo.calloc(1, stack);
                inputInfo.get(0).buffer(slot.inputBuffer.buffer).offset(0).range(INPUT_BYTES);
                VkDescriptorBufferInfo.Buffer outputInfo = VkDescriptorBufferInfo.calloc(1, stack);
                outputInfo.get(0).buffer(slot.outputBuffer.buffer).offset(0).range(OUTPUT_BYTES);
                VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(4, stack);
                writes.get(0).sType$Default().dstSet(slot.descriptorSet).dstBinding(0)
                        .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(inputInfo);
                writes.get(1).sType$Default().dstSet(slot.descriptorSet).dstBinding(1)
                        .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(outputInfo);
                writes.get(2).sType$Default().dstSet(slot.arenaDescriptorSet).dstBinding(0)
                        .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(inputInfo);
                writes.get(3).sType$Default().dstSet(slot.arenaDescriptorSet).dstBinding(1)
                        .descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo(inputInfo);
                vkUpdateDescriptorSets(device, writes, null);
            }
        }
    }

    private void createPipelines() throws IOException {
        try (MemoryStack stack = stackPush()) {
            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(16);
            VkPipelineLayoutCreateInfo layoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default()
                    .pSetLayouts(stack.longs(descriptorSetLayout))
                    .pPushConstantRanges(pushRange);
            LongBuffer result = stack.mallocLong(1);
            check(vkCreatePipelineLayout(device, layoutInfo, null, result), "vkCreatePipelineLayout");
            pipelineLayout = result.get(0);
        }
    }

    private long createPipeline(String resource) throws IOException {
        ByteBuffer code = readResource(resource);
        long shaderModule = 0;
        try (MemoryStack stack = stackPush()) {
            VkShaderModuleCreateInfo shaderInfo = VkShaderModuleCreateInfo.calloc(stack)
                    .sType$Default().pCode(code);
            LongBuffer shaderResult = stack.mallocLong(1);
            check(vkCreateShaderModule(device, shaderInfo, null, shaderResult), "vkCreateShaderModule(" + resource + ")");
            shaderModule = shaderResult.get(0);

            VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                    .sType$Default()
                    .stage(VK_SHADER_STAGE_COMPUTE_BIT)
                    .module(shaderModule)
                    .pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer pipelineInfo = VkComputePipelineCreateInfo.calloc(1, stack);
            pipelineInfo.get(0).sType$Default().stage(stage).layout(pipelineLayout);
            LongBuffer pipelineResult = stack.mallocLong(1);
            check(vkCreateComputePipelines(device, NULL, pipelineInfo, null, pipelineResult),
                    "vkCreateComputePipelines(" + resource + ")");
            return pipelineResult.get(0);
        } finally {
            MemoryUtil.memFree(code);
            if (shaderModule != 0) {
                vkDestroyShaderModule(device, shaderModule, null);
            }
        }
    }

    private ByteBuffer readResource(String resource) throws IOException {
        try (InputStream stream = VulkanComputeBackend.class.getResourceAsStream(resource)) {
            if (stream == null) {
                throw new IOException("Missing SPIR-V resource " + resource);
            }
            byte[] bytes = stream.readAllBytes();
            ByteBuffer buffer = MemoryUtil.memAlloc(bytes.length);
            buffer.put(bytes).flip();
            return buffer;
        }
    }

    private void createCommandObjects() {
        try (MemoryStack stack = stackPush()) {
            VkCommandPoolCreateInfo poolInfo = VkCommandPoolCreateInfo.calloc(stack)
                    .sType$Default()
                    .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                    .queueFamilyIndex(queueFamilyIndex);
            LongBuffer poolResult = stack.mallocLong(1);
            check(vkCreateCommandPool(device, poolInfo, null, poolResult), "vkCreateCommandPool");
            commandPool = poolResult.get(0);

            for (Slot slot : slots) {
                VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack).sType$Default();
                LongBuffer fenceResult = stack.mallocLong(1);
                check(vkCreateFence(device, fenceInfo, null, fenceResult), "vkCreateFence");
                slot.fence = fenceResult.get(0);
                if (timestampValidBits > 0) {
                    VkQueryPoolCreateInfo queryInfo = VkQueryPoolCreateInfo.calloc(stack)
                            .sType$Default().queryType(VK_QUERY_TYPE_TIMESTAMP).queryCount(MEASURE_FILL ? 10 : 3);
                    LongBuffer queryResult = stack.mallocLong(1);
                    check(vkCreateQueryPool(device, queryInfo, null, queryResult), "vkCreateQueryPool");
                    slot.queryPool = queryResult.get(0);
                }
            }
        }
        for (Slot slot : slots) slot.commandBuffer = allocateCommandBuffer();
    }

    private VkCommandBuffer allocateCommandBuffer() {
        try (MemoryStack stack = stackPush()) {
            VkCommandBufferAllocateInfo commandInfo = VkCommandBufferAllocateInfo.calloc(stack)
                    .sType$Default()
                    .commandPool(commandPool)
                    .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                    .commandBufferCount(1);
            PointerBuffer commandResult = stack.mallocPointer(1);
            check(vkAllocateCommandBuffers(device, commandInfo, commandResult), "vkAllocateCommandBuffers");
            return new VkCommandBuffer(commandResult.get(0), device);
        }
    }

    private void recordStagedDensity(Slot slot, StagedDensityBatch batch) {
        VkCommandBuffer command = slot.commandBuffer;
        long queryPool = slot.queryPool;
        try (MemoryStack stack = stackPush()) {
            check(vkResetCommandBuffer(command, 0), "vkResetCommandBuffer(staged density)");
            VkCommandBufferBeginInfo begin = VkCommandBufferBeginInfo.calloc(stack).sType$Default();
            check(vkBeginCommandBuffer(command, begin), "vkBeginCommandBuffer(staged density)");
            if (queryPool != 0) vkCmdResetQueryPool(command, queryPool, 0, MEASURE_FILL ? 10 : 3);
            VkMemoryBarrier.Buffer hostBarrier = VkMemoryBarrier.calloc(1, stack);
            hostBarrier.get(0).sType$Default().srcAccessMask(VK_ACCESS_HOST_WRITE_BIT)
                    .dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, hostBarrier, null, null);
            if (queryPool != 0) vkCmdWriteTimestamp(command,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 0);
            int timestampIndex = 0;
            for (StagedDensityBatch.Stage stage : batch.stages()) {
                long pipeline = stage.kind() == 0 ? doubleSingleNoisePipeline
                        : stage.kind() == 1 ? densityGraphPipeline : densitySplinePipeline;
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipelineLayout, 0, stack.longs(slot.arenaDescriptorSet), null);
                ByteBuffer constants = stack.malloc(16).order(ByteOrder.nativeOrder());
                if (stage.kind() == 0)
                    constants.putInt(stage.table()).putInt(2).putInt(0).putInt(-1);
                else if (stage.kind() == 1)
                    constants.putInt(0).putInt(4).putInt(stage.table()).putInt(0);
                else constants.putInt(0).putInt(0).putInt(stage.table()).putInt(-2);
                constants.flip();
                vkCmdPushConstants(command, pipelineLayout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, constants);
                vkCmdDispatch(command, (stage.samples() + 63) / 64, stage.jobs(), 1);
                VkMemoryBarrier.Buffer between = VkMemoryBarrier.calloc(1, stack);
                between.get(0).sType$Default().srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                        .dstAccessMask(VK_ACCESS_SHADER_READ_BIT);
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, between, null, null);
                if (MEASURE_FILL && queryPool != 0) vkCmdWriteTimestamp(command,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, ++timestampIndex);
            }
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, densityFinalFusedPipeline);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipelineLayout, 0, stack.longs(slot.descriptorSet), null);
            ByteBuffer constants = stack.malloc(16).order(ByteOrder.nativeOrder());
            constants.putInt(batch.blocks() * 3).putInt(1).putInt(0).putInt(0).flip();
            vkCmdPushConstants(command, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constants);
            vkCmdDispatch(command, (batch.blocks() + 63) / 64, 1, 1);
            if (queryPool != 0) vkCmdWriteTimestamp(command,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, MEASURE_FILL ? ++timestampIndex : 1);
            if (stagedOutput) {
                VkMemoryBarrier.Buffer transfer = VkMemoryBarrier.calloc(1, stack);
                transfer.get(0).sType$Default().srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                        .dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT);
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, transfer, null, null);
                VkBufferCopy.Buffer copy = VkBufferCopy.calloc(1, stack);
                copy.get(0).srcOffset(0).dstOffset(0).size((long) batch.blocks() * 3 * Integer.BYTES);
                vkCmdCopyBuffer(command, slot.outputBuffer.buffer, slot.readbackBuffer.buffer, copy);
                if (queryPool != 0) vkCmdWriteTimestamp(command,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, queryPool, MEASURE_FILL ? ++timestampIndex : 2);
                VkMemoryBarrier.Buffer readback = VkMemoryBarrier.calloc(1, stack);
                readback.get(0).sType$Default().srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT)
                        .dstAccessMask(VK_ACCESS_HOST_READ_BIT);
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, readback, null, null);
            } else {
                VkMemoryBarrier.Buffer readback = VkMemoryBarrier.calloc(1, stack);
                readback.get(0).sType$Default().srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                        .dstAccessMask(VK_ACCESS_HOST_READ_BIT);
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, readback, null, null);
            }
            check(vkEndCommandBuffer(command), "vkEndCommandBuffer(staged density)");
        }
    }

    private void submitCommand(Slot slot) {
        try (MemoryStack stack = stackPush()) {
            check(vkResetFences(device, slot.fence), "vkResetFences");
            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack)
                    .sType$Default()
                    .pCommandBuffers(stack.pointers(slot.commandBuffer));
            check(vkQueueSubmit(computeQueue, submitInfo, slot.fence), "vkQueueSubmit");
        }
    }

    private GpuTiming readGpuTimestamp(Slot slot, StagedDensityBatch batch) {
        if (slot.queryPool == 0) {
            return new GpuTiming(-1, -1, new long[8]);
        }
        try (MemoryStack stack = stackPush()) {
            int finalIndex = MEASURE_FILL ? batch.stages().size() + 1 : 1;
            int count = finalIndex + 1 + (stagedOutput ? 1 : 0);
            LongBuffer timestamps = stack.mallocLong(count);
            int result = vkGetQueryPoolResults(device, slot.queryPool, 0, count, timestamps, Long.BYTES,
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            check(result, "vkGetQueryPoolResults");
            long compute = timestampDelta(timestamps.get(0), timestamps.get(finalIndex));
            long copy = stagedOutput ? timestampDelta(timestamps.get(finalIndex), timestamps.get(finalIndex + 1)) : 0;
            long[] stages = new long[8];
            if (MEASURE_FILL) {
                int index = 0;
                for (StagedDensityBatch.Stage stage : batch.stages()) {
                    stages[stage.sourceStage()] += timestampDelta(timestamps.get(index), timestamps.get(++index));
                }
                stages[7] = timestampDelta(timestamps.get(index), timestamps.get(finalIndex));
            }
            return new GpuTiming(compute, copy, stages);
        }
    }

    private long timestampDelta(long start, long end) {
        long delta = end - start;
        if (timestampValidBits < Long.SIZE) delta &= (1L << timestampValidBits) - 1L;
        return (long) (delta * timestampPeriod);
    }

    private static void check(int result, String operation) {
        if (result != VK_SUCCESS) {
            throw new IllegalStateException(operation + " failed with VkResult " + result);
        }
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (device != null) {
            vkDeviceWaitIdle(device);
            for (Slot slot : slots) {
                if (slot.queryPool != 0) vkDestroyQueryPool(device, slot.queryPool, null);
                if (slot.fence != 0) vkDestroyFence(device, slot.fence, null);
            }
            if (commandPool != 0) vkDestroyCommandPool(device, commandPool, null);
            if (doubleSingleNoisePipeline != 0) vkDestroyPipeline(device, doubleSingleNoisePipeline, null);
            if (densityGraphPipeline != 0) vkDestroyPipeline(device, densityGraphPipeline, null);
            if (densitySplinePipeline != 0) vkDestroyPipeline(device, densitySplinePipeline, null);
            if (densityFinalFusedPipeline != 0) vkDestroyPipeline(device, densityFinalFusedPipeline, null);
            if (pipelineLayout != 0) vkDestroyPipelineLayout(device, pipelineLayout, null);
            if (descriptorPool != 0) vkDestroyDescriptorPool(device, descriptorPool, null);
            if (descriptorSetLayout != 0) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, null);
            for (Slot slot : slots) {
                destroyBuffer(slot.outputBuffer);
                destroyBuffer(slot.readbackBuffer);
                destroyBuffer(slot.inputBuffer);
            }
            vkDestroyDevice(device, null);
            device = null;
        }
        if (instance != null) {
            vkDestroyInstance(instance, null);
            instance = null;
        }
    }

    private void destroyBuffer(BufferAllocation allocation) {
        if (allocation == null || device == null) return;
        if (allocation.mapped != null) vkUnmapMemory(device, allocation.memory);
        vkDestroyBuffer(device, allocation.buffer, null);
        vkFreeMemory(device, allocation.memory, null);
    }

    private record GpuTiming(long computeNanos, long copyNanos, long[] stageNanos) {}

    static final class Slot {
        private BufferAllocation inputBuffer, outputBuffer, readbackBuffer;
        private long descriptorSet, arenaDescriptorSet, fence, queryPool;
        private VkCommandBuffer commandBuffer;
        private StagedDensityBatch.Workspace workspace;
        private final int[] residentInput = new int[MAX_INPUT_INTS];
        private final boolean[] residentValid = new boolean[MAX_INPUT_INTS];
        private StagedDensityBatch batch;
        private long totalStart, uploadNanos, commandNanos, submitNanos;

        StagedDensityBatch.Workspace workspace() { return workspace; }
    }

    private record BufferAllocation(long buffer, long memory, ByteBuffer mapped,
                                    int memoryType, int flags, long allocationSize) {
    }

}
