package dev.vulkanchunk;

import net.minecraft.commands.Commands;
import net.minecraft.network.chat.Component;
import net.neoforged.bus.api.IEventBus;
import net.neoforged.bus.api.SubscribeEvent;
import net.neoforged.fml.common.Mod;
import net.neoforged.neoforge.common.NeoForge;
import net.neoforged.neoforge.event.RegisterCommandsEvent;
import net.neoforged.neoforge.event.server.ServerStoppedEvent;
import net.neoforged.neoforge.event.server.ServerStartedEvent;

@Mod(VulkanChunkMod.MOD_ID)
public final class VulkanChunkMod {
    public static final String MOD_ID = "vulkanchunk";

    public VulkanChunkMod(IEventBus modEventBus) {
        NeoForge.EVENT_BUS.register(this);
        VulkanChunkService.initialize();
    }

    @SubscribeEvent
    public void registerCommands(RegisterCommandsEvent event) {
        event.getDispatcher().register(Commands.literal("vulkanchunk")
                .requires(source -> source.hasPermission(2))
                .then(Commands.literal("status").executes(context -> {
                    context.getSource().sendSuccess(() -> Component.literal(VulkanChunkService.status()), false);
                    return 1;
                })));
    }

    @SubscribeEvent
    public void serverStarted(ServerStartedEvent event) {
        ProductionDensity.registerLevels(event.getServer());
    }

    @SubscribeEvent
    public void serverStopped(ServerStoppedEvent event) {
        VulkanChunkService.shutdown();
    }
}
