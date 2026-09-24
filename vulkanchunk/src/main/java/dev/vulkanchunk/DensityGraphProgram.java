package dev.vulkanchunk;

import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.DensityFunctions;
import net.minecraft.util.Mth;

import java.lang.reflect.Method;
import java.lang.reflect.RecordComponent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/** Packed arithmetic for the staged density graph. */
final class DensityGraphProgram {
    static final int LEAF = 0, CONSTANT = 1, ADD = 2, MUL = 3, MIN = 4, MAX = 5;
    static final int ADD_CONSTANT = 6, MUL_CONSTANT = 7, ABS = 8, SQUARE = 9, CUBE = 10;
    static final int HALF_NEGATIVE = 11, QUARTER_NEGATIVE = 12, SQUEEZE = 13, CLAMP = 14, RANGE = 15;
    static final int Y_GRADIENT = 16;
    private static final int HEADER = 5, NODE_STRIDE = 10;
    private final List<Node> nodes = new ArrayList<>();
    private final List<DensityFunction> leaves = new ArrayList<>();
    private final Map<DensityFunction, Integer> memo = new IdentityHashMap<>();
    private final Map<Integer, int[]> batchTemplates = new HashMap<>();
    private final int root;
    private final boolean emptyBlender;
    private int yNode = -1;

    DensityGraphProgram(DensityFunction root) {
        this(root, false);
    }

    DensityGraphProgram(DensityFunction root, boolean emptyBlender) {
        this.emptyBlender = emptyBlender;
        this.root = add(root);
    }

    int nodeCount() { return nodes.size(); }
    int leafCount() { return leaves.size(); }
    List<DensityFunction> leaves() { return leaves; }
    String leafTypes() {
        return leaves.stream().map(leaf -> leaf == null ? "BlockY" : leaf.getClass().getSimpleName()).distinct()
                .reduce((a, b) -> a + "," + b).orElse("none");
    }

    double[] capture(DensityFunction.FunctionContext context) {
        double[] values = new double[leaves.size()];
        for (int i = 0; i < values.length; i++) values[i] = leaves.get(i) == null
                ? context.blockY() : leaves.get(i).compute(context);
        return values;
    }

    Evaluation evaluateJava(double[] leavesAtPoint) {
        double[] values = new double[nodes.size()];
        int branchHash = 0x811c9dc5;
        for (int i = 0; i < values.length; i++) {
            Node node = nodes.get(i);
            double a = node.a >= 0 ? values[node.a] : 0.0;
            double b = node.b >= 0 && node.op != Y_GRADIENT ? values[node.b] : 0.0;
            values[i] = switch (node.op) {
                case LEAF -> leavesAtPoint[node.a];
                case CONSTANT -> node.p;
                case ADD -> a + b;
                case MUL -> a == 0.0 ? 0.0 : a * b;
                case MIN -> Math.min(a, b);
                case MAX -> Math.max(a, b);
                case ADD_CONSTANT -> a + node.p;
                case MUL_CONSTANT -> a * node.p;
                case ABS -> Math.abs(a);
                case SQUARE -> a * a;
                case CUBE -> a * a * a;
                case HALF_NEGATIVE -> a > 0.0 ? a : a * 0.5;
                case QUARTER_NEGATIVE -> a > 0.0 ? a : a * 0.25;
                case SQUEEZE -> {
                    double clamped = Math.max(-1.0, Math.min(1.0, a));
                    yield clamped / 2.0 - clamped * clamped * clamped / 24.0;
                }
                case CLAMP -> Math.max(node.p, Math.min(node.q, a));
                case RANGE -> a >= node.p && a < node.q ? b : values[node.c];
                case Y_GRADIENT -> Mth.clampedMap(a, node.b, node.c, node.p, node.q);
                default -> throw new IllegalStateException("Unknown opcode " + node.op);
            };
            if (node.op == RANGE) branchHash = (branchHash ^
                    (a >= node.p && a < node.q ? 1 : 0)) * 16777619;
        }
        return new Evaluation(values[root], branchHash);
    }

    int[] batch(double[][] samples, int from, int count) {
        int[] words = Arrays.copyOf(batchTemplates.computeIfAbsent(count, this::batchTemplate),
                HEADER + nodes.size() * NODE_STRIDE + count * leaves.size() * 2);
        int cursor = words[4];
        for (int sample = from; sample < from + count; sample++) {
            for (double value : samples[sample]) {
                pair(words, cursor, value);
                cursor += 2;
            }
        }
        return words;
    }

    private int[] batchTemplate(int count) {
        int[] words = new int[HEADER + nodes.size() * NODE_STRIDE + count * leaves.size() * 2];
        words[0] = count;
        words[1] = nodes.size();
        words[2] = leaves.size();
        words[3] = HEADER;
        words[4] = HEADER + nodes.size() * NODE_STRIDE;
        for (int i = 0; i < nodes.size(); i++) {
            Node node = nodes.get(i);
            int base = HEADER + i * NODE_STRIDE;
            words[base] = node.op;
            words[base + 1] = node.a;
            words[base + 2] = node.b;
            words[base + 3] = node.c;
            pair(words, base + 5, node.p);
            pair(words, base + 7, node.q);
            if (node.op == Y_GRADIENT) {
                double inverse = 1.0 / (node.c - node.b);
                float high = (float)inverse;
                words[base + 4] = Float.floatToRawIntBits(high);
                words[base + 9] = Float.floatToRawIntBits((float)(inverse - high));
            }
        }
        return words;
    }

    private int add(DensityFunction function) {
        if (function instanceof DensityFunctions.HolderHolder holder) return add(holder.function().value());
        if (function instanceof DensityFunctions.MarkerOrMarked marker
                && (function.getClass().getSimpleName().equals("CacheOnce")
                || function.getClass().getSimpleName().equals("FlatCache")
                || function.getClass().getSimpleName().equals("Cache2D")))
            return add(marker.wrapped());
        if (emptyBlender && function.getClass().getSimpleName().equals("BlendDensity"))
            return add((DensityFunction)part(function, "input"));
        if (emptyBlender && (function.getClass().getSimpleName().equals("BlendAlpha")
                || function.getClass().getSimpleName().equals("BlendOffset"))) {
            int index = nodes.size();
            nodes.add(new Node(CONSTANT, -1, -1, -1,
                    function.getClass().getSimpleName().equals("BlendAlpha") ? 1.0 : 0.0, 0));
            return index;
        }
        Integer prior = memo.get(function);
        if (prior != null) return prior;
        String type = function.getClass().getSimpleName();
        Node node;
        switch (type) {
            case "Ap2" -> {
                int a = add((DensityFunction)part(function, "argument1"));
                int b = add((DensityFunction)part(function, "argument2"));
                int op = switch (part(function, "type").toString()) {
                    case "ADD" -> ADD; case "MUL" -> MUL; case "MIN" -> MIN; case "MAX" -> MAX;
                    default -> throw new IllegalStateException(type);
                };
                node = new Node(op, a, b, -1, 0, 0);
            }
            case "MulOrAdd" -> {
                int a = add((DensityFunction)part(function, "input"));
                int op = part(function, "specificType").toString().equals("ADD") ? ADD_CONSTANT : MUL_CONSTANT;
                node = new Node(op, a, -1, -1, (double)part(function, "argument"), 0);
            }
            case "Mapped" -> {
                int a = add((DensityFunction)part(function, "input"));
                int op = switch (part(function, "type").toString()) {
                    case "ABS" -> ABS; case "SQUARE" -> SQUARE; case "CUBE" -> CUBE;
                    case "HALF_NEGATIVE" -> HALF_NEGATIVE; case "QUARTER_NEGATIVE" -> QUARTER_NEGATIVE;
                    case "SQUEEZE" -> SQUEEZE;
                    default -> throw new IllegalStateException(type);
                };
                node = new Node(op, a, -1, -1, 0, 0);
            }
            case "Clamp" -> node = new Node(CLAMP, add((DensityFunction)part(function, "input")), -1, -1,
                    (double)part(function, "minValue"), (double)part(function, "maxValue"));
            case "RangeChoice" -> node = new Node(RANGE,
                    add((DensityFunction)part(function, "input")),
                    add((DensityFunction)part(function, "whenInRange")),
                    add((DensityFunction)part(function, "whenOutOfRange")),
                    (double)part(function, "minInclusive"), (double)part(function, "maxExclusive"));
            case "Constant" -> node = new Node(CONSTANT, -1, -1, -1, (double)part(function, "value"), 0);
            case "YClampedGradient" -> {
                if (yNode < 0) {
                    int leaf = leaves.size();
                    leaves.add(null);
                    yNode = nodes.size();
                    nodes.add(new Node(LEAF, leaf, -1, -1, 0, 0));
                }
                node = new Node(Y_GRADIENT, yNode, (int)part(function, "fromY"),
                        (int)part(function, "toY"), (double)part(function, "fromValue"),
                        (double)part(function, "toValue"));
            }
            default -> {
                int leaf = leaves.size();
                leaves.add(function);
                node = new Node(LEAF, leaf, -1, -1, 0, 0);
            }
        }
        int index = nodes.size();
        nodes.add(node);
        memo.put(function, index);
        return index;
    }

    static Object part(DensityFunction function, String name) {
        try {
            for (RecordComponent component : function.getClass().getRecordComponents()) {
                if (component.getName().equals(name)) {
                    Method accessor = component.getAccessor();
                    accessor.setAccessible(true);
                    return accessor.invoke(function);
                }
            }
            throw new IllegalStateException(function.getClass() + " lacks " + name);
        } catch (ReflectiveOperationException failure) {
            throw new IllegalStateException(failure);
        }
    }

    private static void pair(int[] words, int index, double value) {
        float hi = (float)value;
        words[index] = Float.floatToRawIntBits(hi);
        words[index + 1] = Float.floatToRawIntBits((float)(value - hi));
    }

    private record Node(int op, int a, int b, int c, double p, double q) {}
    record Evaluation(double value, int branchHash) {}
}
