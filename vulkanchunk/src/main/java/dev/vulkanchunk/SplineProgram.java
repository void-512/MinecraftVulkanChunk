package dev.vulkanchunk;

import net.minecraft.util.CubicSpline;
import net.minecraft.util.Mth;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.DensityFunctions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class SplineProgram {
    private final List<Node> nodes = new ArrayList<>();
    private final List<Point> points = new ArrayList<>();
    private final List<DensityFunctions.Spline.Coordinate> coordinates = new ArrayList<>();
    private final Map<Integer, int[]> batchTemplates = new HashMap<>();

    SplineProgram(DensityFunctions.Spline density) {
        add(density.spline());
    }

    int nodeCount() { return nodes.size(); }
    int coordinateCount() { return coordinates.size(); }
    int pointCount() { return points.size(); }
    DensityFunction coordinateFunction(int index) { return coordinates.get(index).function().value(); }

    float[] capture(DensityFunction.FunctionContext context) {
        float[] result = new float[coordinates.size()];
        DensityFunctions.Spline.Point point = new DensityFunctions.Spline.Point(context);
        for (int i = 0; i < result.length; i++) result[i] = coordinates.get(i).apply(point);
        return result;
    }

    Evaluation evaluateJava(float[] coordinatesAtPoint) {
        float[] values = new float[nodes.size()];
        int branchHash = 0x811c9dc5;
        for (int n = 0; n < nodes.size(); n++) {
            Node node = nodes.get(n);
            if (node.count == 0) { values[n] = node.constant; continue; }
            float coordinate = coordinatesAtPoint[node.coordinate];
            int interval = -1;
            for (int j = 0; j < node.count; j++) {
                if (coordinate >= points.get(node.start + j).location) interval = j;
                else break;
            }
            branchHash = (branchHash ^ (interval + 1)) * 16777619;
            int a = interval < 0 ? 0 : interval;
            Point p = points.get(node.start + a);
            if (interval < 0 || interval == node.count - 1) {
                float value = values[p.child];
                values[n] = p.derivative == 0 ? value : value + p.derivative * (coordinate - p.location);
            } else {
                Point q = points.get(node.start + a + 1);
                float delta = q.location - p.location;
                float t = (coordinate - p.location) / delta;
                float low = values[p.child], high = values[q.child];
                float m0 = p.derivative * delta - (high - low);
                float m1 = -q.derivative * delta + (high - low);
                values[n] = Mth.lerp(t, low, high) + t * (1.0F - t) * Mth.lerp(t, m0, m1);
            }
        }
        return new Evaluation(values[values.length - 1], branchHash);
    }

    int[] batch(float[][] samples) {
        int[] words = Arrays.copyOf(batchTemplates.computeIfAbsent(samples.length, this::batchTemplate),
                batchTemplates.get(samples.length).length);
        int at = words[6];
        for (float[] sample : samples) for (float value : sample) words[at++] = Float.floatToRawIntBits(value);
        return words;
    }

    private int[] batchTemplate(int sampleCount) {
        int header = 7, nodeOffset = header, pointOffset = nodeOffset + nodes.size() * 5;
        int sampleOffset = pointOffset + points.size() * 3;
        int[] words = new int[sampleOffset + sampleCount * coordinates.size()];
        words[0] = sampleCount;
        words[1] = nodes.size();
        words[2] = points.size();
        words[3] = coordinates.size();
        words[4] = nodeOffset;
        words[5] = pointOffset;
        words[6] = sampleOffset;
        for (int i = 0; i < nodes.size(); i++) {
            Node n = nodes.get(i);
            int at = nodeOffset + i * 5;
            words[at] = n.count == 0 ? 0 : 1;
            words[at + 1] = n.coordinate;
            words[at + 2] = n.start;
            words[at + 3] = n.count;
            words[at + 4] = Float.floatToRawIntBits(n.constant);
        }
        for (int i = 0; i < points.size(); i++) {
            Point p = points.get(i);
            int at = pointOffset + i * 3;
            words[at] = Float.floatToRawIntBits(p.location);
            words[at + 1] = Float.floatToRawIntBits(p.derivative);
            words[at + 2] = p.child;
        }
        return words;
    }

    private int add(CubicSpline<DensityFunctions.Spline.Point, DensityFunctions.Spline.Coordinate> spline) {
        if (spline instanceof CubicSpline.Constant<?, ?> constant) {
            int index = nodes.size();
            nodes.add(new Node(-1, 0, 0, constant.value()));
            return index;
        }
        CubicSpline.Multipoint<DensityFunctions.Spline.Point, DensityFunctions.Spline.Coordinate> multi =
                (CubicSpline.Multipoint<DensityFunctions.Spline.Point, DensityFunctions.Spline.Coordinate>)spline;
        int[] children = new int[multi.values().size()];
        for (int i = 0; i < children.length; i++) children[i] = add(multi.values().get(i));
        int coordinate = coordinates.size();
        coordinates.add(multi.coordinate());
        int start = points.size();
        for (int i = 0; i < children.length; i++)
            points.add(new Point(multi.locations()[i], multi.derivatives()[i], children[i]));
        int index = nodes.size();
        nodes.add(new Node(coordinate, start, children.length, 0));
        return index;
    }

    private record Node(int coordinate, int start, int count, float constant) {}
    private record Point(float location, float derivative, int child) {}
    record Evaluation(float value, int branchHash) {}
}
