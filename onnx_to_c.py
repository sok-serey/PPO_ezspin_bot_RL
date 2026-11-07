"""Export ONNX dense neural network weights to a pure C header file (.h).

Designed for microcontroller deployment (ESP32, Teensy, STM32, Arduino, ARM Cortex-M)
without any runtime library or external dependencies.

Usage:
    python onnx_to_c.py <onnx_path> [output_h_path]

Example:
    python onnx_to_c.py runs/ppo_full_dr/best_model.onnx firmware/actor.h
"""

import argparse
from pathlib import Path
import numpy as np
import onnx
import onnx.numpy_helper


def _array_to_c(arr: np.ndarray, name: str) -> str:
    """Converts a numpy array into a C static const float array declaration."""
    flat = arr.flatten().astype(np.float32)
    shape = " x ".join(str(s) for s in arr.shape)
    
    # Format 8 values per line for readability and small file size
    formatted_vals = []
    chunk_size = 8
    for i in range(0, len(flat), chunk_size):
        chunk = flat[i : i + chunk_size]
        formatted_vals.append("    " + ", ".join(f"{v:.8f}f" for v in chunk))
    values_str = ",\n".join(formatted_vals)

    return (
        f"// Shape: {shape}\n"
        f"static const float {name}[{len(flat)}] = {{\n"
        f"{values_str}\n"
        f"}};\n"
    )


def export_onnx_to_c(onnx_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Exports an ONNX actor network to a self-contained C header file (.h)."""
    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path.with_suffix(".h")
    else:
        output_path = Path(output_path)

    print(f"Loading ONNX model from: {onnx_path}")
    model = onnx.load(str(onnx_path))
    init_map = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

    # Extract Gemm layers following the graph node sequence
    layers = []
    for node in model.graph.node:
        if node.op_type == "Gemm":
            w_name = node.input[1]
            b_name = node.input[2]
            w = init_map[w_name]
            b = init_map[b_name]
            layers.append({"weight": w, "bias": b})

    if not layers:
        # Fallback to scanning initializers
        weights = [arr for arr in init_map.values() if len(arr.shape) == 2]
        biases = [arr for arr in init_map.values() if len(arr.shape) == 1]
        for w, b in zip(weights, biases):
            layers.append({"weight": w, "bias": b})

    layer_info = []  # list of (in_dim, out_dim)
    layer_lines = []

    for i, layer in enumerate(layers):
        w = layer["weight"]
        b = layer["bias"]
        out_dim, in_dim = w.shape[0], w.shape[1]
        layer_info.append((in_dim, out_dim))
        layer_lines.append(_array_to_c(w, f"layer{i}_weight"))
        layer_lines.append(_array_to_c(b, f"layer{i}_bias"))

    obs_dim = layer_info[0][0]
    action_dim = layer_info[-1][1]
    num_layers = len(layer_info)
    max_hidden = max(out_dim for in_dim, out_dim in layer_info[:-1]) if num_layers > 1 else max(obs_dim, action_dim)

    # Build C forward function
    fwd = [
        "/**",
        " * Pure C forward pass for microcontroller deployment.",
        " * Zero dynamic memory allocation (no malloc/free).",
        " *",
        f" * @param obs    Input array [{obs_dim}]:",
        " *                 [0] pitch (rad)",
        " *                 [1] pitch_rate (rad/s)",
        " *                 [2] left_wheel_vel * 0.1 (rad/s)",
        " *                 [3] right_wheel_vel * 0.1 (rad/s)",
        f" * @param action Output array [{action_dim}]:",
        " *                 [0] left_torque_norm in [-1.0, 1.0]",
        " *                 [1] right_torque_norm in [-1.0, 1.0]",
        " */",
        "static inline void actor_forward(const float* obs, float* action) {",
        f"    float buf0[{max_hidden}];",
        f"    float buf1[{max_hidden}];",
        "    const float* in  = obs;",
        "    float*       out = buf0;",
        "",
    ]

    for i, (in_size, out_size) in enumerate(layer_info):
        is_last = (i == num_layers - 1)
        act_desc = "linear + clamp[-1, 1]" if is_last else "tanh"
        fwd.append(f"    // Layer {i}: {in_size} -> {out_size} ({act_desc})")
        fwd.append(f"    for (int o = 0; o < {out_size}; o++) {{")
        fwd.append(f"        float acc = layer{i}_bias[o];")
        fwd.append(f"        for (int j = 0; j < {in_size}; j++) {{")
        fwd.append(f"            acc += layer{i}_weight[o * {in_size} + j] * in[j];")
        fwd.append("        }")
        if not is_last:
            fwd.append("        out[o] = tanhf(acc);")
            fwd.append("    }")
            fwd.append("    in  = out;")
            fwd.append("    out = (out == buf0) ? buf1 : buf0;")
        else:
            # Clamping output layer safely to [-1.0, 1.0]
            fwd.append("        if (acc > 1.0f) acc = 1.0f;")
            fwd.append("        else if (acc < -1.0f) acc = -1.0f;")
            fwd.append("        action[o] = acc;")
            fwd.append("    }")
        fwd.append("")

    fwd.append("}")

    # Build header content
    lines = [
        "// DO NOT EDIT: Auto-generated actor policy for microcontroller deployment",
        f"// Source: {onnx_path.name}",
        "// Compatible with C99 and C++ (Arduino, ESP32, Teensy, STM32, ARM Cortex-M)",
        "",
        "#ifndef ACTOR_POLICY_H",
        "#define ACTOR_POLICY_H",
        "",
        "#include <math.h>  // for tanhf()",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        f"#define ACTOR_OBS_SIZE    {obs_dim}",
        f"#define ACTOR_ACTION_SIZE {action_dim}",
        f"#define ACTOR_NUM_LAYERS  {num_layers}",
        "",
    ]
    lines += layer_lines
    lines += fwd
    lines += [
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        "#endif // ACTOR_POLICY_H",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Success] Generated C header file: {output_path}")
    print(f"          Layers:  {num_layers}")
    print(f"          Inputs:  {obs_dim}")
    print(f"          Outputs: {action_dim}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX actor policy to pure C header file (.h)")
    parser.add_argument("onnx_path", help="Path to input .onnx file")
    parser.add_argument("output_path", nargs="?", default=None, help="Path to output .h file (default: same name as onnx)")
    args = parser.parse_args()

    export_onnx_to_c(args.onnx_path, args.output_path)


if __name__ == "__main__":
    main()

