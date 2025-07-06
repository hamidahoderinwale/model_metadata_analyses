import json
from collections import Counter, defaultdict

# Load config.json
with open("config.json", "r") as f:
    config = json.load(f)

# Load model.safetensors.index.json
with open("model.safetensors.index.json", "r") as f:
    index = json.load(f)

print("=== MODEL ARCHITECTURE ANALYSIS ===")

# 1. Layer count and types (inferred from parameter names)
layer_types = defaultdict(int)
layer_prefix = "model.layers."
layer_ids = set()

for param_name in index["weight_map"]:
    if param_name.startswith(layer_prefix):
        parts = param_name.split(".")
        if len(parts) > 3:
            layer_id = parts[2]
            layer_type = parts[3]
            layer_types[layer_type] += 1
            layer_ids.add(layer_id)

layer_count = config.get("num_hidden_layers", len(layer_ids))
print(f"Layer count: {layer_count}")
print("Layer types (by parameter occurrence):")
for ltype, count in layer_types.items():
    print(f"  {ltype}: {count}")

# 2. Vocab size
vocab_size = config.get("vocab_size", "Unknown")
print(f"Vocab size: {vocab_size}")

# 3. Shape variety
print("\n=== SHAPE VARIETY ===")
if "shapes" in index:
    shape_counter = Counter(tuple(v) for v in index["shapes"].values())
    print("Unique tensor shapes and their counts:")
    for shape, count in shape_counter.items():
        print(f"  shape {shape}: {count}")
else:
    print("No shape information found in index file. (Some index files do not include shapes.)")

# 4. Quantization scheme and subproperties
print("\n=== QUANTIZATION CONFIGURATION ===")
qcfg = config.get("quantization_config")
if qcfg:
    print("Quantization config found:")
    for k, v in qcfg.items():
        print(f"  {k}: {v}")
else:
    print("No quantization_config found in config.json.")

# Optional: explain quantization subproperties if present
if qcfg:
    fmt = qcfg.get("fmt", "")
    quant_method = qcfg.get("quant_method", "")
    block_size = qcfg.get("weight_block_size", "")
    activation_scheme = qcfg.get("activation_scheme", "")
    print("\nQuantization details summary:")
    print(f"- Format: {fmt}")
    print(f"- Method: {quant_method}")
    print(f"- Weight block size: {block_size}")
    print(f"- Activation quantization scheme: {activation_scheme}")
    # FP8 specifics (from DeepSeek docs)
    if quant_method == "fp8":
        print("  (DeepSeek-V3 uses FP8 e4m3 quantization with 128x128 block scaling.)")
        print("  Dequantization uses per-block scale tensors stored as float32.")
