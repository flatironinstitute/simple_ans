import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from simple_ans import ans_encode, ans_decode

# Generate random test data from normal distribution
n = 10_000_000
signal = np.round(np.random.normal(0, 1, n) * 5).astype(np.int32)

# Calculate ideal compression ratio
vals, counts = np.unique(signal, return_counts=True)
probs = counts / len(signal)
ideal_compression_ratio = signal.itemsize * 8 / -np.sum(probs * np.log2(probs))
print(f"Ideal compression ratio: {ideal_compression_ratio}")

# List to store all results
results = []

# Test simple_ans
ans_encode(signal=signal)

def warmup():
    timer = time.time()
    while (time.time() - timer) < 1:
        ans_encode(signal=signal)

warmup()

timer = time.time()
# auto_counts, auto_values = determine_symbol_counts_and_values(
#     signal, index_length=2**16
# )
num_runs = 0
while time.time() - timer < 4:
    encoded = ans_encode(
        signal=signal,
    )  # Using auto-determined symbol counts
    num_runs += 1
elapsed_encode = (time.time() - timer) / num_runs
encoded = ans_encode(signal=signal)

timer = time.time()
num_runs = 0
while time.time() - timer < 4:
    signal_decoded = ans_decode(encoded)
    num_runs += 1
elapsed_decode = (time.time() - timer) / num_runs

signal_decoded = ans_decode(encoded)
assert len(signal_decoded) == len(signal)
assert np.all(signal_decoded == signal)
print("Decoded signal matches original signal")

# 64 bits per bitstream word, 32 bits for state, 32 bits per symbol count, 32 bits per symbol value, 32 bits for num_bits, 32 bits for signal_length
signal_bytes = signal.nbytes
compression_ratio = signal_bytes / encoded.size()

results.append(
    {
        "name": "simple_ans",
        "compression_ratio": float(compression_ratio),
        "pct_of_ideal": float(compression_ratio / ideal_compression_ratio * 100),
        "encode_time": float(elapsed_encode),
        "decode_time": float(elapsed_decode),
        "encode_speed_MBps": float(signal_bytes / elapsed_encode / 1e6),
        "decode_speed_MBps": float(signal_bytes / elapsed_decode / 1e6),
    }
)

print(f"Ideal compression ratio: {ideal_compression_ratio}")
print(f"simple_ans: Compression ratio: {compression_ratio}")
print(
    f"simple_ans: Pct of ideal compression: {compression_ratio/ideal_compression_ratio*100:.2f}%"
)
print("")
print(
    f"simple_ans: Time to encode: {elapsed_encode:.2f} seconds ({signal_bytes/elapsed_encode/1e6:.2f} MB/s)"
)
print(
    f"simple_ans: Time to decode: {elapsed_decode:.2f} seconds ({signal_bytes/elapsed_decode/1e6:.2f} MB/s)"
)
print("")

# Test zlib at different levels
import zlib

zlib_levels = [1, 3, 5, 7, 9]
for level in zlib_levels:
    timer = time.time()
    buf_compressed = zlib.compress(signal.tobytes(), level=level)
    elapsed_zlib = time.time() - timer
    zlib_compression_ratio = signal_bytes / len(buf_compressed)

    timer = time.time()
    signal_decompressed = np.frombuffer(
        zlib.decompress(buf_compressed), dtype=signal.dtype
    )
    elapsed_zlib_decode = time.time() - timer

    results.append(
        {
            "name": f"zlib-{level}",
            "compression_ratio": float(zlib_compression_ratio),
            "pct_of_ideal": float(
                zlib_compression_ratio / ideal_compression_ratio * 100
            ),
            "encode_time": float(elapsed_zlib),
            "decode_time": float(elapsed_zlib_decode),
            "encode_speed_MBps": float(signal_bytes / elapsed_zlib / 1e6),
            "decode_speed_MBps": float(signal_bytes / elapsed_zlib_decode / 1e6),
        }
    )

    print(f"Zlib (level {level}) compression ratio: {zlib_compression_ratio:.2f}")
    print(
        f"Zlib (level {level}) pct of ideal compression: {zlib_compression_ratio/ideal_compression_ratio*100:.2f}%"
    )
    print(
        f"Time to zlib compress: {elapsed_zlib:.2f} seconds ({signal_bytes/elapsed_zlib/1e6:.2f} MB/s)"
    )
    print(
        f"Time to zlib decompress: {elapsed_zlib_decode:.2f} seconds ({signal_bytes/elapsed_zlib_decode/1e6:.2f} MB/s)"
    )
    print("")

# Test zstandard at different levels
import zstandard as zstd

zstd_levels = [4, 7, 10, 13, 16, 19, 22]
for level in zstd_levels:
    cctx = zstd.ZstdCompressor(level=level)
    timer = time.time()
    compressed = cctx.compress(signal.tobytes())
    elapsed_zstd = time.time() - timer
    zstd_compression_ratio = signal_bytes / len(compressed)

    dctx = zstd.ZstdDecompressor()
    timer = time.time()
    signal_decompressed = np.frombuffer(dctx.decompress(compressed), dtype=signal.dtype)
    elapsed_zstd_decode = time.time() - timer

    results.append(
        {
            "name": f"zstd-{level}",
            "compression_ratio": float(zstd_compression_ratio),
            "pct_of_ideal": float(
                zstd_compression_ratio / ideal_compression_ratio * 100
            ),
            "encode_time": float(elapsed_zstd),
            "decode_time": float(elapsed_zstd_decode),
            "encode_speed_MBps": float(signal_bytes / elapsed_zstd / 1e6),
            "decode_speed_MBps": float(signal_bytes / elapsed_zstd_decode / 1e6),
        }
    )

    print(f"Zstandard (level {level}) compression ratio: {zstd_compression_ratio:.2f}")
    print(
        f"Zstandard (level {level}) pct of ideal compression: {zstd_compression_ratio/ideal_compression_ratio*100:.2f}%"
    )
    print(
        f"Time to zstd compress: {elapsed_zstd:.2f} seconds ({signal_bytes/elapsed_zstd/1e6:.2f} MB/s)"
    )
    print(
        f"Time to zstd decompress: {elapsed_zstd_decode:.2f} seconds ({signal_bytes/elapsed_zstd_decode/1e6:.2f} MB/s)"
    )
    print("")

# Test LZMA
import lzma

timer = time.time()
compressed = lzma.compress(signal.tobytes(), preset=3)
elapsed_lzma = time.time() - timer
lzma_compression_ratio = signal_bytes / len(compressed)

timer = time.time()
signal_decompressed = np.frombuffer(lzma.decompress(compressed), dtype=signal.dtype)
elapsed_lzma_decode = time.time() - timer

results.append(
    {
        "name": "lzma",
        "compression_ratio": float(lzma_compression_ratio),
        "pct_of_ideal": float(lzma_compression_ratio / ideal_compression_ratio * 100),
        "encode_time": float(elapsed_lzma),
        "decode_time": float(elapsed_lzma_decode),
        "encode_speed_MBps": float(signal_bytes / elapsed_lzma / 1e6),
        "decode_speed_MBps": float(signal_bytes / elapsed_lzma_decode / 1e6),
    }
)

print(f"LZMA compression ratio: {lzma_compression_ratio:.2f}")
print(
    f"LZMA pct of ideal compression: {lzma_compression_ratio/ideal_compression_ratio*100:.2f}%"
)
print(
    f"Time to lzma compress: {elapsed_lzma:.2f} seconds ({signal_bytes/elapsed_lzma/1e6:.2f} MB/s)"
)
print(
    f"Time to lzma decompress: {elapsed_lzma_decode:.2f} seconds ({signal_bytes/elapsed_lzma_decode/1e6:.2f} MB/s)"
)

# Save results to JSON
output = {
    "ideal_compression_ratio": float(ideal_compression_ratio),
    "results": results,
}

print('Saving benchmark results')
if not os.path.exists("benchmark_output"):
    os.makedirs("benchmark_output")

with open("benchmark_output/benchmark.json", "w") as f:
    json.dump(output, f, indent=2)

# Create visualization
algorithms = [r["name"] for r in results]
metrics = {
    "Compression Ratio": [r["compression_ratio"] for r in results],
    "Encode Speed (MB/s)": [r["encode_speed_MBps"] for r in results],
    "Decode Speed (MB/s)": [r["decode_speed_MBps"] for r in results],
}

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
fig.suptitle("Compression Algorithm Comparison")

# Plot compression ratios
bars1 = ax1.bar(algorithms, metrics["Compression Ratio"])
ax1.set_title("Compression Ratio (higher is better)")
ax1.set_ylabel("Compression Ratio")
ax1.axhline(y=ideal_compression_ratio, color="r", linestyle="--", label="Ideal")
ax1.legend()

# Plot encode speed
bars2 = ax2.bar(algorithms, metrics["Encode Speed (MB/s)"])
ax2.set_title("Encode Speed (higher is better)")
ax2.set_ylabel("Encode Speed (MB/s)")

# Plot decode speed
bars3 = ax3.bar(algorithms, metrics["Decode Speed (MB/s)"])
ax3.set_title("Decode Speed (higher is better)")
ax3.set_ylabel("Decode Speed (MB/s)")


# Add value labels
def autolabel(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


autolabel(bars1, ax1)
autolabel(bars2, ax2)
autolabel(bars3, ax3)

plt.tight_layout()
plt.savefig("benchmark_output/benchmark.png")
