import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from simple_ans import ans_encode, ans_decode
import zstandard as zstd
import zlib

def calculate_ideal_compression_ratio(signal):
    vals, counts = np.unique(signal, return_counts=True)
    probs = counts / len(signal)
    return signal.itemsize * 8 / -np.sum(probs * np.log2(probs))

def get_compression_stats(signal):
    ideal_ratio = calculate_ideal_compression_ratio(signal)

    # Get original size in bits
    original_bits = len(signal.tobytes()) * 8
    signal_bytes = len(signal.tobytes())

    # Test simple_ans
    # First encode to get initial values
    encoded = ans_encode(signal=signal)
    compressed_size_bits = encoded.size() * 8
    simple_ans_ratio = original_bits / compressed_size_bits

    # Encode timing
    timer = time.time()
    num_runs = 0
    while time.time() - timer < 0.3:  # Run for at least 0.3 second
        _ = ans_encode(signal=signal)
        num_runs += 1
    elapsed_encode = (time.time() - timer) / num_runs

    # Decode timing
    timer = time.time()
    num_runs = 0
    while time.time() - timer < 0.3:  # Run for at least 0.3 second
        _ = ans_decode(encoded)
        num_runs += 1
    elapsed_decode = (time.time() - timer) / num_runs

    # Test zstd-22
    # First compress to get initial values
    cctx = zstd.ZstdCompressor(level=22)
    compressed = cctx.compress(signal.tobytes())
    zstd_ratio = original_bits / (len(compressed) * 8)

    # Encode timing
    timer = time.time()
    num_runs = 0
    while time.time() - timer < 0.3:  # Run for at least 0.3 second
        _ = cctx.compress(signal.tobytes())
        num_runs += 1
    elapsed_zstd_encode = (time.time() - timer) / num_runs

    # Decode timing
    dctx = zstd.ZstdDecompressor()
    timer = time.time()
    num_runs = 0
    while time.time() - timer < 0.3:  # Run for at least 0.3 second
        _ = dctx.decompress(compressed)
        num_runs += 1
    elapsed_zstd_decode = (time.time() - timer) / num_runs

    # Test zlib-9
    # First compress to get initial values
    compressed = zlib.compress(signal.tobytes(), level=9)
    zlib_ratio = original_bits / (len(compressed) * 8)

    # Encode timing
    timer = time.time()
    num_runs = 0
    while time.time() - timer < 0.3:  # Run for at least 0.3 second
        _ = zlib.compress(signal.tobytes(), level=9)
        num_runs += 1
    elapsed_zlib_encode = (time.time() - timer) / num_runs

    # Decode timing
    timer = time.time()
    num_runs = 0
    while time.time() - timer < 0.3:  # Run for at least 0.3 second
        _ = zlib.decompress(compressed)
        num_runs += 1
    elapsed_zlib_decode = (time.time() - timer) / num_runs

    return {
        'ideal': float(ideal_ratio),
        'simple_ans': {
            'ratio': float(simple_ans_ratio),
            'encode_MBps': float(signal_bytes / elapsed_encode / 1e6),
            'decode_MBps': float(signal_bytes / elapsed_decode / 1e6)
        },
        'zstd-22': {
            'ratio': float(zstd_ratio),
            'encode_MBps': float(signal_bytes / elapsed_zstd_encode / 1e6),
            'decode_MBps': float(signal_bytes / elapsed_zstd_decode / 1e6)
        },
        'zlib-9': {
            'ratio': float(zlib_ratio),
            'encode_MBps': float(signal_bytes / elapsed_zlib_encode / 1e6),
            'decode_MBps': float(signal_bytes / elapsed_zlib_decode / 1e6)
        }
    }

# Generate test data and run benchmarks
n = 100_000  # Number of samples for each test
results = []
distribution_groups = {'bernoulli': [], 'gaussian': [], 'poisson': []}

print("\nRunning benchmarks with sample size:", n)

# Test Bernoulli distributions
print("\nTesting Bernoulli distributions:")
for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
    print(f"  Processing Bernoulli p={p}")
    # Generate as uint8 for correct ideal ratio and compression
    signal = np.random.binomial(1, p, n).astype(np.uint8)
    stats = get_compression_stats(signal)
    results.append({
        'name': f'bernoulli_p{p}',
        'distribution': 'bernoulli',
        'params': {'p': p},
        'ratios': stats
    })

# Test Gaussian distributions with different quantization steps
print("\nTesting Gaussian distributions:")
for step in [0.5, 0.3, 0.1, 0.05]:
    print(f"  Processing Gaussian step={step}")
    signal = np.round(np.random.normal(0, 1, n) / step).astype(np.int32)
    stats = get_compression_stats(signal)
    results.append({
        'name': f'gaussian_step{step}',
        'distribution': 'gaussian',
        'params': {'step': step},
        'ratios': stats
    })

# Test Poisson distributions
print("\nTesting Poisson distributions:")
for lam in [0.5, 1, 2, 5, 10]:
    print(f"  Processing Poisson lambda={lam}")
    signal = np.random.poisson(lam, n).astype(np.int32)
    stats = get_compression_stats(signal)
    results.append({
        'name': f'poisson_lambda{lam}',
        'distribution': 'poisson',
        'params': {'lambda': lam},
        'ratios': stats
    })

print("\nSaving results...")
# Save results to JSON
if not os.path.exists("benchmark_output"):
    print("Creating benchmark_output directory")
    os.makedirs("benchmark_output")

print("Writing benchmark2.json")
with open("benchmark_output/benchmark2.json", "w") as f:
    json.dump({'results': results}, f, indent=2)

print("\nCreating visualization...")
# Create compression ratio visualization
fig, ax = plt.subplots(figsize=(10, 15))

# Group results by distribution and calculate y positions with gaps
y_positions = []
names = []
ideal_ratios = []
simple_ans_ratios = []
zstd_ratios = []
zlib_ratios = []

current_pos = 0
for dist_type in ['bernoulli', 'gaussian', 'poisson']:
    dist_results = [r for r in results if r['distribution'] == dist_type]
    for r in dist_results:
        y_positions.append(current_pos)
        names.append(r['name'])
        ideal_ratios.append(r['ratios']['ideal'])
        simple_ans_ratios.append(r['ratios']['simple_ans']['ratio'])
        zstd_ratios.append(r['ratios']['zstd-22']['ratio'])
        zlib_ratios.append(r['ratios']['zlib-9']['ratio'])
        current_pos += 1
    current_pos += 1  # Add gap between distribution groups

width = 0.2

# Create horizontal bars
ax.barh([y + 1.5*width for y in y_positions], ideal_ratios, width, label='Ideal', color='lightgray')
ax.barh([y + 0.5*width for y in y_positions], simple_ans_ratios, width, label='simple_ans', color='skyblue')
ax.barh([y - 0.5*width for y in y_positions], zstd_ratios, width, label='zstd-22', color='lightgreen')
ax.barh([y - 1.5*width for y in y_positions], zlib_ratios, width, label='zlib-9', color='lightpink')

# Customize plot
ax.set_xlabel('Compression Ratio')
ax.set_title('Compression Ratio Comparison Across Different Distributions')
ax.set_yticks(y_positions)
ax.set_yticklabels(names)

ax.legend()

def add_value_labels_ratio(rects):
    for rect in rects:
        width = rect.get_width()
        ax.annotate(f'{width:.2f}',
                   xy=(width, rect.get_y() + rect.get_height()/2),
                   xytext=(3, 0),  # 3 points horizontal offset
                   textcoords="offset points",
                   ha='left', va='center')

add_value_labels_ratio(ax.containers[0])  # Ideal
add_value_labels_ratio(ax.containers[1])  # simple_ans
add_value_labels_ratio(ax.containers[2])  # zstd-22
add_value_labels_ratio(ax.containers[3])  # zlib-9

plt.tight_layout()
plt.savefig("benchmark_output/benchmark2_compression_ratio.png")
plt.close()

# Create encode rate visualization
fig, ax = plt.subplots(figsize=(10, 15))

# Prepare data for plotting
simple_ans_encode = [r['ratios']['simple_ans']['encode_MBps'] for r in results]
zstd_encode = [r['ratios']['zstd-22']['encode_MBps'] for r in results]

# Create horizontal bars
ax.barh([y + width for y in y_positions], simple_ans_encode, width, label='simple_ans', color='skyblue')
ax.barh(y_positions, zstd_encode, width, label='zstd-22', color='lightgreen')
ax.barh([y - width for y in y_positions], [r['ratios']['zlib-9']['encode_MBps'] for r in results], width, label='zlib-9', color='lightpink')

# Customize plot
ax.set_xlabel('Encode Speed (MB/s)')
ax.set_title('Encode Speed Comparison Across Different Distributions')
ax.set_yticks(y_positions)
ax.set_yticklabels(names)
ax.legend()

def add_value_labels_speed(rects):
    for rect in rects:
        width = rect.get_width()
        ax.annotate(f'{width:.1f}',
                   xy=(width, rect.get_y() + rect.get_height()/2),
                   xytext=(3, 0),  # 3 points horizontal offset
                   textcoords="offset points",
                   ha='left', va='center')

add_value_labels_speed(ax.containers[0])  # simple_ans
add_value_labels_speed(ax.containers[1])  # zstd-22
add_value_labels_speed(ax.containers[2])  # zlib-9

plt.tight_layout()
plt.savefig("benchmark_output/benchmark2_encode_rate.png")
plt.close()

# Create decode rate visualization
fig, ax = plt.subplots(figsize=(10, 15))

# Prepare data for plotting
simple_ans_decode = [r['ratios']['simple_ans']['decode_MBps'] for r in results]
zstd_decode = [r['ratios']['zstd-22']['decode_MBps'] for r in results]

# Create horizontal bars
ax.barh([y + width for y in y_positions], simple_ans_decode, width, label='simple_ans', color='skyblue')
ax.barh(y_positions, zstd_decode, width, label='zstd-22', color='lightgreen')
ax.barh([y - width for y in y_positions], [r['ratios']['zlib-9']['decode_MBps'] for r in results], width, label='zlib-9', color='lightpink')

# Customize plot
ax.set_xlabel('Decode Speed (MB/s)')
ax.set_title('Decode Speed Comparison Across Different Distributions')
ax.set_yticks(y_positions)
ax.set_yticklabels(names)
ax.legend()

add_value_labels_speed(ax.containers[0])  # simple_ans
add_value_labels_speed(ax.containers[1])  # zstd-22
add_value_labels_speed(ax.containers[2])  # zlib-9

plt.tight_layout()
plt.savefig("benchmark_output/benchmark2_decode_rate.png")
plt.close()

print("Saved benchmark plots")
print("\nBenchmark complete!")
