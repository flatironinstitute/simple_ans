import time
import numpy as np
from simple_ans import ans_encode, ans_decode

# Generate random test data from normal distribution
n = 10_000_000
# Generate signal with normal distribution, ensuring positive values
signal = np.round(np.random.normal(0, 1, n) * 1).astype(np.int16)
signal_bytes = len(signal.tobytes())

# Calculate ideal compression ratio
vals, counts = np.unique(signal, return_counts=True)
probs = counts / len(signal)
ideal_compression_ratio = 16 / -np.sum(probs * np.log2(probs))
print(f"Ideal compression ratio: {ideal_compression_ratio}")

num_trials = 10
encode_times = []
decode_times = []


def warmup(elapsed=3):
    timer = time.time()
    while (time.time() - timer) < elapsed:
        ans_encode(signal=signal)

warmup()


for _ in range(num_trials):
    # Test simple_ans
    timer = time.time()
    encoded = ans_encode(signal=signal)
    elapsed_encode = time.time() - timer
    encode_times.append(elapsed_encode)

    timer = time.time()
    signal_decoded = ans_decode(encoded)
    elapsed_decode = time.time() - timer
    decode_times.append(elapsed_decode)

    assert len(signal_decoded) == len(signal)
    assert np.all(signal_decoded == signal)
    print("Decoded signal matches original signal")

    # 64 bits per bitstream word, 32 bits for state, 32 bits per symbol count, 32 bits per symbol value, 32 bits for num_bits, 32 bits for signal_length
    compressed_size_bits = encoded.size() * 8
    compression_ratio = signal_bytes * 8 / compressed_size_bits

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

print(
    f"Average encode time: {np.mean(encode_times):.2f} seconds (MB/s: {signal_bytes/np.mean(encode_times)/1e6:.2f})"
)
print(
    f"Average decode time: {np.mean(decode_times):.2f} seconds (MB/s: {signal_bytes/np.mean(decode_times)/1e6:.2f})"
)
