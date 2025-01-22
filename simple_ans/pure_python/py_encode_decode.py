import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simple_ans.EncodedSignal import EncodedSignal
from simple_ans.choose_symbol_counts import choose_symbol_counts


def py_ans_encode(signal: np.ndarray, *, index_size: int = 2**16) -> EncodedSignal:
    """Encode a signal using Asymmetric Numeral Systems (ANS).

    Args:
        signal: Input signal to encode as a 1D numpy array. Must be int32, int16, uint32, or uint16.
        index_size: Size of the index table. (default: 2**16).
        Must be a power of 2.
        Must be at least as large as the number of unique symbols in the input signal.

    Returns:
        An EncodedSignal object containing the encoded data.
    """
    if signal.dtype not in [np.int32, np.int16, np.uint32, np.uint16]:
        raise TypeError("Input signal must be int32, int16, uint32, or uint16")
    assert signal.ndim == 1, "Input signal must be a 1D array"

    # Get symbol counts and values using the main implementation
    signal_length = len(signal)
    vals, counts = np.unique(signal, return_counts=True)
    vals = np.array(vals, dtype=signal.dtype)
    probs = counts / np.sum(counts)
    S = len(vals)
    if S > index_size:
        raise ValueError(f"Number of unique symbols cannot be greater than L, got {S} unique symbols and L = {index_size}")

    symbol_counts = choose_symbol_counts(probs, index_size)
    symbol_values = vals

    # Calculate L and verify it's a power of 2
    L = np.sum(symbol_counts)
    if L & (L - 1) != 0:
        raise ValueError("L must be a power of 2")

    # Pre-compute cumulative sums
    C = np.zeros(len(symbol_counts), dtype=np.uint32)
    for i in range(1, len(symbol_counts)):
        C[i] = C[i - 1] + symbol_counts[i - 1]

    # Create symbol index lookup
    symbol_index_lookup = {symbol_values[i]: i for i in range(len(symbol_values))}

    # Create bit count table
    max_f_s = np.max(symbol_counts)
    bit_count_table = np.zeros(2 * max_f_s, dtype=np.uint32)
    for i in range(1, 2 * max_f_s):
        d = 0
        while i * (1 << d) < L:
            d += 1
        bit_count_table[i] = d

    # Initialize state and bits list
    state = np.uint64(L)  # Use uint64 for state
    bits = []

    # Encode each symbol in reverse order
    for i in range(signal_length):
        symbol = signal[i]
        s_ind = symbol_index_lookup[symbol]
        L_s = symbol_counts[s_ind]

        state_normalized = state
        while state_normalized >= 2 * L_s:
            bits.append(int(state_normalized & 1))
            state_normalized >>= 1

        state = L + C[s_ind] + state_normalized - L_s

    # Pack bits into bitstream
    bitstream = pack_bitstream(bits)

    return EncodedSignal(
        state=int(state),
        bitstream=bitstream,
        num_bits=len(bits),
        symbol_counts=symbol_counts,
        symbol_values=symbol_values,
        signal_length=signal_length
    )


def py_ans_decode(E: EncodedSignal) -> np.ndarray:
    """Decode an ANS-encoded signal.

    Args:
        E: EncodedSignal object containing the encoded data.

    Returns:
        Decoded signal as a numpy array.
    """
    # Calculate L and verify it's a power of 2
    L = int(np.sum(E.symbol_counts))  # Convert to Python int
    if L & (L - 1) != 0:
        raise ValueError("L must be a power of 2")

    # Pre-compute cumulative sums
    C = np.zeros(len(E.symbol_counts), dtype=np.uint32)
    for i in range(1, len(E.symbol_counts)):
        C[i] = C[i - 1] + E.symbol_counts[i - 1]

    # Create symbol lookup table
    symbol_lookup = np.zeros(L, dtype=np.uint32)
    for s in range(len(E.symbol_counts)):
        for j in range(E.symbol_counts[s]):
            symbol_lookup[C[s] + j] = s

    # Create state update table
    state_update = np.zeros(L, dtype=np.uint64)
    for i in range(L):
        s = symbol_lookup[i]
        f_s = E.symbol_counts[s]
        state_update[i] = f_s + i - C[s]
        assert f_s <= state_update[i] < 2 * f_s, f"State out of bounds: {state_update[i]}"

    # Create bit count table
    max_f_s = np.max(E.symbol_counts)
    bit_count_table = np.zeros(2 * max_f_s, dtype=np.uint32)
    for i in range(1, 2 * max_f_s):
        d = 0
        while (i << d) < L:
            d += 1
        bit_count_table[i] = d

    # Unpack bitstream to bits
    bits = unpack_bitstream(E.bitstream, E.num_bits)
    bit_pos = np.int32(E.num_bits - 1)

    # Initialize output array
    output = np.zeros(E.signal_length, dtype=E.symbol_values.dtype)
    state = np.uint32(E.state)

    # Decode symbols in reverse order
    for i in range(E.signal_length):
        assert L <= state < 2 * L, f"State out of bounds: {state}"
        # Find symbol
        s_ind = symbol_lookup[state - L]
        output[E.signal_length - i - 1] = E.symbol_values[s_ind]

        state_2 = state_update[state - L]
        f_s = E.symbol_counts[s_ind]
        assert f_s <= state_2 < 2 * f_s, f"State out of bounds: {state_2}"
        d = bit_count_table[state_2]
        new_state = state_2

        # Read d bits from bitstream
        if d > 0:
            bit_word = bits[bit_pos - (d - 1):bit_pos + 1]
            bit_pos -= d
            for jj in range(d):
                new_state <<= 1
                new_state += bit_word[d - 1 - jj]

        state = new_state

    return output


def pack_bitstream(bits: list) -> bytes:
    """Pack a list of bits into a bytes object.

    Args:
        bits: List of bits (0s and 1s) to pack.

    Returns:
        Packed bits as a bytes object.
    """
    num_bits = len(bits)
    num_bytes = (num_bits + 7) // 8
    x = np.zeros((num_bytes), dtype=np.uint8)
    for i in range(len(bits)):
        if bits[i]:
            a = i // 8
            b = i % 8
            x[a] += 2 ** b
    return x.tobytes()

def unpack_bitstream(bitstream: bytes, num_bits: int) -> list:
    """Unpack a bitstream back into a list of bits.

    Args:
        bitstream: Packed bits as a bytes object.
        num_bits: Number of bits to unpack.

    Returns:
        List of unpacked bits (0s and 1s).
    """
    x = np.frombuffer(bitstream, dtype=np.uint8)
    bits = []
    for i in range(num_bits):
        a = i // 8
        b = i % 8
        bits.append((x[a] >> b) % 2)
    return bits


if __name__ == '__main__':
    from simple_ans import ans_encode, ans_decode

    signals = []

    # test 1 - Basic test with small array
    proportions = [1, 2, 3]
    probs = np.array(proportions) / np.sum(proportions)
    signal_length = 20
    signal = np.random.choice(len(proportions), signal_length, p=probs).astype(np.uint16)
    signals.append(signal)

    # test 2 - Large uniform
    num_symbols = 10
    signal_length = 10000
    signal = np.random.randint(num_symbols, size=signal_length).astype(np.int32)
    signals.append(signal)

    # test 3 - Skewed - mostly zeros, and some other values
    signal_length = 10000
    proportions = [1000, 1, 2, 5, 10]
    probs = np.array(proportions) / np.sum(proportions)
    signal = np.random.choice(len(proportions), signal_length, p=probs).astype(np.int16)
    signals.append(signal)

    # test 4 - Negative numbers
    signal = np.random.randint(-10, 10, size=1000).astype(np.int32)
    signals.append(signal)

    # test 5 - Binary signal
    signal = np.random.choice([0, 1], size=50000, p=[0.3, 0.7]).astype(np.uint16)
    signals.append(signal)

    # test 6 - Constant signal
    signal = np.full(1000, 5).astype(np.int16)  # Array of 1000 fives
    signals.append(signal)

    for i, signal in enumerate(signals):
        print(f'Test {i + 1}')
        encoded = py_ans_encode(signal)
        encoded2 = ans_encode(signal)
        assert encoded.state == encoded2.state, f"States do not match for test {i + 1}"
        assert encoded.num_bits == encoded2.num_bits, f"Number of bits do not match for test {i + 1}"
        bits1 = unpack_bitstream(encoded.bitstream, encoded.num_bits)
        bits2 = unpack_bitstream(encoded2.bitstream, encoded2.num_bits)
        assert np.all(bits1 == bits2), f"Bitstreams do not match for test {i + 1}"
        decoded3 = ans_decode(encoded)
        assert np.all(signal == decoded3), f"Test {i + 1} failed"
        decoded = py_ans_decode(encoded)
        assert np.all(signal == decoded), f"Test {i + 1} failed"
        decoded2 = py_ans_decode(encoded2)
        assert np.all(signal == decoded2), f"Test {i + 1} failed"

    print("All tests passed!")
