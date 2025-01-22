import numpy as np
from ..EncodedSignal import EncodedSignal
from ..choose_symbol_counts import choose_symbol_counts


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
    ...


def py_ans_decode(E: EncodedSignal) -> np.ndarray:
    """Decode an ANS-encoded signal.

    Args:
        E: EncodedSignal object containing the encoded data.

    Returns:
        Decoded signal as a numpy array.
    """
    ...


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
            # Pack bits from MSB to LSB within each byte
            x[a] += 2 ** (7 - b)
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
        # Unpack bits from MSB to LSB within each byte
        bits.append((x[a] >> (7 - b)) % 2)
    return bits


if __name__ == '__main__':
    # Test 1: Basic test with small array
    print("Test 1: Basic test")
    proportions = [1, 2, 3]
    probs = np.array(proportions) / np.sum(proportions)
    signal_length = 10
    signal = np.random.choice(len(proportions), signal_length, p=probs).astype(np.uint16)
    encoded = py_ans_encode(signal)
    decoded = py_ans_decode(encoded)
    assert np.all(signal == decoded), "Basic test failed"

    # Test 2: Large uniform
    print("Test 2: Large uniform")
    num_symbols = 10
    signal_length = 10000
    signal = np.random.randint(num_symbols, size=signal_length).astype(np.int32)
    encoded = py_ans_encode(signal)
    decoded = py_ans_decode(encoded)
    assert np.all(signal == decoded), "Large uniform test failed"

    # Test 3: Skewed - mostly zeros, and some other values
    print("Test 3: Skewed")
    signal_length = 100000
    proportions = [1000, 1, 2, 5, 10]
    probs = np.array(proportions) / np.sum(proportions)
    signal = np.random.choice(len(proportions), signal_length, p=probs).astype(np.int16)
    encoded = py_ans_encode(signal)
    decoded = py_ans_decode(encoded)
    assert np.all(signal == decoded), "Skewed test failed"

    # Test 4: Negative numbers
    print("Test 4: Negative numbers")
    signal = np.random.randint(-10, 10, size=1000).astype(np.int32)
    encoded = py_ans_encode(signal)
    decoded = py_ans_decode(encoded)
    assert np.all(signal == decoded), "Negative numbers test failed"

    # Test 5: Binary signal
    print("Test 5: Binary signal")
    signal = np.random.choice([0, 1], size=50000, p=[0.3, 0.7]).astype(np.uint16)
    encoded = py_ans_encode(signal)
    decoded = py_ans_decode(encoded)
    assert np.all(signal == decoded), "Binary signal test failed"

    # Test 6: Constant signal
    print("Test 6: Constant signal")
    signal = np.full(1000, 5).astype(np.int16)  # Array of 1000 fives
    encoded = py_ans_encode(signal)
    decoded = py_ans_decode(encoded)
    assert np.all(signal == decoded), "Constant signal test failed"

    print("All tests passed!")
