import numpy as np
import pytest
from simple_ans import (
    ans_encode,
    ans_decode,
)
from simple_ans.EncodedSignal import EncodedSignal


def test_encode_decode():
    # Test all supported types
    dtypes = [np.int32, np.int16, np.uint32, np.uint16]
    for dtype in dtypes:
        # Create a simple test signal
        signal = np.array([0, 1, 2, 1, 0], dtype=dtype)

        # Encode
        encoded = ans_encode(signal)
        assert isinstance(
            encoded, EncodedSignal
        ), "Result should be EncodedSignal object"
        assert isinstance(
            encoded.bitstream, bytes
        ), "Encoded bitstream should be bytes"
        assert len(encoded.bitstream) % 8 == 0, "Bitstream length should be multiple of 8"

        # Decode
        decoded = ans_decode(encoded)

        # Verify
        assert np.array_equal(
            signal, decoded
        ), f"Decoded signal does not match original for dtype {dtype}"
    print("Test passed: encode/decode works correctly for all types")


def test_auto_symbol_counts():
    print("Starting test_auto_symbol_counts")
    # Test all supported types
    dtypes = [np.int32, np.int16, np.uint32, np.uint16]
    for dtype in dtypes:
        # Test encoding with auto-determined symbol counts
        signal = np.array([0, 1, 2, 1, 0], dtype=dtype)
        print(f"Testing dtype {dtype}")
        encoded = ans_encode(signal)  # No symbol_counts provided
        print("Signal encoded")
        decoded = ans_decode(encoded)
        print("Signal decoded")
        assert np.array_equal(
            signal, decoded
        ), f"Decoded signal does not match original for dtype {dtype}"
    print("Test passed: auto symbol counts works correctly for all types")


def test_incorrect_data_types():
    # Test with incorrect signal dtype
    signal_float = np.array([0, 1, 2, 1, 0], dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        ans_encode(signal_float)

    print("Test passed: incorrect data types handled correctly")


if __name__ == "__main__":
    test_encode_decode()
    test_auto_symbol_counts()
    test_incorrect_data_types()
