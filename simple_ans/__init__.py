from ._simple_ans import (
    encode_int16 as _encode_int16,
    decode_int16 as _decode_int16,
    encode_int32 as _encode_int32,
    decode_int32 as _decode_int32,
    encode_uint16 as _encode_uint16,
    decode_uint16 as _decode_uint16,
    encode_uint32 as _encode_uint32,
    decode_uint32 as _decode_uint32,
)
from dataclasses import dataclass
from typing import Union
import numpy as np

from .encode_decode import ans_encode as ans_encode_new
from .encode_decode import ans_decode as ans_decode_new
from .choose_symbol_counts import choose_symbol_counts

__version__ = "0.2.3"


@dataclass
class EncodedSignal:
    """Container for ANS-encoded signal data.

    Attributes:
        state (int): Integer representing the final encoder state
        bitstream (bytes): Bytes object containing the encoded bitstream
        num_bits (int): Number of bits used in the encoding (may be somewhat less than len(bitstream) * 64)
        symbol_counts (numpy.ndarray): uint32 numpy array containing frequency counts for each symbol
        symbol_values (numpy.ndarray): int32, int16, uint32, or uint16 numpy array containing the actual symbol values
        signal_length (int): Length of the original signal in number of elements
    """

    state: int
    bitstream: bytes  # uint64 array
    num_bits: int
    symbol_counts: np.ndarray  # uint32 array
    symbol_values: np.ndarray  # int32 or int16 array
    signal_length: int

    def size(self) -> int:
        """Return the size of the encoded signal in bytes."""
        return (
            len(self.bitstream) + 4 * self.symbol_counts.nbytes + self.symbol_values.nbytes + 96
        )

    def __post_init__(self):
        """Validate and convert data types after initialization."""
        # Convert lists to numpy arrays if needed
        if not isinstance(self.symbol_counts, np.ndarray):
            self.symbol_counts = np.array(self.symbol_counts, dtype=np.uint32)
        if not isinstance(self.symbol_values, np.ndarray):
            # Keep original dtype if it's already a numpy array, otherwise default to int32
            if isinstance(self.symbol_values, np.ndarray):
                dtype = self.symbol_values.dtype
            else:
                dtype = np.dtype(np.int32)
            self.symbol_values = np.array(self.symbol_values, dtype=dtype)
        if not isinstance(self.bitstream, bytes):
            raise TypeError("bitstream must be a bytes object")
        # length of bitstream must be divisible by 8
        if len(self.bitstream) % 8 != 0:
            raise ValueError("bitstream length must be a multiple of 8")

        # Validate types and sizes
        if not isinstance(self.state, int):
            raise TypeError("state must be an integer")
        if not isinstance(self.num_bits, int):
            raise TypeError("num_bits must be an integer")
        if not isinstance(self.signal_length, int):
            raise TypeError("signal_length must be an integer")

        assert (
            self.symbol_counts.size == self.symbol_values.size
        ), "symbol_counts and symbol_values must have the same size"
        assert self.symbol_counts.dtype == np.uint32, "symbol_counts must be uint32"
        assert self.symbol_values.dtype in [
            np.int32,
            np.int16,
            np.uint32,
            np.uint16,
        ], "symbol_values must be int32, int16, uint32, or uint16"


def determine_symbol_counts_and_values(
    signal: Union[np.ndarray, list],
    index_length: Union[int, None] = None,
    dtype: Union[np.dtype, None] = None,
) -> tuple:
    """Determine symbol counts and unique values from input data.

    Args:
        signal: List or numpy array of integers representing the signal
        index_length: Length of the ANS index (must be a power of 2). If None, automatically
            determines minimum valid power of 2 based on number of unique symbols.
        dtype: Data type for the signal values (np.int32, np.int16, np.uint32, or np.uint16)

    Returns:
        tuple: A pair of numpy arrays (symbol_counts, symbol_values) where:
            - symbol_counts is a uint32 array of frequency counts for each symbol
            - symbol_values is an array of the corresponding unique symbol values with specified dtype
    """
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty")

    if not isinstance(signal, np.ndarray):
        signal = np.array(signal, dtype=dtype)

    if dtype is None:
        dtype = signal.dtype

    if dtype not in [np.int32, np.int16, np.uint32, np.uint16]:
        raise ValueError("dtype must be np.int32, np.int16, np.uint32, or np.uint16")
    assert signal.dtype == dtype

    # Get unique values and count frequencies using numpy
    unique_values, counts = np.unique(signal, return_counts=True)
    total = counts.sum()
    num_symbols = len(unique_values)

    def is_power_of_two(n: int) -> bool:
        """Check if a number is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    if index_length is None:
        # Start at 2^16 and increase until we have enough space for all symbols
        index_length = 2**16
        while index_length < num_symbols:
            index_length *= 2
    elif not isinstance(index_length, int) or index_length <= 0:
        raise ValueError("Index length must be a positive integer")
    elif not is_power_of_two(index_length):
        raise ValueError("Index length must be a power of 2")
    if index_length < num_symbols:
        raise ValueError(
            f"Index length ({index_length}) must be greater than or equal to the number of unique symbols ({num_symbols})"
        )

    # Convert to proportions as numpy array
    proportions = counts.astype(np.float64) / total

    # Use existing choose_symbol_counts to convert proportions to integer counts
    symbol_counts = choose_symbol_counts(proportions, index_length)

    return symbol_counts, unique_values.astype(dtype)


def ans_encode(
    signal: Union[np.ndarray, list],
    symbol_counts: Union[np.ndarray, list, None] = None,
    symbol_values: Union[np.ndarray, list, None] = None,
    dtype: Union[np.dtype, None] = None,
) -> EncodedSignal:
    """Encode a signal using ANS (Asymmetric Numeral Systems).

    Args:
        signal: numpy array or list of signal to encode
        symbol_counts: uint32 numpy array of symbol counts, defaults to None
        symbol_values: numpy array of symbol values matching dtype, defaults to None

    Returns:
        EncodedSignal: Object containing all encoding information
    """
    if dtype is None:
        if isinstance(signal, np.ndarray):
            dtype = signal.dtype
        else:
            dtype = np.int16  # type: ignore
    else:
        if dtype not in [np.int32, np.int16, np.uint32, np.uint16]:
            raise ValueError(
                "dtype must be np.int32, np.int16, np.uint32, or np.uint16"
            )

    if not isinstance(signal, np.ndarray):
        signal = np.array(signal, dtype=dtype)
    if signal.dtype != dtype:
        raise ValueError(f"Signal must be of type {dtype}")

    # If either is None, determine both
    if symbol_counts is None:
        if symbol_values is not None:
            raise ValueError(
                "If symbol_values is provided, symbol_counts must also be provided"
            )
        auto_counts, auto_values = determine_symbol_counts_and_values(
            signal, dtype=dtype  # type: ignore
        )
        symbol_counts = auto_counts
        symbol_values = auto_values
    if symbol_values is None:
        raise ValueError(
            "If symbol_counts is provided, symbol_values must also be provided"
        )

    # Ensure arrays are the right type
    if not isinstance(symbol_counts, np.ndarray):
        symbol_counts = np.array(symbol_counts, dtype=np.uint32)
    if not isinstance(symbol_values, np.ndarray):
        symbol_values = np.array(symbol_values, dtype=dtype)
    if symbol_counts.dtype != np.uint32:
        raise ValueError("Symbol counts must be of type uint32")
    if symbol_values.dtype != dtype:
        raise ValueError(f"Symbol values must be of type {dtype}")

    # Use appropriate encode function based on dtype
    if dtype == np.int32:
        encoded = _encode_int32(signal, symbol_counts, symbol_values)
    elif dtype == np.int16:
        encoded = _encode_int16(signal, symbol_counts, symbol_values)
    elif dtype == np.uint32:
        encoded = _encode_uint32(signal, symbol_counts, symbol_values)
    else:  # dtype == np.uint16
        encoded = _encode_uint16(signal, symbol_counts, symbol_values)

    return EncodedSignal(
        state=encoded.state,
        bitstream=encoded.bitstream,
        num_bits=encoded.num_bits,
        symbol_counts=symbol_counts,  # Already numpy array from above
        symbol_values=symbol_values,  # Already numpy array from above
        signal_length=len(signal),
    )


def ans_decode(encoded: EncodedSignal) -> np.ndarray:
    """Decode an ANS-encoded signal.

    Args:
        encoded: EncodedSignal object containing the encoded data and metadata

    Returns:
        decoded_signal: numpy array of the decoded signal with same dtype as original
    """
    # Use appropriate decode function based on symbol_values dtype
    if encoded.symbol_values.dtype == np.int32:
        return _decode_int32(
            encoded.state,
            encoded.bitstream,
            encoded.num_bits,
            encoded.symbol_counts,
            encoded.symbol_values,
            encoded.signal_length,
        )
    elif encoded.symbol_values.dtype == np.int16:
        return _decode_int16(
            encoded.state,
            encoded.bitstream,
            encoded.num_bits,
            encoded.symbol_counts,
            encoded.symbol_values,
            encoded.signal_length,
        )
    elif encoded.symbol_values.dtype == np.uint32:
        return _decode_uint32(
            encoded.state,
            encoded.bitstream,
            encoded.num_bits,
            encoded.symbol_counts,
            encoded.symbol_values,
            encoded.signal_length,
        )
    else:  # dtype == np.uint16
        return _decode_uint16(
            encoded.state,
            encoded.bitstream,
            encoded.num_bits,
            encoded.symbol_counts,
            encoded.symbol_values,
            encoded.signal_length,
        )


__all__ = [
    "ans_encode",
    "ans_decode",
    "ans_encode_new",
    "ans_decode_new",
    "choose_symbol_counts",
    "determine_symbol_counts_and_values",
    "EncodedSignal",
]
