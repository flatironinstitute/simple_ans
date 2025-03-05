from dataclasses import dataclass
import numpy as np


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
    bitstream: bytes
    num_bits: int
    symbol_counts: np.ndarray  # uint32 array
    symbol_values: np.ndarray  # int32, uint32, int16, uint16, or uint8 array
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
            np.uint8,
        ], "symbol_values must be int32, int16, uint32, uint16, or uint8"
