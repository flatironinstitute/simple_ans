# simple_ans

A Python package that provides **lossless** compression of integer datasets through [Asymmetric Numeral Systems (ANS)](https://ieeexplore.ieee.org/document/7170048), implemented in C++ with pybind11 bindings.

I used the following to guide the implementation:
* [https://graphallthethings.com/posts/streaming-ans-explained/](https://graphallthethings.com/posts/streaming-ans-explained/).
* [https://bjlkeng.io/posts/lossless-compression-with-asymmetric-numeral-systems/](https://bjlkeng.io/posts/lossless-compression-with-asymmetric-numeral-systems/)
* [https://kedartatwawadi.github.io/post--ANS/](https://kedartatwawadi.github.io/post--ANS/)

While there are certainly many ANS implementations that are parts of other packages, this one strives to be as simple as possible, with the [C++ implementation](./simple_ans/cpp) being just a small amount of code in a single file. The Python interface is also simple and easy to use. At the same time it attempts to be as efficient as possible both in terms of compression ratio and encoding/decoding speed.

[Technical overview of ANS and Streaming ANS](./doc/technical_overview.md)

## Installation

First, install the required dependencies:

```bash
pip install pybind11 numpy
```

Then install the package:

```bash
pip install .
```

Or install from source:

```bash
cd simple_ans
pip install -e .
```

## Usage

This package is designed for compressing quantized numerical data.

```python
import numpy as np
from simple_ans import ans_encode, ans_decode

# Example: Compressing quantized Gaussian data
# Generate sample data following normal distribution
n_samples = 10000
# Generate Gaussian data, scale by 4, and quantize to integers
signal = np.round(np.random.normal(0, 1, n_samples) * 4).astype(np.int32)

# Encode (automatically determines optimal symbol counts)
encoded = ans_encode(signal)

# Decode
decoded = ans_decode(encoded)

# Verify
assert np.all(decoded == signal)

# Get compression stats
original_size = signal.nbytes
compressed_size = encoded.size()  # in bits
compression_ratio = original_size / compressed_size
print(f"Compression ratio: {compression_ratio:.2f}x")
```

## Simple benchmark

You can run a very simple benchmark that compares simple_ans with `zlib`, `zstandard`, and `lzma` at various compression levels for a toy dataset of quantized Gaussian noise. See [devel/benchmark.py](./devel/benchmark.py) and [devel/benchmark_ans_only.py](./devel/benchmark_ans_only.py).

The benchmark.py also runs in a CI environment and produces the following graph:

![Benchmark](https://github.com/magland/simple_ans/blob/benchmark-results/benchmark-results/benchmark.png?raw=true)

We see that for this example, the ANS-based compression ratio is higher than the other methods, almost reaching the theoretical ideal. The encode rate in MB/s is better than most other methods, whereas the decode rate is slower than the other methods. I think in principle, we should be able to speed up the decoding. Let me know if you have ideas for this.

## Author

Jeremy Magland, Center for Computational Mathematics, Flatiron Institute
