# Asymmetric Numeral Systems (ANS) - Technical Overview

## Introduction and Basic Concept

The core idea of ANS is to encode a sequence of symbols into a single integer state, with each symbol affecting the state in a way that reflects its probability of occurrence. This encoding is reversible, allowing us to recover the original sequence from the coded integer.

Suppose that $\{0, 1, ..., S-1\}$ are the symbols we want to encode, and assume that they occur with relative frequencies $f_0, ..., f_{S-1}$ (positive integers). The goal is to encode a sequence of symbols $(s_1, s_2, ..., s_k)$ into a single integer state $x$.

We make the following assumptions
* The symbols are sampled randomly and independently of one another
* For efficiency in the calculations, the sum of the frequencies is a power of two:
$$L = 2^l = \sum_{i=0}^{S-1} f_i$$

## Basic ANS Encoding/Decoding

### The Symbol Table
To encode our symbols efficiently, we need a way to map between states and symbols that reflects their frequencies. We accomplish this using an infinite table $T$ that maps natural numbers to symbols: $T: \mathbb{N} \to \{0,...,S-1\}$

We construct this table with a periodic pattern where:
- Each period of length $L$ consists of consecutive blocks:
  * $f_0$ occurrences of symbol 0
  * $f_1$ occurrences of symbol 1
  * ...
  * $f_{S-1}$ occurrences of symbol $S-1$
- This pattern repeats every $L$ positions
- For any position $n$, $T(n) = T(n \text{ mod } L)$
- Define $C_s = \sum_{i=0}^{s-1} f_i$ as the cumulative frequency (start position of symbol $s$ in each period)

### The Encoding/Decoding Process

The encoding process works by maintaining a state value $x$ that evolves as we process each symbol. Starting from state $x_0$:

1. To encode symbol $s_1$:
   - Find the $x_0$-th occurrence of symbol $s_1$ (counting from 0)
   - This position index becomes our new state $x_1$
2. Repeat this process for each new symbol, using the previous state to find the next one

This process effectively "pushes" each symbol onto our state value in a way that can be reversed.

With the periodic structured table, the formula for this is

$$x_{i+1} = (x_i // f_{s_i}) \cdot L + C(s_i) + (x_i \text{ mod } f_{s_i})$$

where $//$ denotes integer division and $\text{ mod }$ is the modulo operation.

Notice that for large $x_i$,

$$x_{i+1} \approx x_i \cdot p_s^{-1}$$

where $p_s = f_s / L$ is the probability of symbol $s$. This means that the expected value of $\log(x_i)$ grows by $-\log(p_s)$ at each step, leading to:

$$\lim_{k \to \infty} E(\frac{1}{k}\log(x_k)) = -\sum_{s=0}^{S-1} p_s \log(p_s).$$

Or in other words, if $H$ is the average number of bits required to encode a single symbol in the sequence, then

$$H = -\sum_{s=0}^{S-1} p_s \log(p_s)$$

which matches the Shannon entropy of the symbol distribution. In other words, the integer $x_k$ encodes the state using the theoretically optimal number of bits.

**Now for decoding.** Given a final state $x_k$, we can recover the original sequence as follows:

1. The last symbol $s_k$ is simply $T[x_k]$ (the symbol at position $x_k$)
2. To get the previous state $x_{k-1}$:
   - Count how many times $s_k$ appeared before position $x_k$
   - This count is our previous state $x_{k-1}$
3. Continue this process to recover all symbols in reverse order

The formula for this decoding is

$$x_i = (x_{i+1} // L) \cdot f_{s_i} + (x_{i+1} \text{ mod } L) - C(s_i)$$

## Streaming ANS

### The Need for Streaming
In practice, we can't work with arbitrarily large integers. Our state $x$ would grow indefinitely as we encode more symbols, and operating on integers with arbitrarily large precision is very inefficient. We therefore need a way to keep the state within a manageable range while preserving the reversibility of the encoding.

### The Solution: State Normalization
We'll maintain our state $x$ within the interval $[L, 2L)$ by streaming out some bits to a separate array at each iteration. We need this process to be reversible.

Let B be an array of bits (bit stream), initialized to the empty array. We'll encode symbol $s$ with state $x$ as follows:

### Encoding $s, (x,B) \to (x',B')$
When encoding symbol $s$ with state $(x, B)$:
1. Find normalization factor $d$: the unique integer where $x // 2^d \in [f_s, 2f_s)$
2. Stream out the lower $d$ bits: $V = x \text{ mod } 2^d$ to bitstream $B$ to form $B'$
3. Use the normalized value to compute next state:
   $$x' = L + C_s + ((x // 2^d) \text{ mod } f_s)$$

### Decoding $(x',B') \to s, (x,B)$
To reverse the process:
1. Read the current symbol: $s = T[x']$
2. Compute the normalized state: $x_2 = f_s + (x' \text{ mod } L) - C[s]$
3. Find $d$: the unique integer where $x_2 \cdot 2^d \in [L, 2L)$
4. Read $d$ bits ($V$) from the end of bitstream $B'$ (and remove them to form $B$)
5. Reconstruct the original state: $x = (x_2 \cdot 2^d) + V$

With this process, the state stays bounded, and we accumulate a bit stream that represents the compressed version of the original sequence.

## Optimality of the Compression

In the above, there are two sources of loss in the compression efficiency:
1. The assumption that the frequencies sum to a power of two
2. The need to normalize the state

The first of these can be addressed by choosing $L=2^l$ large enough so that the proportions $p_i = f_i / L$ are accurate enough so that the loss is negligible.

The second source of loss is more difficult to predict but can be examined empirically. (do this analysis)