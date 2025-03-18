# Some Brainfuck bytecode compiler

This is some optimizing Brainfuck bytecode compiler that represents the operations as repeated
affine linear Matrices with some optimizations (see [below](#Optimizations)).








## Building

```
cargo +nightly build --release
gcc vm.c -O3 -o vm
```

## Running

```
./target/release/SBFBCC ./<brainfuck file>.bf
./vm ./<brainfuck file>.lbf
```
# Optimizations

## Optimizations during Parsing (and AST simplification)

### Affine linear optimizations

When we look at the most common constructs like `++++` → `t[0] += 4` or `>++` → `t[1] += 2`, we can see that they're often affine linear.
So we can think of the Brainfuck memory as an arbitrarily dimensional vector and some parts of the code as affine linear transformations of it.

- chains of n `+` and `-`: `t[i] += n`
- `<` and `>`: `t[i±1]` (different entry)
- `[-]` or `[+]` or `[+++]`: `t[i] = 0`
- `[->++<]`: `t[i+1] = 2 * t[i]; t[i] = 0;`

  or `[->+++<<->]` etc.

By representing all these operations as (arbitrarily sized, homogenous) matrices
$\tilde{A} x \equiv A x + b$, we can combine any combination of these constructs to 
a single matrix containing all the information. If we can find an optimal way to execute the static matrix,
it's equivalent to an optimal way to execute the combination of brainfuck constructs.

This is why they're represented as `OffsetMap o` in code.

As matrix multiplication doesn't happen in-place by default, I chose to PLU-decompose them first as the triangular matrices $L, U$ as well as the permutation $P$
can be executed in place. This leads to $n^2$ Multiplications in the worst case,
as well as $n$ swaps, which is the same as a normal matrix vector multiplication ($n^2$ multiplications)
and $n$ swaps to copy the resulting vector back into memory. However, while decomposing, we can choose the optimal 
pivot to maximize the number of zero entries in the $L$ and $U$ matrix, to lower the number of operations.

Here I just chose a heuristic, by bruteforce-esque comparing each row with each other row, to see,
whether they're a linear factor of each other (and if not, by how many entries they differ apart from a factor).
The row with the most entries, that are just a linear factor of that row apart, is chosen, if the number is greater than 3,
as having another permutation for pivoting would also decrease performance.
This is not optimal, but is a sensible heuristic.

The LU Decomposition is also happening in the Ring $\mathbb{Z}_{256}$ (as Brainfuck runs on a wrapping byte),
which is NOT the Galois Field $GF(256)$. This means, not every number has an inverse (only odd numbers have an inverse),
restricting LU decomposition. However division between $a, b in \mathbb{Z}_{256}$ is still defined, as long as 
$a = c 2^m, b = d 2^n, c, d \text{ odd} => m \geq n $, as $\frac{a}{b} = \frac{c 2^m}{d 2^n} = \frac{c 2^(m - n)}{d}$, and 
as $d$ is odd, it has an inverse, making the division well defined.

This can be easily computed, by taking the row, whose entry has the least amount of trailing zeros in bit representation: `w8.0.trailing_zeros()` in Rust.

This results in the following AST:

```rs
pub enum Optree<T: BFAffineT<isize>> {
    OffsetMap(T, DebugPosition), // containing the linear transform
    Branch(Vec<Optree<T>>, isize, isize, DebugPosition), // is a [...] with the numbers being the preshift i.e. number of shifts before [
                                                         // and itershift, i.e. the number of shifts before ]
                                                         // e.g.  >+>->[ ...>[>] >.>,> ] would have 2 preshift and 3 itershift (not 4, as one is captured by the inner loop)
    Input(isize, DebugPosition), // simple ,
    Output(isize, DebugPosition), // simple .
    DebugBreakpoint(DebugPosition), // added instruction # as its quite common in other bf compilers and useful
}
```

## Optimizations at AST simplification

### Input overwrite

`++++++,` would overwrite the first `+` chain, rendering them useless. We can
easily apply this optimization, by checking whether `OffsetMap o` is followed by `Input(index)`
and removing the `index` row from `o` (as the rows in a matrix determine the result of that entry).


### Simple Constant propagation 

We can collect constant terms (i.e. OffsetMaps with zero rows (not empty, as empty would mean identity) and some affine value) and propagate them through the program.

If a branch `Branch(inner, preshift, postshift)` is followed by

- an `OffsetMap o` with a row at index `preshift` containing nothing but a `0` at the identity line (making the result 0)
- another branch and `preshift = 0`
- any of the above and a bunch of output statements between them

it can be discarded, as the current value always 0.

### Affine only lines

If a `OffsetMap o` contains a zero row in the linear part (so only affine, i.e. `t[i] = x`) and gets repeated,
it doesn't make sense to repeatedly set the same value again:

```rust

while t[0] {
    t[a] = ... + a * t[b];
    t[b] = q;
} 
```

We don't have to set `t[b]` at every iteration, if it's gonna be q at every iteration anyways

first we unroll the first iteration:


```rust
if t[0] {
    t[a] = ... + a * t[b];
    t[b] = q;
    while(t[0]) {
        t[a] = ... + a * t[b];
        t[b] = q;
    }
}
```

to

```rust 
if t[0] {
    t[a] = ... + a * t[b];
    t[b] = q;
    while(t[0]) {
        t[a] = ... + a * q; // part of the constant, a * q is compile time computed
    }
}
```

we can turn it back into while loops to continue working with `Branch`:

```rust
while t[0] {
    t[a] = ... + a * t[b];
    t[b] = q;
    while t[0] {
        t[a] = ... + a * q;
    }
}
```

This may seem to make it more inefficient, as we now have another while loop, but
we handle it with JNEZ bundling.

This works, as long es the entire branch 

- doesn't shift (`itershift = 0` or for all other branches inside: `itershift_inner = 0, sum of preshifts + itershift_outer = 0`)
- nothing else sets `t[b]` (i.e. no `Input`, no `OffsetMap`)

`Branch[OffsetMap o, ...] -> Branch[OffsetMap o, Branch[ OffsetMap o.remove_affine_only(), ... ]]`

### Square detection / Repeated linear operations / 2nd degree Polynomials (TODO)


`o.is_affine_at0() && o.affine[0] % 2 == 1 && ...`

This is a repeated matrix multiplication, which can indicate different things

e.g. [the pattern](https://esolangs.org/wiki/Brainfuck_algorithms#x%C2%B4_=_x_*_x) `x[temp0+x-]temp0[-[temp1+x++temp0-]x+temp1[temp0+temp1-]temp0]` for $x := x^2$

becomes:

```rust
temp0 = x; x = 0;
while temp0 > 0 {
    temp0 -= 1;
    x += 2 * temp0;
    x += 1;
   temp1 = 0;
}
```

⇒ linear then affine representation ⇒

```
while temp0 > 0 {
    x += 2 * temp0;
    x -= 1;
    temp0 -= 1;
    temp1 = 0;
}
```

Which is just a repeated matrix multiplication and represents a square.

We can try to generalize this to any case, where the counter variable $k$ (at offset "0") is only
transformed affine by $r$ and every other variable either depends on that counter (with factor $a$) or a constant $c$
(and not each other).

The value of the counter at each iteration $i$ is then: $k + i * r$

and the end value of the other variable is: (with n being the number of iterations)


$$\sum_{i = 0}^{n - 1} (k + i * r) * a + c$$

$$n \cdot c + a \sum_{i = 0}^{n - 1} (k + i \cdot r)$$

$$n \cdot c + a k n + a r  \sum_{i = 0}^{n - 1} i$$

$$n \cdot c + a k n + a r  n(n - 1)/2$$

for the x = x^2 example, we have: $n = x, c = -1, k = n, a = 2, r = -1$

$$= -n + 2 n^2 - n(n - 1)$$
$$= -n + 2 n^2 - n(n - 1)$$
$$= n (-1 + 2 n - n + 1)$$
$$= n n = x^2$$

we can also solve for $n$:

$k + n \cdot r = 0 \Rightarrow n = - k / r$ (therefore r has to be odd, otherwise, n might be infinite)

$n \cdot c + a k n + a r  n(n - 1)/2$ becomes 

$$- k /r \cdot c + a k (- k /r) + a r (- k / r) (- k/r - 1) / 2$$
$$= - c k / r - a k^2 / r + a k (k/r + 1) / 2$$
$$= - c k / r - a k  (k/r) 2 / 2 + a k (k/r + 1) / 2$$
$$= - c k / r + a k (-k/r + 1) / 2$$
$$= a k (-k/r + 1) / 2 - c k / r$$


we can set $s = -1/r$ at compile time and $l = s k$  at runtime
$= a k (s k + 1) / 2 + c s k$

(k is even -> $k (s k + 1)$ is even -> divisible by 2)
(k is odd -> $s k$ is odd -> $s k + 1$ is even -> $k (s k + 1)$ is even -> divisible by 2)

How do we compute it? (as much in compile time as we want, but as little runtime as possible!)


```
if a is even: b = a/2 is computed at compiletime:

    a k (l + 1) / 2 + c l
    = b k (l + 1)  + c l

    compiletime:
    b = a / 2 
    s = -1/r

    runtime:
    l = s * k
    result += c * l
    l += 1
    l *= k 
    result += b * l     => we only need one extra register (which we alr. do for
                           permutations) and 5 lin comb instructions

    if s is 1 (which we can detect at compiletime):

        b k (k + 1)  + c k = b k^2 + (b + c) k

        l = k * k
        result += b * l
        result += (b + c) k      // b + c is computed at compiletime
                                 // reduced to 3 lin comb inst

if a is odd, we can't incorporate /2 into a, but have to watch it at runtime:
    if k % 2 == 0 { (k/2) * ... } else {(s k + 1)/2 * ...}

$= a k (s k + 1) / 2 + c s k$

    s = -1/r

    runtime:
    l = s * k
    result += c * l
    l += 1
    if (k % 2 == 0) { // jump if even
        l = l * (k / 2);
    } else {          // jump
        l = (l / 2) * k;
    }
    result += a * l; // 6 lin comb inst, 2 jump inst
```


## Optimizations at Code Gen

### Sensible defaults

### Bundling JNEZ

### JNEZ, JEZ, J redirection

## Optimizations of the VM

### Threaded dispatch

### JIT Compiling (TODO)
