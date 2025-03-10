# Some Brainfuck bytecode compiler

## Building

```
cargo +nightly build --release
clang vm.c -O3 -ffast-math -o vm
```

## Running

```
./target/release/SBFBCC ./<brainfuck file>.bf
./vm ./<brainfuck file>.lbf
```
