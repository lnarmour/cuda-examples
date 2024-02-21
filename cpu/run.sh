#!/bin/bash

echo compiling...
make -B

echo
echo running matmult...
for r in {1..4}; do ./matmult; done;

echo
echo running matmult.simd...
for r in {1..4}; do ./matmult.simd; done;
