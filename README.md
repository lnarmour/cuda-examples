# How CUDA?

This repository contains several examples of CUDA C/C++ code.

## Prereqs

You need to set up your environment so it can find the compiler, debugger, and toolchain.
You only need to do this once.
On any department machine, run the following to add the cuda module:
```
echo 'module load cuda' >> ~/.bashrc
source ~/.bashrc
```

Confirm that you can find the `nvcc` compiler with the following command:
```
$ which nvcc
/usr/local/cuda/12.0/bin/nvcc
```
And confirm that you can run it by checking the version:
```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Mon_Oct_24_19:12:58_PDT_2022
Cuda compilation tools, release 12.0, V12.0.76
Build cuda_12.0.r12.0/compiler.31968024_0
```
(note the "$" is part the shell prompt and not the commands)

## GPU machines

The examples here must be run on machines with GPUs:

* fish machines have [NVIDIA GeForce GTX 1060 6GB](https://www.techpowerup.com/gpu-specs/geforce-gtx-1060-6-gb.c2862) cards (anchovy, barracuda, blowfish, bonito, brill, bullhead, char, cod, dorado, eel, flounder, grouper, halibut, herring, mackerel, marlin, perch, pollock, sardine, shark, sole, swordfish, tarpon, turbot, tuna, wahoo, earth)
* planet machines have [NVIDIA TITAN V](https://www.techpowerup.com/gpu-specs/titan-v.c3051) cards (jupiter, mars, mercury, neptune, saturn, uranus, venus)

## Examples

The examples directory contains several small examples of increasing complexity.
Each example has a Makefile that can be used to build it and a README showing its output.
