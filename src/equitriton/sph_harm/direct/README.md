# Direct spherical harmonics

This module implements spherical harmonics of up to $l=10$ _directly_ in terms
of $x,y,z$. Each submodule implements a particular order, comprising four objects:
a PyTorch `autograd.Function` wrapper, forward and backward Triton kernels,
and a PyTorch implementation of the forward kernel. The PyTorch implementation
is not necessarily intended for performance, rather for double checking that
the Triton versions are behaving as intended.

Currently, the kernels are heavily computer assisted, and may not be optimal
particularly on the register front: there are a lot of redudant constants,
and we are relying heavily on the LLVM compiler to realize this and group
them at run time. Similarly, the variable names are also not very human-friendly;
this is unlikely to change; they might have a high maintenance burden,
but we're unlikely to touch them very much.
