"""Check the four PyTorch RNG-stream premises of Proposition 1 (Appendix B.2).

Runs 13 boolean assertions on the installed torch build, on both the CPU and
the CUDA device, and exits non-zero if any fails:

  P1 (3) rand / randint / randperm without generator= advance the global CPU state
  P2 (3) fork_rng restores CPU and CUDA state; the next draw matches a no-fork run
  P3 (2) draws through an explicit torch.Generator leave both global states unchanged
  P4 (5) CPU and CUDA default generators are distinct objects with separate streams
"""
import sys

import torch

if not torch.cuda.is_available():
    sys.exit("rng_premise_check: a CUDA device is required (P2-P4 test the CUDA stream).")

results = []


def check(name, ok):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


def cpu_state():
    return torch.get_rng_state().clone()


def cuda_state():
    return torch.cuda.get_rng_state(0).clone()


print("torch", torch.__version__)
torch.manual_seed(1234)

# P1: factory calls without generator= advance the global CPU state.
s0 = cpu_state()
torch.rand(4)
s1 = cpu_state()
torch.randint(0, 10, (4,))
s2 = cpu_state()
torch.randperm(8)
s3 = cpu_state()
check("P1 rand advances global CPU state", not torch.equal(s0, s1))
check("P1 randint advances global CPU state", not torch.equal(s1, s2))
check("P1 randperm advances global CPU state", not torch.equal(s2, s3))

# P2: fork_rng restores CPU and listed-device state.
torch.manual_seed(777)
ref_next = torch.rand(3)
torch.manual_seed(777)
s5, c5 = cpu_state(), cuda_state()
with torch.random.fork_rng(devices=[0]):
    torch.rand(500)
    torch.rand(500, device="cuda")
check("P2 fork_rng restores CPU state", torch.equal(s5, cpu_state()))
check("P2 fork_rng restores CUDA state", torch.equal(c5, cuda_state()))
check("P2 draw after fork_rng equals no-fork draw", torch.equal(torch.rand(3), ref_next))

# P3: an explicit generator leaves both global streams untouched.
g = torch.Generator()
g.manual_seed(0)
s4, c4 = cpu_state(), cuda_state()
torch.rand(1000, generator=g)
torch.randint(0, 10, (1000,), generator=g)
torch.randperm(1000, generator=g)
check("P3 explicit-generator draws leave global CPU state unchanged", torch.equal(s4, cpu_state()))
check("P3 explicit-generator draws leave global CUDA state unchanged", torch.equal(c4, cuda_state()))

# P4: CPU and CUDA default generators are separate streams.
check("P4 CPU and CUDA default generators are distinct objects",
      torch.default_generator is not torch.cuda.default_generators[0])
s6, c6 = cpu_state(), cuda_state()
torch.rand(64, device="cuda")
check("P4 CUDA draw advances CUDA state", not torch.equal(c6, cuda_state()))
check("P4 CUDA draw leaves CPU state unchanged", torch.equal(s6, cpu_state()))
s7, c7 = cpu_state(), cuda_state()
torch.rand(64)
check("P4 CPU draw advances CPU state", not torch.equal(s7, cpu_state()))
check("P4 CPU draw leaves CUDA state unchanged", torch.equal(c7, cuda_state()))

passed = sum(ok for _, ok in results)
print(f"{passed}/{len(results)} assertions pass")
sys.exit(0 if passed == len(results) == 13 else 1)
