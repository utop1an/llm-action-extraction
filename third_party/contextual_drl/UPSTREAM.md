# Source provenance

Source: https://github.com/Shivam-Miglani/contextual_drl

Revision: `b6b2acb8e5cc8009adb2dacfb363e0157a08f922` (master inspected 2026-11-18).

License: Apache-2.0, included in LICENSE. These are source files from the authors,
not a new PyTorch implementation. Training uses EADQN.py, as upstream cmain.py
does; cEADQN.py is the separate GUI variant with hard-coded dimensions.

Local compatibility changes: EADQN.py omits its unused ipdb import (which creates
an IPython shell and writes history); utils.py imports matplotlib only inside plot_results,
with its noninteractive backend selected first. The original top-level pyplot
import hangs while creating a Windows IPython asyncio socket pair in this runtime.
Network architecture, reward logic, and upstream Agent/ReplayMemory are retained.

scripts/ceasdrl_local.py supplies an explicit dataset adapter and training harness.
It defaults to an independent cloned target network because upstream constructs
both Model objects from the same layer tensors. Use --target upstream-shared to
retain the original behavior for a separate diagnostic. Its compact replay,
document splits, span projection and context differences are recorded in receipts.

The full original environment constructor is not used: it hard-codes paths and
splits argument instances after expansion. The original act()/isTerminal() reward
and transition methods are used on explicitly initialized examples.
