# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Differentiable direct reinforcement learning extension for Isaac Lab.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
# Guard against headless environments where omni is not available.
try:
    from .ui_extension_example import *  # noqa: F401, F403
except ModuleNotFoundError:
    # UI extensions require the full Omniverse environment
    # (omni.ext, omni.ui, etc.) and are safely skipped in
    # headless / task-registration-only usage.
    pass
