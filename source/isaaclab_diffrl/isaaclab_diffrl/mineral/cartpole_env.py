# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backward-compatible Mineral cartpole adapter exports."""

from __future__ import annotations

from ..integrations.mineral.direct_env_adapter import MineralDirectEnvAdapter


class MineralCartpoleEnvAdapter(MineralDirectEnvAdapter):
    """Compatibility alias for the generic Mineral direct-environment adapter."""


__all__ = ["MineralCartpoleEnvAdapter"]
