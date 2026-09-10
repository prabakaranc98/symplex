"""Operator-owned bounded execution profiles; never mutable by agent proposals."""

import json
from pathlib import Path

from symplex.core.contracts import digest

POLICY = json.loads(Path(__file__).with_name("profiles.json").read_text())
DEPTHS = POLICY["profiles"]
POLICY_DIGEST = digest(POLICY)
