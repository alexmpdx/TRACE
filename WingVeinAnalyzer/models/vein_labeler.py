"""Topology-based vein identity assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import networkx as nx
from skan import Skeleton


class VeinStatus(Enum):
    COMPLETE = "complete"
    FRAGMENTED = "fragmented"
    TRUNCATED = "truncated"
    ABSENT = "absent"


@dataclass
class VeinAssignment:
    vein_id: str
    status: VeinStatus
    edge_ids: list[int]
    confidence: float
    evidence: list[str] = field(default_factory=list)
    length_px: float = 0.0
    gap_px: float | None = None
    length_um: float | None = None


def assign_veins(
    skan_skeleton: Skeleton,
    graph: nx.Graph,
    margin_tolerance_px: float = 5.0,
    max_gap_px: float = 20.0,
) -> list[VeinAssignment]:
    """Assign vein identities to skeleton edges using topology cues."""
    assignments: list[VeinAssignment] = []
    return assignments
