from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Sequence

from .kml_processor import LocalFrame
from .lawnmower import Track

ACTIVE_DELIVERY = {"DELIVERY_ASSIGNED", "DELIVERY_RESERVED", "DELIVERY_ON_WAY",
                   "ARRIVED_AT_TARGET", "DELIVERY_IN_PROGRESS"}
DISPATCHABLE_MISSION = {"SEARCHING", "TARGET_DETECTED"}


class SectorAllocator:
    """Splits coverage tracks into n contiguous strips with balanced path length."""

    def __init__(self, n_sectors: int):
        self.n = n_sectors

    def split(self, tracks: Sequence[Track]) -> List[List[Track]]:
        n = self.n
        if len(tracks) >= n:
            weights = [t.length() + 1.0 for t in tracks]
            total = sum(weights)
            sectors: List[List[Track]] = [[] for _ in range(n)]
            cum = 0.0
            for t, w in zip(tracks, weights):
                idx = min(n - 1, int((cum + w / 2.0) / (total / n)))
                sectors[idx].append(t)
                cum += w
            if all(sectors):
                return sectors
        # fallback: split the concatenated point stream into n contiguous chunks
        pts = [p for t in tracks for p in t.points]
        size = max(1, math.ceil(len(pts) / n))
        out: List[List[Track]] = []
        for i in range(n):
            chunk = pts[i * size:(i + 1) * size]
            out.append([Track(index=i, line=i, points=chunk)] if chunk else [])
        return out

    @staticmethod
    def assign(sectors: List[List[Track]], drone_ids: Sequence[str]) -> Dict[str, List[Track]]:
        return {d: sectors[i] if i < len(sectors) else [] for i, d in enumerate(drone_ids)}


class TaskAllocator:
    """Ranks drones for a target. Hard constraints first, then a weighted score."""

    def __init__(self, cfg: dict, delivery_cfg: dict, frame: LocalFrame):
        self.frame = frame
        a = cfg.get("allocation", {})
        self.peer_timeout = a.get("peer_timeout_s", 6.0)
        self.max_range = a.get("max_range_m", 3000.0)
        self.w = a.get("weights", {"distance": 0.45, "battery": 0.2, "boxes": 0.1, "workload": 0.25})
        self.boost = a.get("priority_distance_boost", 1.0)
        self.min_batt = delivery_cfg.get("min_battery_pct", 30.0)
        self.reserve = delivery_cfg.get("battery_reserve_pct", 15.0)
        self.drain = delivery_cfg.get("drain_pct_per_km", 6.0)
        self.box_count = cfg.get("payload", {}).get("box_count", 4)

    @staticmethod
    def order_targets(targets: Iterable) -> list:
        return sorted(targets, key=lambda t: (-t.priority_score, t.first_seen, t.target_id))

    def rank(self, target, required_boxes: int, fleet: Dict[str, dict],
             exclude: Iterable[str] = (), busy: Iterable[str] = ()) -> List[dict]:
        now = time.time()
        exclude, busy = set(exclude), set(busy)
        out = []
        for did in sorted(fleet):
            st = fleet[did]
            reasons: List[str] = []
            pos = st.get("position") or {}
            pay = st.get("payload") or {}
            bat = (st.get("battery") or {}).get("percentage")
            dl = st.get("delivery") or {}
            ms = st.get("mission") or {}
            conn = st.get("connection") or {}
            fl = st.get("flight") or {}

            if now - st.get("_rx", 0) > self.peer_timeout:
                reasons.append("stale status")
            if not conn.get("mavros") or not conn.get("mqtt"):
                reasons.append("link down")
            if did in exclude:
                reasons.append("excluded for this target")
            if did in busy:
                reasons.append("has pending proposal")
            if dl.get("status") in ACTIVE_DELIVERY:
                reasons.append("delivery in progress")
            if ms.get("status") not in DISPATCHABLE_MISSION:
                reasons.append(f"mission state {ms.get('status')}")
            if not fl.get("armed"):
                reasons.append("not armed")
            if int(pay.get("available_boxes", 0)) < required_boxes:
                reasons.append(f"available boxes {pay.get('available_boxes', 0)} < {required_boxes}")
            lat, lon = pos.get("latitude"), pos.get("longitude")
            dist = None
            if lat is None or lon is None:
                reasons.append("no position")
            else:
                dist = self.frame.distance_m(lat, lon, target.lat, target.lon)
            if bat is None:
                reasons.append("battery unknown")
            elif dist is not None:
                needed = self.min_batt + self.reserve * 0 + dist / 1000.0 * self.drain * 2 + self.reserve
                if bat < needed:
                    reasons.append(f"battery {bat:.0f}% < needed {needed:.0f}%")

            score = -1.0
            if not reasons:
                d_norm = 1.0 - min(dist / self.max_range, 1.0)
                b_norm = max(0.0, min(1.0, (bat - self.min_batt) / max(1.0, 100.0 - self.min_batt)))
                spare = (int(pay.get("available_boxes", 0)) - required_boxes) / max(1, self.box_count)
                total_wp = max(1, int(ms.get("total_waypoints") or 1))
                remaining = max(0.0, 1.0 - float(ms.get("current_waypoint") or 0) / total_wp)
                reserved_pen = 0.1 * int(pay.get("reserved_boxes", 0)) / max(1, self.box_count)
                score = (self.w["distance"] * (1.0 + self.boost * target.priority_score) * d_norm
                         + self.w["battery"] * b_norm
                         + self.w["boxes"] * spare
                         + self.w["workload"] * (1.0 - remaining)
                         - reserved_pen)
            out.append({"drone_id": did, "eligible": not reasons, "score": round(score, 4),
                        "distance_m": None if dist is None else round(dist, 1), "reasons": reasons})
        out.sort(key=lambda r: (not r["eligible"], -r["score"], r["drone_id"]))
        return out

    @staticmethod
    def best(ranking: List[dict]) -> Optional[dict]:
        return ranking[0] if ranking and ranking[0]["eligible"] else None