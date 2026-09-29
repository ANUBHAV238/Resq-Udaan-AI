from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .kml_processor import LocalFrame
from .messages import TargetStatus as TS

log = logging.getLogger("fusion")


@dataclass
class Target:
    target_id: str
    cls: str
    lat: float
    lon: float
    alt: float
    confidence: float
    priority_score: float
    priority_category: str
    first_detected_by: str
    last_seen_by: str
    observations: int
    first_seen: float
    last_seen: float
    required_boxes: int
    required_payload_weight_kg: float
    status: str = TS.NEW.value
    delivery_id: Optional[str] = None
    proposed_drone: Optional[str] = None
    proposal_expires: float = 0.0
    retry_after: float = 0.0
    observers: Dict[str, int] = field(default_factory=dict)
    excluded: Dict[str, float] = field(default_factory=dict)
    detection_ids: List[str] = field(default_factory=list)
    _w: float = 0.0
    _pw: float = 0.0

    def excluded_drones(self, now: float) -> List[str]:
        return [d for d, until in self.excluded.items() if until > now]

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id, "class": self.cls,
            "location": {"latitude": self.lat, "longitude": self.lon, "altitude": self.alt},
            "confidence": self.confidence,
            "priority": {"priority_score": self.priority_score, "priority_category": self.priority_category},
            "first_detected_by": self.first_detected_by, "last_seen_by": self.last_seen_by,
            "observations": self.observations, "timestamp": self.last_seen,
            "required_boxes": self.required_boxes,
            "required_payload_weight_kg": self.required_payload_weight_kg,
            "status": self.status, "delivery_id": self.delivery_id,
        }


class TargetFusion:
    def __init__(self, drone_id: str, frame: LocalFrame, cfg: dict, category_fn: Callable[[float], str]):
        self.me = drone_id
        self.frame = frame
        self.radius = cfg.get("merge_radius_m", 4.0)
        self.category_fn = category_fn
        self._t: Dict[str, Target] = {}
        self._alias: Dict[str, str] = {}
        self._seen = deque(maxlen=4000)
        self._seen_set = set()
        self._n = 0
        self._lock = threading.RLock()

    # ---- lookup ----
    def resolve(self, tid: Optional[str]) -> Optional[str]:
        seen = 0
        while tid in self._alias and seen < 50:
            tid = self._alias[tid]
            seen += 1
        return tid

    def get(self, tid: Optional[str]) -> Optional[Target]:
        with self._lock:
            return self._t.get(self.resolve(tid)) if tid else None

    def all(self) -> List[Target]:
        with self._lock:
            return list(self._t.values())

    def pending(self, now: float) -> List[Target]:
        with self._lock:
            return [t for t in self._t.values() if t.status == TS.NEW.value and t.retry_after <= now
                    and t.required_boxes > 0]

    def _xy(self, t: Target) -> Tuple[float, float]:
        return self.frame.to_local(t.lat, t.lon)

    def _nearest(self, xy, cls: str) -> Optional[Target]:
        best, bd = None, self.radius
        for t in self._t.values():
            if t.cls != cls:
                continue
            d = math.dist(xy, self._xy(t))
            if d <= bd:
                best, bd = t, d
        return best

    # ---- update ----
    def update(self, msg: dict) -> Tuple[Optional[Target], str]:
        det = msg["detection"]
        loc = det["location"]
        did = det["detection_id"]
        with self._lock:
            if did in self._seen_set:
                return None, "duplicate"
            if len(self._seen) == self._seen.maxlen:
                self._seen_set.discard(self._seen[0])
            self._seen.append(did)
            self._seen_set.add(did)

            who = msg.get("detecting_drone") or msg.get("drone_id") or "unknown"
            ts = float(msg.get("timestamp", time.time()))
            conf = float(det["confidence"])
            pr = msg.get("priority") or {}
            pscore = float(pr.get("priority_score", 0.0))
            xy = self.frame.to_local(loc["latitude"], loc["longitude"])

            t = self.get(msg.get("target_id"))
            if t is None or t.cls != det["class"]:
                t = self._nearest(xy, det["class"])
            event = "updated"
            if t is None:
                tid = msg.get("target_id")
                if not tid or tid in self._t:
                    self._n += 1
                    tid = f"target_{who}_{self._n:03d}"
                t = Target(target_id=tid, cls=det["class"], lat=loc["latitude"], lon=loc["longitude"],
                           alt=float(loc.get("altitude", 0.0)), confidence=conf, priority_score=pscore,
                           priority_category=pr.get("priority_category") or self.category_fn(pscore),
                           first_detected_by=who, last_seen_by=who, observations=0, first_seen=ts, last_seen=ts,
                           required_boxes=int(msg["required_boxes"]),
                           required_payload_weight_kg=float(msg["required_payload_weight_kg"]))
                self._t[tid] = t
                event = "new"
            w = max(conf, 0.05)
            if t.observations > 0:
                t.lat = (t.lat * t._w + loc["latitude"] * w) / (t._w + w)
                t.lon = (t.lon * t._w + loc["longitude"] * w) / (t._w + w)
                t.priority_score = (t.priority_score * t._pw + pscore * w) / (t._pw + w)
            t._w = min(t._w + w, 25.0)
            t._pw = min(t._pw + w, 25.0)
            t.priority_category = self.category_fn(t.priority_score)
            t.confidence = max(t.confidence, conf)
            t.observations += 1
            t.observers[who] = t.observers.get(who, 0) + 1
            t.last_seen_by = who
            t.last_seen = max(t.last_seen, ts)
            t.required_boxes = max(t.required_boxes, int(msg["required_boxes"]))
            t.required_payload_weight_kg = max(t.required_payload_weight_kg,
                                               float(msg["required_payload_weight_kg"]))
            t.detection_ids = (t.detection_ids + [did])[-20:]
            t = self._merge_close(t)
            return t, event

    def _merge_close(self, t: Target) -> Target:
        for other in list(self._t.values()):
            if other is t or other.cls != t.cls:
                continue
            if math.dist(self._xy(t), self._xy(other)) > self.radius:
                continue
            older, younger = sorted((t, other), key=lambda x: (x.first_seen, x.target_id))
            wa, wb = max(older._w, 0.05), max(younger._w, 0.05)
            older.lat = (older.lat * wa + younger.lat * wb) / (wa + wb)
            older.lon = (older.lon * wa + younger.lon * wb) / (wa + wb)
            older.priority_score = (older.priority_score * older._pw + younger.priority_score * younger._pw) \
                / max(1e-9, older._pw + younger._pw)
            older.priority_category = self.category_fn(older.priority_score)
            older._w = min(wa + wb, 25.0)
            older._pw = min(older._pw + younger._pw, 25.0)
            older.confidence = max(older.confidence, younger.confidence)
            older.observations += younger.observations
            for d, n in younger.observers.items():
                older.observers[d] = older.observers.get(d, 0) + n
            older.last_seen = max(older.last_seen, younger.last_seen)
            older.last_seen_by = younger.last_seen_by if younger.last_seen >= older.last_seen else older.last_seen_by
            older.required_boxes = max(older.required_boxes, younger.required_boxes)
            older.required_payload_weight_kg = max(older.required_payload_weight_kg,
                                                   younger.required_payload_weight_kg)
            older.detection_ids = (older.detection_ids + younger.detection_ids)[-20:]
            if older.status == TS.NEW.value and younger.status != TS.NEW.value:
                older.status, older.delivery_id = younger.status, younger.delivery_id
                older.proposed_drone, older.proposal_expires = younger.proposed_drone, younger.proposal_expires
            self._alias[younger.target_id] = older.target_id
            self._t.pop(younger.target_id, None)
            log.info("merged %s into %s", younger.target_id, older.target_id)
            t = older
        return t

    # ---- status handling ----
    def set_status(self, tid: str, status: TS, delivery_id: Optional[str] = None,
                   proposed_drone: Optional[str] = None, expires: float = 0.0,
                   retry_after: Optional[float] = None, exclude_drone: Optional[str] = None,
                   exclude_until: float = 0.0):
        with self._lock:
            t = self.get(tid)
            if t is None:
                return
            t.status = status.value
            t.delivery_id = delivery_id
            t.proposed_drone = proposed_drone
            t.proposal_expires = expires
            if retry_after is not None:
                t.retry_after = retry_after
            if exclude_drone:
                t.excluded[exclude_drone] = exclude_until

    def expire(self, now: float):
        with self._lock:
            for t in self._t.values():
                if t.status == TS.PROPOSED.value and t.proposal_expires and now > t.proposal_expires:
                    t.status, t.delivery_id, t.proposed_drone = TS.NEW.value, None, None

    def proposed_drones(self, now: float) -> List[str]:
        with self._lock:
            return [t.proposed_drone for t in self._t.values()
                    if t.status == TS.PROPOSED.value and t.proposed_drone and t.proposal_expires > now]

    def snapshot(self) -> List[dict]:
        return [t.to_dict() for t in self.all()]