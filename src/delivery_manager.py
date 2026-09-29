from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .messages import DeliveryState as S

ACTIVE = {S.DELIVERY_ASSIGNED, S.DELIVERY_RESERVED, S.DELIVERY_ON_WAY,
          S.ARRIVED_AT_TARGET, S.DELIVERY_IN_PROGRESS}
TERMINAL = {S.DELIVERED, S.DELIVERY_FAILED, S.DELIVERY_CANCELLED, S.RETURNING}

TRANSITIONS: Dict[S, set] = {
    S.DELIVERY_ASSIGNED: {S.DELIVERY_RESERVED, S.DELIVERY_CANCELLED, S.DELIVERY_FAILED},
    S.DELIVERY_RESERVED: {S.DELIVERY_ON_WAY, S.DELIVERY_CANCELLED, S.DELIVERY_FAILED},
    S.DELIVERY_ON_WAY: {S.ARRIVED_AT_TARGET, S.DELIVERY_CANCELLED, S.DELIVERY_FAILED},
    S.ARRIVED_AT_TARGET: {S.DELIVERY_IN_PROGRESS, S.DELIVERY_CANCELLED, S.DELIVERY_FAILED},
    S.DELIVERY_IN_PROGRESS: {S.DELIVERED, S.DELIVERY_FAILED},
    S.DELIVERED: {S.RETURNING},
    S.DELIVERY_FAILED: {S.RETURNING},
    S.DELIVERY_CANCELLED: {S.RETURNING},
    S.RETURNING: set(),
}


class DeliveryError(Exception):
    pass


@dataclass
class Delivery:
    delivery_id: str
    drone_id: str
    target_id: str
    destination: dict
    boxes: int
    payload_weight_kg: float
    status: S = S.DELIVERY_ASSIGNED
    created_at: float = field(default_factory=time.time)
    status_ts: float = field(default_factory=time.time)
    reason: Optional[str] = None
    priority: Optional[dict] = None
    history: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "message_type": "delivery",
            "delivery_id": self.delivery_id,
            "drone_id": self.drone_id,
            "target_id": self.target_id,
            "destination": self.destination,
            "boxes": self.boxes,
            "payload_weight_kg": self.payload_weight_kg,
            "status": self.status.value,
            "reason": self.reason,
            "priority": self.priority,
            "timestamp": self.status_ts,
            "created_at": self.created_at,
        }


class DeliveryManager:
    def __init__(self, drone_id: str, terminal_hold_s: float = 5.0,
                 on_change: Optional[Callable[[Delivery], None]] = None):
        self.drone_id = drone_id
        self.hold_s = terminal_hold_s
        self.on_change = on_change
        self._lock = threading.RLock()
        self._records: Dict[str, Delivery] = {}
        self._current: Optional[Delivery] = None

    def create(self, delivery_id: str, target_id: str, destination: dict, boxes: int,
               payload_weight_kg: float, priority: Optional[dict] = None) -> Delivery:
        with self._lock:
            if delivery_id in self._records:
                raise DeliveryError(f"duplicate delivery id {delivery_id}")
            if self.active() is not None:
                raise DeliveryError("a delivery is already active on this drone")
            d = Delivery(delivery_id, self.drone_id, target_id, destination, boxes, payload_weight_kg,
                         priority=priority)
            d.history.append({"status": d.status.value, "ts": d.status_ts, "reason": "created"})
            self._records[delivery_id] = d
            self._current = d
        self._notify(d)
        return d

    def transition(self, delivery_id: str, new: S, reason: Optional[str] = None) -> Delivery:
        with self._lock:
            d = self._records.get(delivery_id)
            if d is None:
                raise DeliveryError(f"unknown delivery {delivery_id}")
            if new not in TRANSITIONS.get(d.status, set()):
                raise DeliveryError(f"illegal transition {d.status.value} -> {new.value}")
            d.status = new
            d.status_ts = time.time()
            d.reason = reason
            d.history.append({"status": new.value, "ts": d.status_ts, "reason": reason})
        self._notify(d)
        return d

    def try_transition(self, delivery_id: str, new: S, reason: Optional[str] = None) -> bool:
        try:
            self.transition(delivery_id, new, reason)
            return True
        except DeliveryError:
            return False

    def mark_returning(self):
        with self._lock:
            d = self._current
        if d is not None and d.status in {S.DELIVERED, S.DELIVERY_FAILED, S.DELIVERY_CANCELLED}:
            self.try_transition(d.delivery_id, S.RETURNING, "returning to home")

    def get(self, delivery_id: Optional[str]) -> Optional[Delivery]:
        with self._lock:
            return self._records.get(delivery_id) if delivery_id else None

    def active(self) -> Optional[Delivery]:
        with self._lock:
            d = self._current
            return d if d is not None and d.status in ACTIVE else None

    def view(self) -> dict:
        with self._lock:
            d = self._current
            if d is None or (d.status not in ACTIVE and time.time() - d.status_ts > self.hold_s):
                return {"status": S.NO_DELIVERY.value, "delivery_id": None, "target_id": None,
                        "boxes": 0, "payload_weight_kg": 0}
            return {"status": d.status.value, "delivery_id": d.delivery_id, "target_id": d.target_id,
                    "boxes": d.boxes, "payload_weight_kg": d.payload_weight_kg}

    def _notify(self, d: Delivery):
        if self.on_change:
            self.on_change(d)