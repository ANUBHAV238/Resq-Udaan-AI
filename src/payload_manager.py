from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List

log = logging.getLogger("payload")


class PayloadManager:
    """Tracks physical box slots so two deliveries can never get the same box."""

    def __init__(self, box_count: int, box_weight_kg: float):
        if box_count <= 0 or box_weight_kg <= 0:
            raise ValueError("box_count and box_weight_kg must be > 0")
        self.box_count = box_count
        self.box_weight_kg = float(box_weight_kg)
        self._lock = threading.RLock()
        self._slot: Dict[int, str] = {i: "available" for i in range(box_count)}
        self._res: Dict[str, List[int]] = {}

    def reserve(self, delivery_id: str, n: int) -> bool:
        with self._lock:
            if n <= 0 or delivery_id in self._res:
                return False
            free = sorted(s for s, st in self._slot.items() if st == "available")
            if len(free) < n:
                return False
            picked = free[:n]
            for s in picked:
                self._slot[s] = "reserved"
            self._res[delivery_id] = picked
            return True

    def slots_for(self, delivery_id: str) -> List[int]:
        with self._lock:
            return list(self._res.get(delivery_id, []))

    def commit(self, delivery_id: str) -> bool:
        with self._lock:
            slots = self._res.pop(delivery_id, None)
            if slots is None:
                return False
            for s in slots:
                self._slot[s] = "delivered"
            return True

    def release(self, delivery_id: str) -> bool:
        with self._lock:
            slots = self._res.pop(delivery_id, None)
            if slots is None:
                return False
            for s in slots:
                self._slot[s] = "available"
            return True

    def status(self) -> dict:
        with self._lock:
            av = sum(1 for v in self._slot.values() if v == "available")
            rs = sum(1 for v in self._slot.values() if v == "reserved")
            dl = sum(1 for v in self._slot.values() if v == "delivered")
        return {
            "total_boxes": self.box_count,
            "available_boxes": av,
            "reserved_boxes": rs,
            "delivered_boxes": dl,
            "box_weight_kg": self.box_weight_kg,
            "total_payload_kg": round(self.box_count * self.box_weight_kg, 3),
            "remaining_payload_kg": round((av + rs) * self.box_weight_kg, 3),
        }


class ServoReleaseMechanism:
    """One servo per box slot, driven with MAV_CMD_DO_SET_SERVO through MAVROS."""

    MAV_CMD_DO_SET_SERVO = 183

    def __init__(self, cfg: dict, box_count: int, command_long: Callable[..., bool]):
        self.servos = cfg.get("servos", [])
        if len(self.servos) < box_count:
            raise ValueError(f"payload_release.servos needs {box_count} entries, got {len(self.servos)}")
        self.hold_s = cfg.get("hold_s", 1.5)
        self.between_s = cfg.get("between_s", 0.5)
        self._cmd = command_long

    def release(self, slots: List[int]) -> bool:
        if not slots:
            return False
        for slot in slots:
            sv = self.servos[slot]
            if not self._cmd(self.MAV_CMD_DO_SET_SERVO, sv["channel"], sv["open_pwm"]):
                log.error("servo open failed for slot %s", slot)
                self._cmd(self.MAV_CMD_DO_SET_SERVO, sv["channel"], sv["closed_pwm"])
                return False
            time.sleep(self.hold_s)
            if not self._cmd(self.MAV_CMD_DO_SET_SERVO, sv["channel"], sv["closed_pwm"]):
                log.error("servo close failed for slot %s", slot)
                return False
            time.sleep(self.between_s)
        return True