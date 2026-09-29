from __future__ import annotations

import json
import math
import time
from enum import Enum
from typing import Any, Optional


class MissionState(str, Enum):
    IDLE = "IDLE"
    READY = "READY"
    SEARCHING = "SEARCHING"
    TARGET_DETECTED = "TARGET_DETECTED"
    TASK_ASSIGNED = "TASK_ASSIGNED"
    DELIVERY_RESERVED = "DELIVERY_RESERVED"
    DELIVERY_ON_WAY = "DELIVERY_ON_WAY"
    ARRIVED_AT_TARGET = "ARRIVED_AT_TARGET"
    DELIVERY_IN_PROGRESS = "DELIVERY_IN_PROGRESS"
    DELIVERED = "DELIVERED"
    RETURNING = "RETURNING"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    ERROR = "ERROR"


class DeliveryState(str, Enum):
    NO_DELIVERY = "NO_DELIVERY"
    DELIVERY_ASSIGNED = "DELIVERY_ASSIGNED"
    DELIVERY_RESERVED = "DELIVERY_RESERVED"
    DELIVERY_ON_WAY = "DELIVERY_ON_WAY"
    ARRIVED_AT_TARGET = "ARRIVED_AT_TARGET"
    DELIVERY_IN_PROGRESS = "DELIVERY_IN_PROGRESS"
    DELIVERED = "DELIVERED"
    DELIVERY_FAILED = "DELIVERY_FAILED"
    DELIVERY_CANCELLED = "DELIVERY_CANCELLED"
    RETURNING = "RETURNING"


class PriorityCategory(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TargetStatus(str, Enum):
    NEW = "NEW"
    PROPOSED = "PROPOSED"
    ASSIGNED = "ASSIGNED"
    SERVED = "SERVED"


# coordination message types
TASK_RECOMMENDATION = "TASK_RECOMMENDATION"
TASK_REJECTED = "TASK_REJECTED"
OPERATOR_COMMAND = "OPERATOR_COMMAND"
OPERATOR_ACK = "OPERATOR_ACK"
MISSION_COMMAND = "MISSION_COMMAND"


class Topics:
    COORDINATION = "swarm/coordination"
    MISSION = "swarm/mission"

    @staticmethod
    def status(d: str) -> str:
        return f"swarm/{d}/status"

    @staticmethod
    def telemetry(d: str) -> str:
        return f"swarm/{d}/telemetry"

    @staticmethod
    def detection(d: str) -> str:
        return f"swarm/{d}/detection"

    @staticmethod
    def delivery(d: str) -> str:
        return f"swarm/{d}/delivery"


def now() -> float:
    return time.time()


def sanitize(o: Any) -> Any:
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [sanitize(v) for v in o]
    if hasattr(o, "item") and callable(o.item):
        try:
            return sanitize(o.item())
        except Exception:
            return None
    return o


def dumps(obj: Any) -> str:
    return json.dumps(sanitize(obj), separators=(",", ":"), allow_nan=False)


def is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def valid_latlon(lat: Any, lon: Any) -> bool:
    return is_num(lat) and is_num(lon) and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def validate_detection_message(m: dict) -> Optional[str]:
    det = m.get("detection")
    if not isinstance(det, dict):
        return "missing detection"
    if not isinstance(det.get("detection_id"), str) or not isinstance(det.get("class"), str):
        return "bad detection_id/class"
    if not is_num(det.get("confidence")) or not 0.0 <= det["confidence"] <= 1.0:
        return "bad confidence"
    loc = det.get("location")
    if not isinstance(loc, dict) or not valid_latlon(loc.get("latitude"), loc.get("longitude")):
        return "bad location"
    if not isinstance(m.get("detecting_drone"), str):
        return "missing detecting_drone"
    pr = m.get("priority")
    if not isinstance(pr, dict) or not is_num(pr.get("priority_score")):
        return "bad priority"
    rb = m.get("required_boxes")
    if not isinstance(rb, int) or isinstance(rb, bool) or not 0 <= rb <= 64:
        return "bad required_boxes"
    if not is_num(m.get("required_payload_weight_kg")):
        return "bad required_payload_weight_kg"
    return None


def validate_status(m: dict) -> Optional[str]:
    if not isinstance(m.get("drone_id"), str):
        return "missing drone_id"
    for k in ("position", "battery", "payload", "mission", "delivery", "connection"):
        if not isinstance(m.get(k), dict):
            if m.get("lwt"):
                return None
            return f"missing {k}"
    return None


def validate_delivery(m: dict) -> Optional[str]:
    for k in ("delivery_id", "drone_id", "target_id", "status"):
        if not isinstance(m.get(k), str):
            return f"missing {k}"
    if m["status"] not in {s.value for s in DeliveryState}:
        return "unknown status"
    return None


def detection_message(drone_id: str, target_id: str, detection: dict, priority: dict,
                      required_boxes: int, required_payload_weight_kg: float) -> dict:
    return {
        "message_type": "detection",
        "drone_id": drone_id,
        "detecting_drone": drone_id,
        "timestamp": now(),
        "target_id": target_id,
        "detection": detection,
        "location": detection.get("location"),
        "confidence": detection.get("confidence"),
        "priority": priority,
        "required_boxes": required_boxes,
        "required_payload_weight_kg": required_payload_weight_kg,
    }