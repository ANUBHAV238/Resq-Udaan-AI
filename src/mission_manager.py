from __future__ import annotations

import hmac
import logging
import queue
import re
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from .delivery_manager import ACTIVE as ACTIVE_DELIVERY, Delivery, DeliveryError, DeliveryManager
from .kml_processor import LocalFrame
from .messages import (MISSION_COMMAND, OPERATOR_ACK, OPERATOR_COMMAND, TASK_RECOMMENDATION, TASK_REJECTED,
                       DeliveryState as DS, MissionState as MS, TargetStatus as TS, Topics,
                       validate_delivery, validate_detection_message)
from .mqtt_manager import MqttManager
from .payload_manager import PayloadManager
from .target_fusion import TargetFusion
from .task_allocator import TaskAllocator

log = logging.getLogger("mission")

_DELIVERY_NUM = re.compile(r"delivery_(\d+)")


class MissionManager:
    """Mission state machine + delivery workflow. Every delivery step that commits the drone
    (dispatch, release) needs an explicit, authenticated operator command."""

    def __init__(self, drone_id: str, cfg: dict, frame: LocalFrame, waypoints: List[Tuple[float, float]],
                 mavros, mqtt: MqttManager, payload: PayloadManager, release, deliveries: DeliveryManager,
                 fusion: TargetFusion, allocator: TaskAllocator, get_fleet: Callable[[], Dict[str, dict]]):
        self.id = drone_id
        self.cfg = cfg
        self.dcfg = cfg["delivery"]
        self.flight = cfg["flight"]
        self.modes = self.flight["modes"]
        self.frame = frame
        self.waypoints = waypoints
        self.mavros = mavros
        self.mqtt = mqtt
        self.payload = payload
        self.release = release
        self.deliveries = deliveries
        self.fusion = fusion
        self.allocator = allocator
        self.get_fleet = get_fleet
        self.token = (cfg.get("operator") or {}).get("token") or ""

        self._q: "queue.Queue[Tuple[str, dict]]" = queue.Queue()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self.state = MS.READY if waypoints else MS.ERROR
        self.error_reason: Optional[str] = None if waypoints else "empty sector"
        self.route_done = False
        self.resume_seq: Optional[int] = None
        self._diverted = False
        self._paused = False
        self._resume_at: Optional[float] = None
        self._td_since = 0.0
        self._flight_t0 = 0.0
        self._return_t = 0.0
        self._last_pub: Dict[str, float] = {}
        self._max_delivery_no = 0
        self._last_reco_log = 0.0

    # ------------------------------------------------------------------ public
    def set_state(self, new: MS, reason: str = ""):
        with self._lock:
            if new != self.state:
                log.info("mission %s -> %s %s", self.state.value, new.value, reason)
                self.state = new

    def progress(self) -> dict:
        wp = self.mavros.get_current_waypoint()
        n = len(self.waypoints)
        cur = min(max(wp["current_seq"] or 0, 0), n)
        return {"status": self.state.value, "current_waypoint": cur, "total_waypoints": n,
                "route_done": self.route_done, "error": self.error_reason}

    def enqueue(self, topic: str, data: dict):
        self._q.put((topic, data))

    def run(self):
        while not self._stop.is_set():
            try:
                self._drain()
                now = time.time()
                with self._lock:
                    self._tick_state(now)
                    self._tick_delivery(now)
                self.fusion.expire(now)
                self._tick_coordination(now)
            except Exception:
                log.exception("mission tick failed")
            time.sleep(0.2)

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------ own detections
    def on_own_detection(self, msg: dict):
        target, event = self.fusion.update(msg)
        if target is None:
            return
        now = time.time()
        with self._lock:
            if event == "new" and self.state == MS.SEARCHING:
                self._td_since = now
                self.set_state(MS.TARGET_DETECTED, target.target_id)
        if event == "new" or now - self._last_pub.get(target.target_id, 0.0) >= self.dcfg.get("republish_s", 3.0):
            out = dict(msg)
            out["target_id"] = target.target_id
            self.mqtt.publish(Topics.detection(self.id), out, "detection")
            self._last_pub[target.target_id] = now

    def boxes_for(self, category: str) -> int:
        n = int(self.dcfg.get("boxes_by_category", {}).get(category, 1))
        return max(0, min(n, self.payload.box_count))

    # ------------------------------------------------------------------ message routing
    def _drain(self):
        while True:
            try:
                topic, data = self._q.get_nowait()
            except queue.Empty:
                return
            try:
                if topic == Topics.MISSION:
                    self._on_mission_command(data)
                elif topic == Topics.COORDINATION:
                    t = data.get("type")
                    if t == TASK_RECOMMENDATION:
                        self._on_recommendation(data)
                    elif t == TASK_REJECTED:
                        self._on_rejected(data)
                    elif t == OPERATOR_COMMAND:
                        self._on_operator_command(data)
                elif topic.endswith("/detection"):
                    self._on_peer_detection(data)
                elif topic.endswith("/delivery"):
                    self._on_peer_delivery(data)
            except Exception:
                log.exception("message handling failed on %s", topic)

    def _on_peer_detection(self, data: dict):
        err = validate_detection_message(data)
        if err:
            log.warning("invalid peer detection dropped: %s", err)
            return
        self.fusion.update(data)

    def _on_peer_delivery(self, data: dict):
        err = validate_delivery(data)
        if err:
            log.warning("invalid delivery message dropped: %s", err)
            return
        self._apply_delivery_to_fusion(data)

    def _apply_delivery_to_fusion(self, d: dict):
        m = _DELIVERY_NUM.fullmatch(d["delivery_id"])
        if m:
            self._max_delivery_no = max(self._max_delivery_no, int(m.group(1)))
        st, tid, did, drone = d["status"], d["target_id"], d["delivery_id"], d["drone_id"]
        now = time.time()
        if st in ("DELIVERY_ASSIGNED", "DELIVERY_RESERVED", "DELIVERY_ON_WAY", "ARRIVED_AT_TARGET",
                  "DELIVERY_IN_PROGRESS"):
            self.fusion.set_status(tid, TS.ASSIGNED, delivery_id=did, proposed_drone=drone)
        elif st == "DELIVERED":
            self.fusion.set_status(tid, TS.SERVED, delivery_id=did)
        elif st in ("DELIVERY_FAILED", "DELIVERY_CANCELLED"):
            cd = self.dcfg.get("retry_cooldown_s", 60.0)
            self.fusion.set_status(tid, TS.NEW, retry_after=now + 5.0, exclude_drone=drone,
                                   exclude_until=now + cd)

    def publish_delivery(self, d: Delivery):
        payload = d.to_dict()
        self.mqtt.publish(Topics.delivery(self.id), payload, "delivery")
        self._apply_delivery_to_fusion(payload)

    # ------------------------------------------------------------------ operator / mission commands
    def _authorized(self, data: dict) -> bool:
        tok = data.get("token")
        return bool(self.token) and isinstance(tok, str) and hmac.compare_digest(tok, self.token)

    def _ack(self, data: dict, ok: bool, detail: str = ""):
        self.mqtt.publish(Topics.COORDINATION, {"type": OPERATOR_ACK, "drone_id": self.id,
                                                "command": data.get("command"),
                                                "delivery_id": data.get("delivery_id"),
                                                "accepted": ok, "detail": detail}, "coordination")

    def _on_mission_command(self, data: dict):
        if data.get("type") != MISSION_COMMAND:
            return
        if data.get("target") not in ("all", self.id):
            return
        if not self._authorized(data):
            log.warning("unauthenticated mission command ignored")
            return
        cmd = data.get("command")
        with self._lock:
            if cmd == "START":
                if self.state != MS.READY:
                    return self._ack(data, False, f"state {self.state.value}")
                threading.Thread(target=self._start_flight, name="start-flight", daemon=True).start()
            elif cmd == "HOLD":
                self._paused = True
                self.mavros.set_mode(self.modes["hold"])
            elif cmd == "RESUME":
                if self._paused and not self._diverted:
                    self._paused = False
                    self.mavros.set_mode(self.modes["auto"])
            elif cmd == "RTL":
                d = self.deliveries.active()
                if d:
                    self._terminate(d, DS.DELIVERY_CANCELLED, "operator RTL", resume=False)
                self._begin_return("operator RTL")
            else:
                return self._ack(data, False, "unknown command")
        self._ack(data, True)

    def _start_flight(self):
        try:
            if not self.waypoints:
                raise RuntimeError("no waypoints")
            pos = self.mavros.get_position()
            if not self.mavros.get_state()["mavros_connected"] or pos["latitude"] is None:
                raise RuntimeError("MAVROS/GPS not ready")
            if not self.mavros.push_mission(self.waypoints, self.cfg["coverage"]["altitude_m"],
                                            (pos["latitude"], pos["longitude"])):
                raise RuntimeError("mission upload failed")
            alt = self.flight["takeoff_altitude_m"]
            if self.flight.get("auto_takeoff"):
                if not (self.mavros.set_mode(self.modes["guided"]) and self.mavros.arm(True)
                        and self.mavros.takeoff(alt)):
                    raise RuntimeError("arm/takeoff rejected")
            t0 = time.time()
            while time.time() - t0 < self.flight.get("takeoff_wait_s", 120):
                st, p = self.mavros.get_state(), self.mavros.get_position()
                if st["armed"] and (p["altitude_m"] or 0.0) >= 0.85 * alt:
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("timeout waiting for armed + takeoff altitude")
            if not self.mavros.set_mode(self.modes["auto"]):
                raise RuntimeError("AUTO mode rejected")
            with self._lock:
                self._flight_t0 = time.time()
                self.route_done = False
                self.set_state(MS.SEARCHING, "mission started")
        except Exception as e:
            log.error("start failed: %s", e)
            with self._lock:
                self.error_reason = str(e)
                self.set_state(MS.ERROR, str(e))

    # ------------------------------------------------------------------ recommendation flow
    def _on_recommendation(self, m: dict):
        for k in ("delivery_id", "target_id", "assigned_drone", "boxes", "destination"):
            if k not in m:
                return
        did, tid, who = m["delivery_id"], m["target_id"], m["assigned_drone"]
        mm = _DELIVERY_NUM.fullmatch(did)
        if mm:
            self._max_delivery_no = max(self._max_delivery_no, int(mm.group(1)))
        self.fusion.set_status(tid, TS.PROPOSED, delivery_id=did, proposed_drone=who,
                               expires=float(m.get("expires", time.time() + 30)))
        if who != self.id or self.deliveries.get(did) is not None:
            return
        boxes = int(m["boxes"])
        reasons = []
        if self.deliveries.active():
            reasons.append("delivery already active")
        if self.state not in (MS.SEARCHING, MS.TARGET_DETECTED):
            reasons.append(f"mission state {self.state.value}")
        if self.payload.status()["available_boxes"] < boxes:
            reasons.append("not enough available boxes")
        dest = m["destination"]
        if reasons:
            self.mqtt.publish(Topics.COORDINATION, {"type": TASK_REJECTED, "drone_id": self.id,
                                                    "delivery_id": did, "target_id": tid,
                                                    "reason": "; ".join(reasons)}, "coordination")
            return
        with self._lock:
            try:
                self.deliveries.create(did, tid, {"latitude": dest["latitude"], "longitude": dest["longitude"]},
                                       boxes, boxes * self.payload.box_weight_kg, m.get("priority"))
            except DeliveryError as e:
                log.warning("cannot create delivery: %s", e)
                return
            self.set_state(MS.TASK_ASSIGNED, did)
        log.info("delivery %s assigned to me for %s; waiting for operator approval", did, tid)

    def _on_rejected(self, m: dict):
        tid = m.get("target_id")
        who = m.get("drone_id")
        if tid and who:
            self.fusion.set_status(tid, TS.NEW, retry_after=time.time() + 2.0, exclude_drone=who,
                                   exclude_until=time.time() + self.dcfg.get("retry_cooldown_s", 60.0))

    def _on_operator_command(self, m: dict):
        d = self.deliveries.get(m.get("delivery_id"))
        if d is None or d.drone_id != self.id:
            return
        if not self._authorized(m):
            log.warning("unauthenticated operator command ignored")
            return self._ack(m, False, "unauthorized")
        cmd = m.get("command")
        with self._lock:
            if cmd == "APPROVE_DELIVERY":
                if d.status != DS.DELIVERY_ASSIGNED:
                    return self._ack(m, False, f"delivery is {d.status.value}")
                self._approve(d)
                return self._ack(m, True)
            if cmd == "REJECT_DELIVERY":
                if d.status != DS.DELIVERY_ASSIGNED:
                    return self._ack(m, False, f"delivery is {d.status.value}")
                self._terminate(d, DS.DELIVERY_CANCELLED, "operator rejected")
                return self._ack(m, True)
            if cmd == "CONFIRM_RELEASE":
                if d.status != DS.ARRIVED_AT_TARGET:
                    return self._ack(m, False, f"delivery is {d.status.value}")
                self.deliveries.transition(d.delivery_id, DS.DELIVERY_IN_PROGRESS, "operator confirmed release")
                self.set_state(MS.DELIVERY_IN_PROGRESS)
                threading.Thread(target=self._do_release, args=(d,), name="release", daemon=True).start()
                return self._ack(m, True)
            if cmd == "ABORT_DELIVERY":
                if d.status not in ACTIVE_DELIVERY or d.status == DS.DELIVERY_IN_PROGRESS:
                    return self._ack(m, False, f"cannot abort in {d.status.value}")
                self._terminate(d, DS.DELIVERY_CANCELLED, "operator aborted")
                return self._ack(m, True)
        self._ack(m, False, "unknown command")

    # ------------------------------------------------------------------ delivery execution
    def _battery_ok_for(self, d: Delivery) -> Tuple[bool, str]:
        pct = self.mavros.get_battery()["percentage"]
        pos = self.mavros.get_position()
        if pct is None or pos["latitude"] is None:
            return False, "battery/position unknown"
        dist = self.frame.distance_m(pos["latitude"], pos["longitude"], d.destination["latitude"],
                                     d.destination["longitude"])
        need = self.dcfg["min_battery_pct"] + dist / 1000.0 * self.dcfg["drain_pct_per_km"] * 2 \
            + self.dcfg["battery_reserve_pct"]
        return (pct >= need), f"battery {pct:.0f}% < required {need:.0f}%"

    def _approve(self, d: Delivery):
        ok, why = self._battery_ok_for(d)
        if not ok:
            return self._terminate(d, DS.DELIVERY_FAILED, why)
        if not self.payload.reserve(d.delivery_id, d.boxes):
            return self._terminate(d, DS.DELIVERY_FAILED, "box reservation failed")
        self.deliveries.transition(d.delivery_id, DS.DELIVERY_RESERVED, "operator approved")
        self.set_state(MS.DELIVERY_RESERVED)
        wp = self.mavros.get_current_waypoint()
        self.resume_seq = wp["current_seq"] or ((wp["last_reached"] or 0) + 1)
        if not self.mavros.set_mode(self.modes["guided"]):
            return self._terminate(d, DS.DELIVERY_FAILED, "GUIDED mode rejected")
        self._diverted = True
        self.mavros.start_goto(d.destination["latitude"], d.destination["longitude"], self.dcfg["altitude_m"])
        self.deliveries.transition(d.delivery_id, DS.DELIVERY_ON_WAY, "en route")
        self.set_state(MS.DELIVERY_ON_WAY)

    def _do_release(self, d: Delivery):
        ok = False
        try:
            ok = self.release.release(self.payload.slots_for(d.delivery_id))
        except Exception:
            log.exception("release actuation error")
        with self._lock:
            if ok and self.payload.commit(d.delivery_id):
                self.deliveries.transition(d.delivery_id, DS.DELIVERED, "release actuator confirmed")
                self.set_state(MS.DELIVERED)
                self._resume_at = time.time() + self.dcfg.get("terminal_hold_s", 5.0)
            else:
                self._terminate(d, DS.DELIVERY_FAILED, "release actuation failed")

    def _terminate(self, d: Delivery, new: DS, reason: str, resume: bool = True):
        self.payload.release(d.delivery_id)                       # reservation is always given back
        self.deliveries.try_transition(d.delivery_id, new, reason)
        log.warning("delivery %s -> %s (%s)", d.delivery_id, new.value, reason)
        if resume:
            self._end_delivery_flight()

    def _end_delivery_flight(self):
        if self._diverted:
            self.mavros.stop_goto()
            self._diverted = False
            if not self.route_done and self.state not in (MS.RETURNING, MS.MISSION_COMPLETE):
                self.mavros.set_mode(self.modes["auto"])
                if self.resume_seq:
                    self.mavros.set_current_waypoint(self.resume_seq)
        self._resume_at = None
        if self.state in (MS.ERROR, MS.RETURNING, MS.MISSION_COMPLETE):
            return
        if self.route_done:
            self._begin_return("route complete after delivery")
        else:
            self.set_state(MS.SEARCHING, "resumed search")

    def _begin_return(self, reason: str):
        self.mavros.stop_goto()
        self._diverted = False
        self.mavros.set_mode(self.modes["rtl"])
        self._return_t = time.time()
        self.deliveries.mark_returning()
        self.set_state(MS.RETURNING, reason)

    # ------------------------------------------------------------------ periodic checks
    def _tick_state(self, now: float):
        st = self.state
        if st == MS.TARGET_DETECTED and now - self._td_since > self.dcfg.get("target_detected_hold_s", 5.0):
            self.set_state(MS.SEARCHING)
        if st in (MS.SEARCHING, MS.TARGET_DETECTED, MS.TASK_ASSIGNED) and not self._diverted and self.waypoints:
            wp = self.mavros.get_current_waypoint()
            lr, lt = wp["last_reached"], wp["last_reached_t"]
            if lr is not None and lt and lt > self._flight_t0 and lr >= len(self.waypoints):
                self.route_done = True
            if self.route_done and not self.deliveries.active():
                self._begin_return("coverage complete")
        if st == MS.RETURNING and now - self._return_t > 10.0 and not self.mavros.get_state()["armed"]:
            self.set_state(MS.MISSION_COMPLETE)

    def _tick_delivery(self, now: float):
        if self._resume_at and now >= self._resume_at:
            self._end_delivery_flight()
        d = self.deliveries.active()
        if d is None:
            return
        if d.status == DS.DELIVERY_ASSIGNED:
            if now - d.created_at > self.dcfg["approval_timeout_s"]:
                self._terminate(d, DS.DELIVERY_CANCELLED, "operator approval timeout")
        elif d.status == DS.DELIVERY_ON_WAY:
            pct = self.mavros.get_battery()["percentage"]
            if pct is not None and pct < self.dcfg["min_battery_pct"] * 0.6:
                return self._terminate(d, DS.DELIVERY_FAILED, "battery critical")
            if now - d.status_ts > self.dcfg["max_transit_s"]:
                return self._terminate(d, DS.DELIVERY_FAILED, "transit timeout")
            if self._arrived(d):
                self.deliveries.transition(d.delivery_id, DS.ARRIVED_AT_TARGET, "at target; awaiting operator")
                self.set_state(MS.ARRIVED_AT_TARGET)
        elif d.status == DS.ARRIVED_AT_TARGET:
            if now - d.status_ts > self.dcfg["release_confirm_timeout_s"]:
                self._terminate(d, DS.DELIVERY_FAILED, "no operator release confirmation")

    def _arrived(self, d: Delivery) -> bool:
        pos, vel = self.mavros.get_position(), self.mavros.get_velocity()
        if pos["latitude"] is None or pos["altitude_m"] is None:
            return False
        dist = self.frame.distance_m(pos["latitude"], pos["longitude"], d.destination["latitude"],
                                     d.destination["longitude"])
        return (dist <= self.dcfg["arrival_radius_m"]
                and abs(pos["altitude_m"] - self.dcfg["altitude_m"]) <= self.dcfg["arrival_alt_tol_m"]
                and (vel["speed"] or 0.0) <= self.dcfg["arrival_speed_mps"])

    # ------------------------------------------------------------------ leader: recommendations
    def _tick_coordination(self, now: float):
        if self.state in (MS.ERROR, MS.IDLE):
            return
        fleet = self.get_fleet()
        peer_to = self.allocator.peer_timeout
        alive = sorted(d for d, s in fleet.items() if now - s.get("_rx", 0) <= peer_to)
        if not alive or alive[0] != self.id:
            return                                                    # only the lowest live drone id recommends
        busy = set(self.fusion.proposed_drones(now))
        for t in self.allocator.order_targets(self.fusion.pending(now)):
            ranking = self.allocator.rank(t, t.required_boxes, fleet, t.excluded_drones(now), busy)
            best = self.allocator.best(ranking)
            if best is None:
                if now - self._last_reco_log > 10.0:
                    self._last_reco_log = now
                    log.info("no eligible drone for %s (%d boxes): %s", t.target_id, t.required_boxes,
                             [(r["drone_id"], r["reasons"]) for r in ranking])
                continue
            self._max_delivery_no += 1
            did = f"delivery_{self._max_delivery_no:03d}"
            expires = now + self.dcfg.get("proposal_timeout_s", 30.0)
            self.mqtt.publish(Topics.COORDINATION, {
                "type": TASK_RECOMMENDATION, "leader": self.id, "delivery_id": did,
                "target_id": t.target_id, "assigned_drone": best["drone_id"],
                "boxes": t.required_boxes, "payload_weight_kg": t.required_payload_weight_kg,
                "destination": {"latitude": t.lat, "longitude": t.lon},
                "priority": {"priority_score": t.priority_score, "priority_category": t.priority_category},
                "ranking": ranking, "expires": expires, "requires_operator_approval": True,
            }, "coordination")
            self.fusion.set_status(t.target_id, TS.PROPOSED, delivery_id=did,
                                   proposed_drone=best["drone_id"], expires=expires)
            busy.add(best["drone_id"])