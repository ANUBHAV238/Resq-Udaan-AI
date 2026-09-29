from __future__ import annotations

import copy
import logging
import signal
import threading
import time
from typing import Dict, Optional

import yaml

from .delivery_manager import DeliveryManager
from .detector import Detector, FrameSource
from .kml_processor import load_area
from .lawnmower import generate_tracks, stitch
from .mavros_adapter import MavrosAdapter
from .messages import Topics, detection_message, validate_status
from .mission_manager import MissionManager
from .mqtt_manager import MqttManager
from .payload_manager import PayloadManager, ServoReleaseMechanism
from .priority_model import PriorityModel, build_features
from .target_fusion import TargetFusion
from .task_allocator import SectorAllocator, TaskAllocator

log = logging.getLogger("drone")


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(mission_path: str, drone_path: str) -> dict:
    with open(mission_path) as f:
        m = yaml.safe_load(f) or {}
    with open(drone_path) as f:
        d = yaml.safe_load(f) or {}
    return _merge(m, d)


class DroneManager:
    def __init__(self, drone_id: str, namespace: str, drone_cfg_path: str, mission_cfg_path: str):
        cfg = load_config(mission_cfg_path, drone_cfg_path)
        if cfg.get("drone_id") != drone_id or cfg.get("mavros_namespace") != namespace:
            raise ValueError("config drone_id/mavros_namespace does not match this entry point")
        if namespace != f"/{drone_id}/mavros":
            raise ValueError("MAVROS namespace must be /<drone_id>/mavros")
        self.cfg, self.id, self.ns = cfg, drone_id, namespace
        self.drones = cfg["drones"]
        if drone_id not in self.drones:
            raise ValueError(f"{drone_id} not in mission drones list")
        self._stop = threading.Event()
        self._fleet: Dict[str, dict] = {}
        self._fleet_lock = threading.Lock()

        # ---- mission area & coverage plan (deterministic: every drone computes the same plan) ----
        area = load_area(cfg["mission_area_kml"])
        cov = cfg["coverage"]
        tracks = generate_tracks(area.rings, cov["TRACK_SPACING_M"], cov["WAYPOINT_SPACING_M"],
                                 cov.get("boundary_margin_m", 0.0), cov.get("heading_deg"))
        sectors = SectorAllocator(len(self.drones)).split(tracks)
        my_tracks = SectorAllocator.assign(sectors, self.drones)[drone_id]
        idx = cfg.get("sector_index", self.drones.index(drone_id))
        if idx != self.drones.index(drone_id):
            my_tracks = sectors[idx]
        path = stitch(my_tracks)
        self.waypoints = [area.frame.to_geo(x, y) for x, y in path]
        self.frame = area.frame
        log.info("%s: %d waypoints in sector %d", drone_id, len(self.waypoints), idx)

        # ---- components ----
        offline = {"drone_id": drone_id, "connection": {"mavros": False, "mqtt": False}, "offline": True}
        self.mqtt = MqttManager(drone_id, cfg["mqtt"], lwt=(Topics.status(drone_id), offline))
        self.mavros = MavrosAdapter(drone_id, namespace, cfg.get("mavros"))
        pc = cfg["payload"]
        self.payload = PayloadManager(pc["box_count"], pc["box_weight_kg"])
        self.release = ServoReleaseMechanism(cfg["payload_release"], pc["box_count"], self.mavros.send_command_long)
        self.priority = PriorityModel(cfg["priority"])
        self.fusion = TargetFusion(drone_id, self.frame, cfg["fusion"], self.priority.categorize)
        self.allocator = TaskAllocator(cfg, cfg["delivery"], self.frame)
        self.deliveries = DeliveryManager(drone_id, cfg["delivery"].get("terminal_hold_s", 5.0))
        self.mission = MissionManager(drone_id, cfg, self.frame, self.waypoints, self.mavros, self.mqtt,
                                      self.payload, self.release, self.deliveries, self.fusion,
                                      self.allocator, self.get_fleet)
        self.deliveries.on_change = self.mission.publish_delivery

        cam = cfg.get("camera", {})
        self.camera_on = bool(cam.get("enabled"))
        self.frames = FrameSource(cam) if self.camera_on else None
        self.detector = Detector(drone_id, cfg["detector"], cam) if self.camera_on else None

        # ---- MQTT wiring: own process only publishes own topics; subscribes to peers ----
        peers = [d for d in self.drones if d != drone_id]
        for p in peers:
            self.mqtt.subscribe(Topics.status(p), self._on_status)
            self.mqtt.subscribe(Topics.telemetry(p), self._on_telemetry)
            self.mqtt.subscribe(Topics.detection(p), self.mission.enqueue)
            self.mqtt.subscribe(Topics.delivery(p), self.mission.enqueue)
        self.mqtt.subscribe(Topics.COORDINATION, self.mission.enqueue)
        self.mqtt.subscribe(Topics.MISSION, self.mission.enqueue)
        self._peer_telemetry: Dict[str, dict] = {}

    # ------------------------------------------------------------------ peers
    def _on_status(self, topic: str, data: dict):
        err = validate_status(data)
        did = data.get("drone_id")
        if err or did != topic.split("/")[1] or did == self.id:
            log.warning("invalid status on %s: %s", topic, err or "id mismatch")
            return
        data = dict(data)
        data["_rx"] = time.time()
        if data.get("lwt") or data.get("offline"):
            data["_rx"] = 0.0
        with self._fleet_lock:
            self._fleet[did] = data

    def _on_telemetry(self, topic: str, data: dict):
        did = topic.split("/")[1]
        if did != self.id:
            self._peer_telemetry[did] = data

    def get_fleet(self) -> Dict[str, dict]:
        with self._fleet_lock:
            fleet = dict(self._fleet)
        me = self.build_status()
        me["_rx"] = time.time()
        fleet[self.id] = me
        return fleet

    # ------------------------------------------------------------------ status / telemetry
    def build_status(self) -> dict:
        pos, bat = self.mavros.get_position(), self.mavros.get_battery()
        st, vel = self.mavros.get_state(), self.mavros.get_velocity()
        prog = self.mission.progress()
        return {
            "message_type": "status",
            "drone_id": self.id,
            "timestamp": time.time(),
            "position": {"latitude": pos["latitude"], "longitude": pos["longitude"],
                         "altitude_m": pos["altitude_m"], "local": pos["local"]},
            "velocity": vel,
            "battery": {"percentage": bat["percentage"], "voltage": bat["voltage"]},
            "flight": {"armed": st["armed"], "mode": st["mode"]},
            "payload": self.payload.status(),
            "mission": {"status": prog["status"], "current_waypoint": prog["current_waypoint"],
                        "total_waypoints": prog["total_waypoints"], "route_done": prog["route_done"],
                        "error": prog["error"]},
            "delivery": self.deliveries.view(),
            "detector": {"enabled": self.camera_on, "priority_model_loaded": self.priority.model_loaded},
            "connection": {"mavros": st["mavros_connected"], "mqtt": self.mqtt.connected},
        }

    def build_telemetry(self) -> dict:
        pos, bat = self.mavros.get_position(), self.mavros.get_battery()
        st, vel = self.mavros.get_state(), self.mavros.get_velocity()
        wp = self.mavros.get_current_waypoint()
        return {
            "message_type": "telemetry", "drone_id": self.id, "timestamp": time.time(),
            "latitude": pos["latitude"], "longitude": pos["longitude"], "altitude_m": pos["altitude_m"],
            "altitude_amsl_m": pos["altitude_amsl_m"], "gps_fix": pos["gps_fix"], "local": pos["local"],
            "velocity": vel,
            "battery": {"percentage": bat["percentage"], "voltage": bat["voltage"], "current": bat["current"]},
            "armed": st["armed"], "flight_mode": st["mode"], "mavros_connected": st["mavros_connected"],
            "current_waypoint": wp["current_seq"], "total_waypoints": len(self.waypoints),
            "mission_status": self.mission.state.value,
        }

    def _publish_loop(self):
        s_int = 1.0 / self.cfg.get("status_hz", 2.0)
        t_int = 1.0 / self.cfg.get("telemetry_hz", 5.0)
        last_s = last_t = 0.0
        while not self._stop.is_set():
            now = time.time()
            if now - last_s >= s_int:
                self.mqtt.publish(Topics.status(self.id), self.build_status(), "status")
                last_s = now
            if now - last_t >= t_int:
                self.mqtt.publish(Topics.telemetry(self.id), self.build_telemetry(), "telemetry")
                last_t = now
            time.sleep(0.05)

    # ------------------------------------------------------------------ vision
    def _vision_loop(self):
        dcfg = self.cfg["detector"]
        interval = dcfg.get("interval_s", 0.3)
        min_alt = dcfg.get("min_altitude_m", 5.0)
        target_classes = set(dcfg.get("class_map", {0: "person"}).values())
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._vision_step(min_alt, target_classes)
            except Exception:
                log.exception("vision step failed")
            time.sleep(max(0.01, interval - (time.time() - t0)))

    def _vision_step(self, min_alt: float, target_classes: set):
        st = self.mavros.get_state()
        pos, att = self.mavros.get_position(), self.mavros.get_attitude()
        if not st["armed"] or att is None or pos["latitude"] is None or pos["altitude_m"] is None:
            return
        if pos["altitude_m"] < min_alt or (pos["age_s"] or 99) > 1.5 or att["age_s"] > 1.0:
            return
        frame = self.frames.read()
        if frame is None:
            return
        pose = {"latitude": pos["latitude"], "longitude": pos["longitude"],
                "altitude_m": pos["altitude_m"], "quaternion": att["quaternion"]}
        for det in self.detector.detect(frame, pose):
            if det["class"] not in target_classes or det["location"] is None:
                continue
            feats = build_features(det, pose, pos["latitude"], pos["longitude"])
            feats["detection_id"] = det["detection_id"]
            pr = self.priority.predict(feats)
            boxes = self.mission.boxes_for(pr["priority_category"])
            msg = detection_message(self.id, "", det, pr, boxes, boxes * self.payload.box_weight_kg)
            msg["target_id"] = ""
            self.mission.on_own_detection(self._with_target_hint(msg))

    def _with_target_hint(self, msg: dict) -> dict:
        msg["target_id"] = None
        return msg

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        self.mqtt.start()
        threading.Thread(target=self.mission.run, name="mission", daemon=True).start()
        threading.Thread(target=self._publish_loop, name="publisher", daemon=True).start()
        if self.camera_on:
            self.frames.start()
            threading.Thread(target=self._vision_loop, name="vision", daemon=True).start()

    def shutdown(self):
        self._stop.set()
        self.mission.stop()
        if self.frames:
            self.frames.stop()
        try:
            self.mqtt.publish(Topics.status(self.id), {"drone_id": self.id, "offline": True, "lwt": True,
                                                       "connection": {"mavros": False, "mqtt": False}}, "status")
            time.sleep(0.3)
            self.mqtt.stop()
        finally:
            self.mavros.shutdown()


def run_drone(drone_id: str, namespace: str, drone_cfg: str, mission_cfg: str) -> int:
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s [{drone_id}] %(name)s %(levelname)s: %(message)s")
    dm = DroneManager(drone_id, namespace, drone_cfg, mission_cfg)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    dm.start()
    try:
        while not stop.is_set():
            time.sleep(0.5)
    finally:
        dm.shutdown()
    return 0