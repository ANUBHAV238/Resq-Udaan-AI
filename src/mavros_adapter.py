from __future__ import annotations

import logging
import math
import threading
import time
from typing import List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import GlobalPositionTarget, State, Waypoint, WaypointList, WaypointReached
from mavros_msgs.srv import (CommandBool, CommandLong, CommandTOL, SetMode, WaypointClear,
                             WaypointPush, WaypointSetCurrent)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from sensor_msgs.msg import BatteryState, NavSatFix
from std_msgs.msg import Float64

log = logging.getLogger("mavros")


def _f(v) -> Optional[float]:
    return float(v) if v is not None and math.isfinite(v) else None


class MavrosAdapter:
    """ROS 2 MAVROS access for exactly one vehicle namespace (e.g. /drone1/mavros)."""

    def __init__(self, drone_id: str, namespace: str, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.drone_id = drone_id
        self.ns = namespace.rstrip("/")
        self.stale_s = cfg.get("stale_after_s", 3.0)
        self.svc_timeout = cfg.get("service_timeout_s", 3.0)
        if not rclpy.ok():
            rclpy.init()
        self.node = Node(f"{drone_id}_mavros_adapter")
        self._lock = threading.Lock()
        self._d = {}
        self._goto: Optional[Tuple[float, float, float]] = None
        self._goto_lock = threading.Lock()

        sub = self.node.create_subscription
        ns = self.ns
        sub(State, f"{ns}/state", lambda m: self._store("state", m), qos_profile_system_default)
        sub(NavSatFix, f"{ns}/global_position/global", lambda m: self._store("fix", m), qos_profile_sensor_data)
        sub(Float64, f"{ns}/global_position/rel_alt", lambda m: self._store("rel_alt", m), qos_profile_sensor_data)
        sub(PoseStamped, f"{ns}/local_position/pose", lambda m: self._store("pose", m), qos_profile_sensor_data)
        sub(TwistStamped, f"{ns}/local_position/velocity_local", lambda m: self._store("vel", m),
            qos_profile_sensor_data)
        sub(BatteryState, f"{ns}/battery", lambda m: self._store("batt", m), qos_profile_sensor_data)
        sub(WaypointList, f"{ns}/mission/waypoints", lambda m: self._store("wps", m), qos_profile_system_default)
        sub(WaypointReached, f"{ns}/mission/reached", lambda m: self._store("reached", m),
            qos_profile_system_default)

        self._sp_pub = self.node.create_publisher(GlobalPositionTarget, f"{ns}/setpoint_raw/global",
                                                  qos_profile_system_default)
        self._c_mode = self.node.create_client(SetMode, f"{ns}/set_mode")
        self._c_arm = self.node.create_client(CommandBool, f"{ns}/cmd/arming")
        self._c_tol = self.node.create_client(CommandTOL, f"{ns}/cmd/takeoff")
        self._c_push = self.node.create_client(WaypointPush, f"{ns}/mission/push")
        self._c_clear = self.node.create_client(WaypointClear, f"{ns}/mission/clear")
        self._c_cur = self.node.create_client(WaypointSetCurrent, f"{ns}/mission/set_current")
        self._c_cmd = self.node.create_client(CommandLong, f"{ns}/cmd/command")
        self.node.create_timer(1.0 / max(0.5, cfg.get("setpoint_hz", 5.0)), self._stream_setpoint)

        self._exec = MultiThreadedExecutor(num_threads=3)
        self._exec.add_node(self.node)
        self._thread = threading.Thread(target=self._exec.spin, name=f"{drone_id}-ros", daemon=True)
        self._thread.start()

    # ---- storage ----
    def _store(self, key, msg):
        with self._lock:
            self._d[key] = (msg, time.time())

    def _get(self, key):
        with self._lock:
            return self._d.get(key, (None, None))

    # ---- telemetry API ----
    def get_position(self) -> dict:
        fix, t_fix = self._get("fix")
        rel, _ = self._get("rel_alt")
        pose, t_pose = self._get("pose")
        p = pose.pose.position if pose else None
        return {
            "latitude": _f(fix.latitude) if fix else None,
            "longitude": _f(fix.longitude) if fix else None,
            "altitude_amsl_m": _f(fix.altitude) if fix else None,
            "altitude_m": _f(rel.data) if rel else None,
            "gps_fix": bool(fix and fix.status.status >= 0),
            "local": {"x": _f(p.x) if p else None, "y": _f(p.y) if p else None, "z": _f(p.z) if p else None},
            "age_s": None if t_fix is None else time.time() - t_fix,
        }

    def get_attitude(self) -> Optional[dict]:
        pose, t = self._get("pose")
        if pose is None:
            return None
        q = pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return {"quaternion": (q.x, q.y, q.z, q.w), "yaw_rad": yaw, "age_s": time.time() - t}

    def get_battery(self) -> dict:
        b, t = self._get("batt")
        if b is None:
            return {"percentage": None, "voltage": None, "current": None}
        pct = b.percentage
        pct = None if (not math.isfinite(pct) or pct < 0) else (pct * 100.0 if pct <= 1.0 else pct)
        return {"percentage": _f(pct), "voltage": _f(b.voltage), "current": _f(b.current)}

    def get_state(self) -> dict:
        s, t = self._get("state")
        alive = s is not None and (time.time() - t) < self.stale_s and bool(s.connected)
        return {"mavros_connected": alive, "fcu_connected": bool(s.connected) if s else False,
                "armed": bool(s.armed) if s else False, "mode": s.mode if s else None,
                "guided": bool(s.guided) if s else False}

    def get_velocity(self) -> dict:
        v, _ = self._get("vel")
        if v is None:
            return {"vx": None, "vy": None, "vz": None, "speed_h": None, "speed": None}
        l = v.twist.linear
        return {"vx": _f(l.x), "vy": _f(l.y), "vz": _f(l.z),
                "speed_h": math.hypot(l.x, l.y), "speed": math.sqrt(l.x ** 2 + l.y ** 2 + l.z ** 2)}

    def get_current_waypoint(self) -> dict:
        w, _ = self._get("wps")
        r, t_r = self._get("reached")
        return {"current_seq": int(w.current_seq) if w else None,
                "list_length": len(w.waypoints) if w else 0,
                "last_reached": int(r.wp_seq) if r else None,
                "last_reached_t": t_r}

    # ---- services ----
    def _call(self, client, req):
        if not client.wait_for_service(timeout_sec=self.svc_timeout):
            log.error("service %s unavailable", client.srv_name)
            return None
        fut = client.call_async(req)
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > self.svc_timeout:
                log.error("service %s timed out", client.srv_name)
                return None
            time.sleep(0.02)
        return fut.result()

    def set_mode(self, mode: str) -> bool:
        r = self._call(self._c_mode, SetMode.Request(base_mode=0, custom_mode=mode))
        return bool(r and r.mode_sent)

    def arm(self, value: bool) -> bool:
        r = self._call(self._c_arm, CommandBool.Request(value=value))
        return bool(r and r.success)

    def takeoff(self, altitude_m: float) -> bool:
        r = self._call(self._c_tol, CommandTOL.Request(min_pitch=0.0, yaw=0.0, latitude=0.0,
                                                       longitude=0.0, altitude=float(altitude_m)))
        return bool(r and r.success)

    def push_mission(self, points_latlon: Sequence[Tuple[float, float]], alt_m: float,
                     home_latlon: Tuple[float, float]) -> bool:
        """ArduPilot layout: item 0 = home placeholder, items 1..N = waypoints."""
        self._call(self._c_clear, WaypointClear.Request())
        items: List[Waypoint] = []
        h = Waypoint()
        h.frame, h.command, h.is_current, h.autocontinue = 0, 16, False, True
        h.x_lat, h.y_long, h.z_alt = float(home_latlon[0]), float(home_latlon[1]), 0.0
        items.append(h)
        for lat, lon in points_latlon:
            w = Waypoint()
            w.frame, w.command, w.is_current, w.autocontinue = 3, 16, False, True   # GLOBAL_REL_ALT, NAV_WAYPOINT
            w.param1, w.param2, w.param3, w.param4 = 0.0, 2.0, 0.0, 0.0
            w.x_lat, w.y_long, w.z_alt = float(lat), float(lon), float(alt_m)
            items.append(w)
        r = self._call(self._c_push, WaypointPush.Request(start_index=0, waypoints=items))
        return bool(r and r.success and r.wp_transfered == len(items))

    def set_current_waypoint(self, seq: int) -> bool:
        r = self._call(self._c_cur, WaypointSetCurrent.Request(wp_seq=int(seq)))
        return bool(r and r.success)

    def send_command_long(self, command: int, p1=0.0, p2=0.0, p3=0.0, p4=0.0, p5=0.0, p6=0.0, p7=0.0) -> bool:
        req = CommandLong.Request(broadcast=False, command=int(command), confirmation=0,
                                  param1=float(p1), param2=float(p2), param3=float(p3), param4=float(p4),
                                  param5=float(p5), param6=float(p6), param7=float(p7))
        r = self._call(self._c_cmd, req)
        return bool(r and r.success)

    # ---- guided goto (setpoint streamed until stop_goto) ----
    def start_goto(self, lat: float, lon: float, rel_alt_m: float):
        with self._goto_lock:
            self._goto = (lat, lon, rel_alt_m)

    def stop_goto(self):
        with self._goto_lock:
            self._goto = None

    def _stream_setpoint(self):
        with self._goto_lock:
            g = self._goto
        if g is None:
            return
        m = GlobalPositionTarget()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.coordinate_frame = GlobalPositionTarget.FRAME_GLOBAL_REL_ALT
        m.type_mask = 8 | 16 | 32 | 64 | 128 | 256 | 1024 | 2048   # position only
        m.latitude, m.longitude, m.altitude = float(g[0]), float(g[1]), float(g[2])
        self._sp_pub.publish(m)

    def shutdown(self):
        try:
            self._exec.shutdown()
            self.node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()