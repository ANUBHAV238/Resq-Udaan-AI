from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import paho.mqtt.client as mqtt

from .messages import Topics, dumps

log = logging.getLogger("mqtt")

KIND_QOS = {"status": 1, "telemetry": 0, "detection": 1, "delivery": 1, "coordination": 1, "mission": 1}
KIND_RETAIN = {"status": True, "delivery": True}


def kind_of(topic: str) -> str:
    if topic == Topics.COORDINATION:
        return "coordination"
    if topic == Topics.MISSION:
        return "mission"
    return topic.rsplit("/", 1)[-1]


def _ok(rc) -> bool:
    if hasattr(rc, "is_failure"):
        return not rc.is_failure
    return rc == 0


class MqttManager:
    def __init__(self, drone_id: str, cfg: dict, lwt: Optional[Tuple[str, dict]] = None):
        self.id = drone_id
        self.cfg = cfg
        self._stale = cfg.get("stale_after_s", {})
        self._subs: List[Tuple[str, Callable]] = []
        self._q: "queue.Queue[Tuple[str, bytes]]" = queue.Queue(maxsize=2000)
        self._connected = threading.Event()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._stop = threading.Event()
        self.stats = {"rx": 0, "tx": 0, "malformed": 0, "stale": 0, "queue_dropped": 0, "reconnects": 0}

        client_id = cfg.get("client_id") or f"{drone_id}-{os.getpid()}"
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id,
                                      protocol=mqtt.MQTTv311)
        except AttributeError:
            self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        if cfg.get("username"):
            self.client.username_pw_set(cfg["username"], cfg.get("password"))
        if cfg.get("tls"):
            self.client.tls_set()
        self.client.reconnect_delay_set(cfg.get("reconnect_min_s", 1), cfg.get("reconnect_max_s", 15))
        self.client.max_queued_messages_set(cfg.get("max_queued", 500))
        if lwt:
            topic, payload = lwt
            payload = dict(payload)
            payload["lwt"] = True
            payload["timestamp"] = time.time()
            self.client.will_set(topic, dumps(payload), qos=1, retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def start(self):
        self.client.connect_async(self.cfg.get("host", "127.0.0.1"), int(self.cfg.get("port", 1883)),
                                  int(self.cfg.get("keepalive", 15)))
        self.client.loop_start()
        threading.Thread(target=self._dispatch_loop, name="mqtt-dispatch", daemon=True).start()

    def stop(self):
        self._stop.set()
        try:
            self.client.disconnect()
        finally:
            self.client.loop_stop()

    def subscribe(self, topic: str, callback: Callable[[str, dict], None]):
        self._subs.append((topic, callback))
        if self.connected:
            self.client.subscribe(topic, qos=1)

    def publish(self, topic: str, payload: dict, kind: Optional[str] = None,
                retain: Optional[bool] = None) -> bool:
        kind = kind or kind_of(topic)
        qos = KIND_QOS.get(kind, 1)
        if not self.connected and qos == 0:
            return False
        msg = dict(payload)
        msg.setdefault("timestamp", time.time())
        msg.setdefault("sender", self.id)
        with self._seq_lock:
            self._seq += 1
            msg["seq"] = self._seq
        try:
            info = self.client.publish(topic, dumps(msg), qos=qos,
                                       retain=KIND_RETAIN.get(kind, False) if retain is None else retain)
        except Exception:
            log.exception("publish failed on %s", topic)
            return False
        self.stats["tx"] += 1
        return info.rc == mqtt.MQTT_ERR_SUCCESS

    # ---- callbacks (paho thread) ----
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if not _ok(rc):
            log.error("MQTT connect refused: %s", rc)
            return
        log.info("MQTT connected")
        self._connected.set()
        for topic, _ in self._subs:
            client.subscribe(topic, qos=1)

    def _on_disconnect(self, client, userdata, *args):
        rc = args[1] if len(args) >= 2 else (args[0] if args else 0)
        self._connected.clear()
        self.stats["reconnects"] += 1
        log.warning("MQTT disconnected (%s); auto-reconnect active", rc)

    def _on_message(self, client, userdata, msg):
        try:
            self._q.put_nowait((msg.topic, bytes(msg.payload)))
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait((msg.topic, bytes(msg.payload)))
            except queue.Empty:
                pass
            self.stats["queue_dropped"] += 1

    # ---- dispatch thread ----
    def _dispatch_loop(self):
        while not self._stop.is_set():
            try:
                topic, raw = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            self.stats["rx"] += 1
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                self.stats["malformed"] += 1
                log.warning("malformed JSON on %s dropped", topic)
                continue
            if not isinstance(data, dict):
                self.stats["malformed"] += 1
                continue
            ts = data.get("timestamp")
            if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
                self.stats["malformed"] += 1
                log.warning("message without valid timestamp on %s dropped", topic)
                continue
            limit = self._stale.get(kind_of(topic))
            if limit and not data.get("lwt") and time.time() - ts > limit:
                self.stats["stale"] += 1
                continue
            for sub, cb in list(self._subs):
                if mqtt.topic_matches_sub(sub, topic):
                    try:
                        cb(topic, data)
                    except Exception:
                        log.exception("handler failed for %s", topic)