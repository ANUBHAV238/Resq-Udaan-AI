from __future__ import annotations

import logging
import math
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .kml_processor import LocalFrame

log = logging.getLogger("detector")


# ---------------- frame source ----------------
class FrameSource:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._frame = None
        self._ts = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._run, name="camera", daemon=True).start()

    def stop(self):
        self._stop.set()

    def _open(self):
        src = self.cfg.get("source", 0)
        cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER) if isinstance(src, str) and "!" in src \
            else cv2.VideoCapture(src)
        if isinstance(src, int):
            if self.cfg.get("width"):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["width"])
            if self.cfg.get("height"):
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["height"])
        return cap

    def _run(self):
        cap = None
        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                cap = self._open()
                if not cap.isOpened():
                    log.error("camera open failed, retrying")
                    time.sleep(2.0)
                    continue
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                time.sleep(0.5)
                continue
            with self._lock:
                self._frame, self._ts = frame, time.time()
        if cap is not None:
            cap.release()

    def read(self, max_age_s: float = 1.0) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None or time.time() - self._ts > max_age_s:
                return None
            return self._frame.copy()


# ---------------- detection backends ----------------
class HogBackend:
    def __init__(self, cfg: dict):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.thr = cfg.get("conf_threshold", 0.4)

    def infer(self, frame) -> List[dict]:
        h, w = frame.shape[:2]
        scale = 640.0 / w if w > 640 else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame
        rects, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        out = []
        for (x, y, rw, rh), wt in zip(rects, np.asarray(weights).reshape(-1)):
            conf = float(min(1.0, max(0.0, wt / 2.5)))
            if conf >= self.thr:
                out.append({"class": "person", "confidence": conf,
                            "bbox": [x / scale, y / scale, (x + rw) / scale, (y + rh) / scale]})
        return out


class YoloOnnxBackend:
    """YOLOv8-style ONNX export via cv2.dnn. Output layout [1, 4+nc, N] or [1, N, 4+nc]."""

    def __init__(self, cfg: dict):
        self.net = cv2.dnn.readNetFromONNX(cfg["model_path"])
        self.size = int(cfg.get("input_size", 640))
        self.conf = cfg.get("conf_threshold", 0.4)
        self.nms = cfg.get("nms_threshold", 0.45)
        self.class_map = {int(k): v for k, v in cfg.get("class_map", {0: "person"}).items()}

    def infer(self, frame) -> List[dict]:
        h, w = frame.shape[:2]
        r = min(self.size / h, self.size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (self.size - nw) / 2.0, (self.size - nh) / 2.0
        img = cv2.resize(frame, (nw, nh))
        img = cv2.copyMakeBorder(img, int(round(dh - 0.1)), int(round(dh + 0.1)),
                                 int(round(dw - 0.1)), int(round(dw + 0.1)),
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()[0]
        if out.shape[0] < out.shape[1]:
            out = out.T
        boxes, scores, cls_ids = [], [], []
        for row in out:
            cs = row[4:]
            cid = int(np.argmax(cs))
            sc = float(cs[cid])
            if sc < self.conf or cid not in self.class_map:
                continue
            cx, cy, bw, bh = row[:4]
            x1 = (cx - bw / 2 - dw) / r
            y1 = (cy - bh / 2 - dh) / r
            boxes.append([x1, y1, bw / r, bh / r])
            scores.append(sc)
            cls_ids.append(cid)
        res = []
        if boxes:
            for i in np.array(cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.nms)).reshape(-1):
                x, y, bw, bh = boxes[i]
                res.append({"class": self.class_map[cls_ids[i]], "confidence": scores[i],
                            "bbox": [max(0.0, x), max(0.0, y), min(w - 1.0, x + bw), min(h - 1.0, y + bh)]})
        return res


# ---------------- geo-referencing ----------------
def quat_to_matrix(q: Tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Georeferencer:
    """Pixel -> ground point (lat/lon). Body frame FLU, world ENU (MAVROS convention)."""

    def __init__(self, cam_cfg: dict):
        self.cfg = cam_cfg
        self.tilt = math.radians(cam_cfg.get("tilt_forward_deg", 0.0))
        self.ref = cam_cfg.get("ref_point", "center")
        self.ground_rel = cam_cfg.get("ground_elevation_rel_m", 0.0)
        # camera(optical: x right, y down, z fwd) -> body FLU for a nadir camera, image-top toward vehicle nose
        r_nadir = np.array([[0, -1, 0],
                            [-1, 0, 0],
                            [0, 0, -1]], dtype=float)
        c, s = math.cos(-self.tilt), math.sin(-self.tilt)   # forward tilt = rotation about body-left (y) axis
        ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        self.r_body_cam = ry @ r_nadir

    def _intrinsics(self, w: int, h: int):
        cfg = self.cfg
        if cfg.get("camera_matrix"):
            k = np.array(cfg["camera_matrix"], dtype=np.float64)
            cs = cfg.get("calib_size")
            if cs:
                k = k.copy()
                k[0, :] *= w / cs[0]
                k[1, :] *= h / cs[1]
        else:
            fx = (w / 2.0) / math.tan(math.radians(cfg.get("hfov_deg", 90.0)) / 2.0)
            k = np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])
        dist = np.array(cfg["dist_coeffs"], dtype=np.float64) if cfg.get("dist_coeffs") else None
        return k, dist

    def locate(self, bbox, frame_size, lat: float, lon: float, rel_alt_m: float,
               quaternion, max_tilt_deg: float = 25.0) -> Optional[dict]:
        w, h = frame_size
        x1, y1, x2, y2 = bbox
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0 if self.ref == "center" else y2
        k, dist = self._intrinsics(w, h)
        n = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float32), k, dist)[0, 0]
        d_cam = np.array([n[0], n[1], 1.0])
        d_cam /= np.linalg.norm(d_cam)
        ray = quat_to_matrix(quaternion) @ (self.r_body_cam @ d_cam)      # ENU
        if ray[2] >= -1e-3:
            return None                                                    # ray does not hit the ground
        if math.degrees(math.acos(max(-1.0, min(1.0, -ray[2])))) > 60.0 + max_tilt_deg:
            return None
        height = rel_alt_m - self.ground_rel
        if height <= 0.5:
            return None
        t = height / -ray[2]
        east, north = ray[0] * t, ray[1] * t
        glat, glon = LocalFrame(lat, lon).to_geo(east, north)
        return {"latitude": glat, "longitude": glon, "altitude": self.ground_rel, "range_m": float(t)}


# ---------------- public interface ----------------
class Detector:
    def __init__(self, drone_id: str, det_cfg: dict, cam_cfg: dict):
        self.drone_id = drone_id
        self.cfg = det_cfg
        self.max_tilt = det_cfg.get("max_tilt_deg", 25.0)
        self.geo = Georeferencer(cam_cfg)
        self._n = 0
        backend = det_cfg.get("backend", "onnx")
        if backend == "onnx":
            try:
                self.backend = YoloOnnxBackend(det_cfg)
                log.info("detector: ONNX %s", det_cfg["model_path"])
            except Exception as e:
                log.error("ONNX model load failed (%s); falling back to HOG people detector", e)
                self.backend = HogBackend(det_cfg)
        else:
            self.backend = HogBackend(det_cfg)

    def detect(self, frame, pose: Optional[dict] = None) -> List[dict]:
        """pose: {latitude, longitude, altitude_m (rel), quaternion}. Returns spec-format detections."""
        h, w = frame.shape[:2]
        out = []
        for d in self.backend.infer(frame):
            self._n += 1
            det = {
                "detection_id": f"det_{self.drone_id}_{self._n:05d}",
                "drone_id": self.drone_id,
                "class": d["class"],
                "confidence": round(float(d["confidence"]), 4),
                "bbox": [int(round(v)) for v in d["bbox"]],
                "frame_size": [w, h],
                "location": None,
            }
            if pose:
                loc = self.geo.locate(d["bbox"], (w, h), pose["latitude"], pose["longitude"],
                                      pose["altitude_m"], pose["quaternion"], self.max_tilt)
                if loc is None:
                    continue                  # cannot geo-locate reliably -> do not publish a guess
                det["location"] = loc
            out.append(det)
        return out