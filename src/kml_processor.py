from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
E2 = WGS84_F * (2 - WGS84_F)

LatLon = Tuple[float, float]
XY = Tuple[float, float]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


class LocalFrame:
    """East-North local tangent plane around (lat0, lon0)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        s = math.sin(math.radians(lat0))
        w = 1 - E2 * s * s
        n = WGS84_A / math.sqrt(w)
        self._m = WGS84_A * (1 - E2) / (w ** 1.5)
        self._kx = n * math.cos(math.radians(lat0))

    def to_local(self, lat: float, lon: float) -> XY:
        return (math.radians(lon - self.lon0) * self._kx, math.radians(lat - self.lat0) * self._m)

    def to_geo(self, x: float, y: float) -> LatLon:
        return (self.lat0 + math.degrees(y / self._m), self.lon0 + math.degrees(x / self._kx))

    def distance_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        x1, y1 = self.to_local(lat1, lon1)
        x2, y2 = self.to_local(lat2, lon2)
        return math.hypot(x2 - x1, y2 - y1)


@dataclass
class MissionArea:
    frame: LocalFrame
    rings: List[List[XY]]            # rings[0] = outer boundary, rest = holes (local ENU metres)
    outer_geo: List[LatLon]


def _tag(e) -> str:
    return e.tag.rsplit("}", 1)[-1]


def _ring(text: str) -> List[LatLon]:
    pts: List[LatLon] = []
    for tok in text.split():
        parts = tok.split(",")
        if len(parts) < 2:
            continue
        pts.append((float(parts[1]), float(parts[0])))  # KML is lon,lat[,alt]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    return pts


def _shoelace(ring: List[LatLon]) -> float:
    a = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i][1], ring[i][0]
        x2, y2 = ring[(i + 1) % len(ring)][1], ring[(i + 1) % len(ring)][0]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def parse_kml_polygon(path: str) -> Tuple[List[LatLon], List[List[LatLon]]]:
    root = ET.parse(path).getroot()
    best = None
    for poly in (e for e in root.iter() if _tag(e) == "Polygon"):
        outer, holes = None, []
        for child in poly:
            t = _tag(child)
            if t not in ("outerBoundaryIs", "innerBoundaryIs"):
                continue
            coords = [c for c in child.iter() if _tag(c) == "coordinates"]
            if not coords or not coords[0].text:
                continue
            ring = _ring(coords[0].text)
            if len(ring) < 3:
                continue
            if t == "outerBoundaryIs":
                outer = ring
            else:
                holes.append(ring)
        if outer and (best is None or _shoelace(outer) > _shoelace(best[0])):
            best = (outer, holes)
    if best is None:
        raise ValueError(f"no valid Polygon found in {path}")
    return best


def load_area(path: str) -> MissionArea:
    outer, holes = parse_kml_polygon(path)
    lat0 = sum(p[0] for p in outer) / len(outer)
    lon0 = sum(p[1] for p in outer) / len(outer)
    frame = LocalFrame(lat0, lon0)
    rings = [[frame.to_local(la, lo) for la, lo in outer]]
    rings += [[frame.to_local(la, lo) for la, lo in h] for h in holes]
    return MissionArea(frame=frame, rings=rings, outer_geo=outer)