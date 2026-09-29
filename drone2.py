import sys

from src.drone_manager import run_drone

DRONE_ID = "drone2"
MAVROS_NAMESPACE = "/drone2/mavros"

if __name__ == "__main__":
    sys.exit(run_drone(DRONE_ID, MAVROS_NAMESPACE, "config/drone2.yaml", "config/mission.yaml"))