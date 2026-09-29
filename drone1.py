import sys

from src.drone_manager import run_drone

DRONE_ID = "drone1"
MAVROS_NAMESPACE = "/drone1/mavros"

if __name__ == "__main__":
    sys.exit(run_drone(DRONE_ID, MAVROS_NAMESPACE, "config/drone1.yaml", "config/mission.yaml"))