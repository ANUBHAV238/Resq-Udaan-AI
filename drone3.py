import sys

from src.drone_manager import run_drone

DRONE_ID = "drone3"
MAVROS_NAMESPACE = "/drone3/mavros"

if __name__ == "__main__":
    sys.exit(run_drone(DRONE_ID, MAVROS_NAMESPACE, "config/drone3.yaml", "config/mission.yaml"))