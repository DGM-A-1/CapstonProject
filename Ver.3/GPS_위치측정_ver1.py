#!/usr/bin/env python3
"""
GPS.py
----------------------------------------------------------------
Reads DGPS-corrected position information via gpsd (NTRIP) and converts
latitude/longitude into a local XY coordinate system.
The origin is set as the average of the first five valid measurements.
"""

import threading
import time
import logging

from matplotlib import pyplot as plt
from geopy import Point
from geopy.distance import geodesic

import gpsd
import numpy as np

class GPSReader:
    def __init__(self, host='localhost', port=2947):
        """
        Initialize the GPSReader for DGPS via gpsd.

        Args:
            host (str): gpsd server host (default: localhost)
            port (int): gpsd server port (default: 2947)
        """
        self.host = host
        self.port = port
        self.lock = threading.Lock()
        self.position = None            # geopy Point(lat, lon)
        self.position_xy = None         # (x, y) in meters global frame => [N, E]   
        self.position_xy_body = None    # (x, y) in meters body frame => [차 앞쪽, 차 왼쪽]
        self.origin = None              # initial Point after averaging
        self.origin_buffer = []         # buffer for first N points
        self.running = False
        self.thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self):
        """
        Connect to gpsd and start the reader thread.
        Blocks until the origin is established.

        Returns:
            bool: True if origin established, False otherwise.
        """
        try:
            gpsd.connect(host=self.host, port=self.port)
            logging.info(f"Connected to gpsd at {self.host}:{self.port}")
        except Exception as e:
            logging.error(f"Failed to connect to gpsd: {e}")
            return False

        self.origin = Point(35.8875834, 128.6117572)
        self.position_xy = (0.0, 0.0)  # origin in
        self.running = True
        self.thread.start()
        return True

    def _read_loop(self):
        """
        Continuously read position reports from gpsd.
        """
        while self.running:
            try:
                report = gpsd.get_current()
                lat = getattr(report, 'lat', None)
                lon = getattr(report, 'lon', None)
                if lat is not None and lon is not None and lat != 0 and lon != 0:
                    self._update_position(Point(lat, lon))
            except Exception as e:
                logging.error(f"GPS read error: {e}")
            time.sleep(0.2)

    def _update_position(self, new_point: Point):
        """
        Thread-safe update of position and conversion to local XY.
        For the first five valid measurements, buffer and average to set origin.
        """
        with self.lock:
            self.position = new_point

            # Once origin is set, convert to local XY
            # North-South (x)
            lat_dist = geodesic(
                (self.origin.latitude, self.origin.longitude),
                (new_point.latitude,    self.origin.longitude)
            ).meters
            x = lat_dist if new_point.latitude >= self.origin.latitude else -lat_dist
            # East-West (y)
            lon_dist = geodesic(
                (self.origin.latitude, self.origin.longitude),
                (self.origin.latitude, new_point.longitude)
            ).meters
            y = lon_dist if new_point.longitude >= self.origin.longitude else -lon_dist
            self.position_xy = (x, y)

    """
    Retrieve the latest local XY coordinates.
    Returns:
    tuple: (x, y) in meters, or None if not yet fixed.
    """
    def get_position_xy(self):
        with self.lock:
            return self.position_xy
    
    def get_position(self):
        with self.lock:
            return self.origin.latitude, self.origin.longitude
    """
    Rotate the global XY into the body frame using yaw.
    """
    def get_position_xy_body(self, yaw_rad):
        with self.lock:
            if self.position_xy is not None:
                x, y = self.position_xy
                body_x = x * np.cos(yaw_rad) + y * np.sin(yaw_rad)
                body_y = -x * np.sin(yaw_rad) + y * np.cos(yaw_rad)
                self.position_xy_body = (body_x, body_y)
            return self.position_xy_body

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        logging.info("GPSReader stopped")
        
if __name__ == "__main__":
    # GPS 시작 (origin 설정)
    gps_reader = GPSReader()
    if not gps_reader.start():
        print("GPS origin ???? ????, ???????? ????")
        exit(1)

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("????(0,0) ???? ???? ???? ????")

    xs, ys = [], []
    try:
        while True:
            pos = gps_reader.get_position_xy()
            if pos is not None:
                x, y = pos
                xs.append(x)
                ys.append(y)

                ax.clear()
                ax.scatter(xs, ys, s=10)
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_title("????(0, 0) ???? ???? ???? ????")
                ax.grid(True)

                plt.pause(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        gps_reader.stop()
        plt.ioff()