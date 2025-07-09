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
        self.position_xy = None         # (x, y) in meters global frame
        self.position_xy_body = None    # (x, y) in meters body frame
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

        self.running = True
        self.thread.start()

        # Wait for origin to be set (after averaging)
        while self.running and self.origin is None:
            time.sleep(0.1)

        if self.origin:
            logging.info(f"Origin set at {self.origin.latitude:.6f}, {self.origin.longitude:.6f}")
            return True
        else:
            logging.warning("Could not establish GPS origin.")
            return False

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

            # If origin not yet set, buffer points
            if self.origin is None:
                self.origin_buffer.append(new_point)
                if len(self.origin_buffer) >= 5:
                    # Compute average lat/lon
                    avg_lat = sum(p.latitude for p in self.origin_buffer) / len(self.origin_buffer)
                    avg_lon = sum(p.longitude for p in self.origin_buffer) / len(self.origin_buffer)
                    self.origin = Point(avg_lat, avg_lon)
                    self.position_xy = (0.0, 0.0)
                return

            # Once origin is set, convert to local XY
            # North-South (y)
            lat_dist = geodesic(
                (self.origin.latitude, self.origin.longitude),
                (new_point.latitude,    self.origin.longitude)
            ).meters
            y = lat_dist if new_point.latitude >= self.origin.latitude else -lat_dist
            # East-West (x)
            lon_dist = geodesic(
                (self.origin.latitude, self.origin.longitude),
                (self.origin.latitude, new_point.longitude)
            ).meters
            x = lon_dist if new_point.longitude >= self.origin.longitude else -lon_dist
            self.position_xy = (x, y)

    def get_position_xy(self):
        """
        Retrieve the latest local XY coordinates.

        Returns:
            tuple: (x, y) in meters, or None if not yet fixed.
        """
        with self.lock:
            return self.position_xy
        
    def get_position_xy_body(self, yaw_rad):
        """
        Rotate the global XY into the body frame using yaw.
        """
        with self.lock:
            if self.position_xy is not None:
                x, y = self.position_xy
                body_x = x * np.cos(yaw_rad) + y * np.sin(yaw_rad)
                body_y = -x * np.sin(yaw_rad) + y * np.cos(yaw_rad)
                self.position_xy_body = (body_x, body_y)
            return self.position_xy_body

    def stop(self):
        """
        Stop the reader thread.
        """
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        logging.info("GPSReader stopped")
