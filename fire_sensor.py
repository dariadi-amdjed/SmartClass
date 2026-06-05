
"""
Fire Detection System Module
Handles Arduino connection, fire detection, and device control
"""

# Python 3.11 Compatibility: Fix for PyFirmata using deprecated inspect.getargspec
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import threading
import time
from pyfirmata import Arduino, util

class FireSensorController:
    def __init__(self, port='COM3'):
        """
        Initialize fire detection system
        port: COM port to connect to the Arduino
        """
        self.port = port
        self.board = None
        self.flame_sensor = None
        self.servo = None
        self.buzzer = None
        self.motor_in1 = None
        self.motor_in2 = None
        self.iterator = None
        
        self.is_running = False
        self.fire_detected_callback = None
        self.fire_cleared_callback = None
        self.monitor_thread = None
        
        self.SERVO_CLOSED = 90
        self.SERVO_OPEN = 40
        
        self.fire_state = False
        
    def connect(self):

        try:
            print(f"🔌 Trying To Connect To Arduino{self.port}...")
            self.board = Arduino(self.port)
            
            self.iterator = util.Iterator(self.board)
            self.iterator.start()
            
            self.flame_sensor = self.board.get_pin('d:8:i')
            self.servo = self.board.get_pin('d:9:s')
            self.buzzer = self.board.get_pin('d:10:o')
            self.motor_in1 = self.board.get_pin('d:6:o')
            self.motor_in2 = self.board.get_pin('d:7:o')
            
            time.sleep(1)
            if self.servo:
                self.servo.write(self.SERVO_CLOSED)
            time.sleep(1)
            
            print("✅ Arduino Connected Successfully")
            print("🔥 Fire Detection System Ready")
            return True
            
        except Exception as e:
            print(f"❌ Failed To Connect To Arduino: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        try:
            if self.is_running:
                self.stop_monitoring()
            if self.board:
                self.board.exit()
                self.board = None
            print("✅ Arduino Unconnected Successfully")
        except Exception as e:
            print(f"❌ Error Disconnecting From Arduino: {e}")
    
    def set_fire_detected_callback(self, callback):
        """Set a callback function when fire is detected
        callback: Function called when fire is detected
        """
        self.fire_detected_callback = callback
    
    def set_fire_cleared_callback(self, callback):
        """Set a callback function when fire is cleared
        callback: Function called when fire is cleared
        """
        self.fire_cleared_callback = callback
    
    def _monitor_loop(self):
        """Monitoring loop - reads the sensor state continuously"""
        while self.is_running:
            try:
                if self.flame_sensor is None:
                    time.sleep(0.1)
                    continue
                
                flame_state = self.flame_sensor.read()
                
                if flame_state is not None:
                    if flame_state == 0:
                        if not self.fire_state:
                            self.fire_state = True
                            self._activate_fire_response()
                            if self.fire_detected_callback:
                                self.fire_detected_callback()
                    else:
                        if self.fire_state:
                            self.fire_state = False
                            self._deactivate_fire_response()
                            if self.fire_cleared_callback:
                                self.fire_cleared_callback()
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error In Monitoring Loop: {e}")
                time.sleep(0.5)
    
    def _activate_fire_response(self):
        """Activate response when fire is detected"""
        try:
            print("🔥🚨 Fire detected! Activating responses...")
            if self.buzzer:
                self.buzzer.write(1)
            if self.servo:
                self.servo.write(self.SERVO_OPEN)
                time.sleep(0.2)  # wait for servo movement
            if self.motor_in1:
                self.motor_in1.write(1)
            if self.motor_in2:
                self.motor_in2.write(0)
                
        except Exception as e:
            print(f"❌ Error activating fire response: {e}")
    
    def _deactivate_fire_response(self):
        """Deactivate response when fire is cleared"""
        try:
            print("✅ Fire cleared. Stopping responses...")
            if self.buzzer:
                self.buzzer.write(0)
            if self.servo:
                self.servo.write(self.SERVO_CLOSED)
                time.sleep(0.2)  # wait for servo movement
            if self.motor_in1:
                self.motor_in1.write(0)
            if self.motor_in2:
                self.motor_in2.write(0)
                
        except Exception as e:
            print(f"❌ Error deactivating fire response: {e}")
    
    def start_monitoring(self):
        """Start fire sensor monitoring"""
        if self.board is None:
            print("❌ Arduino is not connected. Please connect first.")
            return False
        
        if self.is_running:
            print("⚠️ Monitoring is already running")
            return False
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🟢 Fire sensor monitoring started")
        return True
    
    def stop_monitoring(self):
        """Stop fire sensor monitoring"""
        if not self.is_running:
            print("⚠️ Monitoring is not active")
            return False
        
        self.is_running = False
        if self.fire_state:
            self._deactivate_fire_response()
        
        # Wait for monitoring thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        print("🔴 Fire sensor monitoring stopped")
        return True
    
    def get_fire_state(self):
        """Get current fire status (True = fire, False = no fire)"""
        return self.fire_state
    
    def manual_activate(self):
        """Trigger responses manually (testing)"""
        self._activate_fire_response()
        self.fire_state = True
    
    def manual_deactivate(self):
        """Stop responses manually (testing)"""
        self._deactivate_fire_response()
        self.fire_state = False


# Allow a global controller instance
_fire_controller = None

def get_fire_controller():
    """Get the current fire controller (single instance only)"""
    global _fire_controller
    if _fire_controller is None:
        _fire_controller = FireSensorController()
    return _fire_controller
