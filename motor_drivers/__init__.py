"""Python serial drivers for the Hiwonder bus servo and Emm42 V5.0.

本包通过串口控制两类执行器：
  - ``BusServo``   : Hiwonder 总线舵机 (半双工单线, 见 :mod:`motor_drivers.bus_servo`)
  - ``Emm42Driver``: Emm42 V5.0 闭环步进驱动器 (全双工 TTL, 见 :mod:`motor_drivers.emm42`)

两个驱动都需要先 ``initialize()`` 打开串口；
测试时可通过 ``serial_port=`` 参数注入 FakeSerial 而无需真串口。
"""

from .bus_servo import BusServo
from .emm42 import Direction, Emm42Driver, HomeMode

__all__ = ["BusServo", "Direction", "Emm42Driver", "HomeMode"]