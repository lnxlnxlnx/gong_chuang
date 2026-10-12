import serial
import time
port = 'COM3'
baudrate = 230400

def parse_data(data):
    if data[0] != 0xAA:
        return None
    measurements = []
    for i in range(10, 195, 15):
        if i + 15 < len(data):
            distance = (data[i + 1] << 8) | data[i]
            noise = (data[i + 3] << 8) | data[i + 2]
            peak = (data[i + 7] << 24) | (data[i + 6] << 16) | (data[i + 5] << 8) | data[i + 4]
            confidence = data[i + 8]
            intg = (data[i + 12] << 24) | (data[i + 11] << 16) | (data[i + 10] << 8) | data[i + 9]
            reftof = (data[i + 14] << 8) | data[i + 13]
            measurements.append((distance, noise, peak, confidence, intg, reftof))
    return measurements

def read_serial_data():
    with serial.Serial(port, baudrate, timeout=3) as ser:
        time.sleep(2)
        buffer = bytearray()
        while True:
            if ser.in_waiting > 0:
                buffer.extend(ser.read(ser.in_waiting))
                while len(buffer) > 0 and buffer[0] != 0xAA:
                    buffer.pop(0)
                if len(buffer) >= 195:
                    data = list(buffer[:195])
                    measurements = parse_data(data)
                    if measurements:
                        for idx, (distance, noise, peak, confidence, intg, reftof) in enumerate(measurements):
                            print(
                                f"测量点 {idx + 1}: 距离数据Distance = {distance} mm, 环境噪声Noise = {noise}, 接收强度信息Peak = {peak}, 置信度Confidence = {confidence}, 积分次数Intg = {intg}, 温度表征值Reftof = {reftof}")
                        print("------------")
                        buffer = buffer[195:]
                    else:
                        buffer.pop(0)
                if len(buffer) > 300:
                    buffer = buffer[-100:]


if __name__ == "__main__":
    read_serial_data()