# -*- coding: utf-8 -*-
import serial
import time
# 设置串口参数
port = '/dev/wheeltec_STP23'  # 串口号，根据实际情况修改
baudrate = 230400  # 波特率
def parse_data(data):
    # 检查起始符
    if data[0] != 0xAA:
        print("Invalid data packet: Start byte not found.")
        return []

    measurements = []
    
    # 从第11个字节开始解析每个测量数据点
    for i in range(10, 195, 15):
        if i + 15 < len(data):  # 确保不会越界
            #---------------距离数据distance------------------
            distance1 = data[i + 1]
            distance1=hex(distance1)   #将十进制整数转换为十六进制字符串表示
            interval1 = distance1[2:].upper()   #（去掉十六进制前面的0x）
            str1 = str(interval1)     #将 interval1 转为字符串
            distance2 = data[i]
            distance2 = hex(distance2)
            interval2 = distance2[2:].upper()
            str2= str(interval2)
            string1=str1+str2           #拼接高字节和低字节（用字符串来表示）
            distance =int(string1,16)    #将一个字符串 按照 16 进制的方式解析并转换为整数

            #------------------环境噪声noise--------------------
            noise1 = data[i + 3] 
            noise1 = hex(noise1)
            noise1_16 = noise1[2:].upper()
            noise1_str = str(noise1_16)
            noise2 = data[i + 2]
            noise2 = hex(noise2)
            noise2_16 = noise2[2:].upper()
            noise2_str = str(noise2_16)
            noise_str = noise1_str+noise2_str
            noise = int(noise_str,16)

            #-------------------------接收强度信息peak--------------------
            peak1 = data[i + 7]
            peak1 = hex(peak1)
            peak1_16 = peak1[2:].upper()
            peak1_str = str(peak1_16)
            peak2 = data[i + 6]
            peak2 = hex(peak2)
            peak2_16 = peak2[2:].upper()
            peak2_str = str(peak2_16)
            peak3 = data[i + 5]
            peak3 = hex(peak3)
            peak3_16 = peak3[2:].upper()
            peak3_str = str(peak3_16)
            peak4 = data[i + 4]
            peak4 = hex(peak4)
            peak4_16 = peak4[2:].upper()
            peak4_str = str(peak4_16)
            peak_str = peak1_str+peak2_str+peak3_str+peak4_str
            peak = int(peak_str,16)

            #-------------------------置信度confidence--------------------
            confidence = data[i + 8]

            #-------------------------积分次数intg---------------------
            intg1 = data[i + 12]
            intg1 = hex(intg1)
            intg1_16 = intg1[2:].upper()
            intg1_str = str(intg1_16)
            intg2 = data[i + 11]
            intg2 = hex(intg2)
            intg2_16 = intg2[2:].upper()
            intg2_str = str(intg2_16)
            intg3 = data[i + 10]
            intg3 = hex(intg3)
            intg3_16 = intg3[2:].upper()
            intg3_str = str(intg3_16)
            intg4 = data[i + 9]
            intg4 = hex(intg4)
            intg4_16 = intg4[2:].upper()
            intg4_str = str(intg4_16)
            intg_str = intg1_str+intg2_str+intg3_str+intg4_str
            intg = int(intg_str, 16)

            #--------------------------温度表征值reftof-----------------
            reftof1 = data[i + 14]
            reftof1 = hex(reftof1)
            reftof1_16 = reftof1[2:].upper()
            reftof1_str = str(reftof1_16)
            reftof2 = data[i + 13]
            reftof2 = hex(reftof2)
            reftof2_16 = reftof2[2:].upper()
            reftof2_str = str(reftof2_16)
            reftof_str = reftof1_str+reftof2_str
            reftof = int(reftof_str, 16)
            measurements.append((distance, noise, peak, confidence, intg, reftof))
    
    return measurements

def read_serial_data():
    with serial.Serial(port, baudrate, timeout=5) as ser:
        time.sleep(2)  # 等待设备准备
        while True:
            if ser.in_waiting > 0:
            	raw_data = b''
            	while len(raw_data) < 195:
                	# 读取串口中所有可用的数据
                	raw_data += ser.read(ser.in_waiting)
                	time.sleep(0.01)  # 可以加一些延时，避免CPU占用过高
            	# 确保读取到完整的195字节
            	if len(raw_data) >= 195:
                	data = list(bytearray(raw_data[:195]))  # 取前195字节，防止溢出
                	measurements = parse_data(data)
                if measurements:
                    for idx, (distance, noise, peak, confidence, intg, reftof) in enumerate(measurements):
                        print("测量点 {}: 距离数据Distance = {} mm, 环境噪声Noise = {}, 接收强度信息Peak = {}, 置信度Confidence = {}, 积分次数Intg = {}, 温度表征值Reftof = {}".format(idx + 1, distance, noise, peak, confidence, intg, reftof))
			print("------------------------------")

if __name__ == "__main__":
    read_serial_data()


