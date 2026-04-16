#!/usr/bin/env python3
from SerialHandler import SerialHandler
from XbusPacket import XbusPacket
from DataPacketParser import DataPacketParser, XsDataPacket
import time
from datetime import datetime
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os

# Класс для расчёта вибрации (с вычитанием гравитации)
class VibrationAnalyzer:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.acc_x = deque(maxlen=window_size)
        self.acc_y = deque(maxlen=window_size)
        self.acc_z = deque(maxlen=window_size)
        # Для сохранения всей истории
        self.vibration_log = []  # (timestamp, total_vibration, rms_x, rms_y, rms_z)
        self.start_time = time.time()
    
    def add_data(self, x, y, z):
        self.acc_x.append(x)
        self.acc_y.append(y)
        self.acc_z.append(z)
    
    def get_vibration_rms(self):
        if len(self.acc_x) < self.window_size:
            return None
        
        mean_x = np.mean(self.acc_x)
        mean_y = np.mean(self.acc_y)
        mean_z = np.mean(self.acc_z)
        
        vib_x = np.array(self.acc_x) - mean_x
        vib_y = np.array(self.acc_y) - mean_y
        vib_z = np.array(self.acc_z) - mean_z
        
        rms_x = np.sqrt(np.mean(vib_x**2))
        rms_y = np.sqrt(np.mean(vib_y**2))
        rms_z = np.sqrt(np.mean(vib_z**2))
        total = np.sqrt(rms_x**2 + rms_y**2 + rms_z**2)
        
        # Сохраняем в лог
        elapsed_time = time.time() - self.start_time
        self.vibration_log.append((elapsed_time, total, rms_x, rms_y, rms_z))
        
        return rms_x, rms_y, rms_z, total
    
    def save_log(self, directory):
        """Сохраняет лог вибрации в CSV файл"""
        filename = os.path.join(directory, 'vibration_log.csv')
        with open(filename, 'w') as f:
            f.write("time_seconds,total_vibration,rms_x,rms_y,rms_z\n")
            for log in self.vibration_log:
                f.write(f"{log[0]:.3f},{log[1]:.6f},{log[2]:.6f},{log[3]:.6f},{log[4]:.6f}\n")
        print(f"\n📁 Лог вибрации сохранён: {filename}")
        return filename
    
    def plot_vibration(self, directory):
        """Строит график вибрации от времени"""
        if len(self.vibration_log) == 0:
            print("Нет данных для построения графика")
            return None
        
        times = [log[0] for log in self.vibration_log]
        total_vib = [log[1] for log in self.vibration_log]
        rms_x = [log[2] for log in self.vibration_log]
        rms_y = [log[3] for log in self.vibration_log]
        rms_z = [log[4] for log in self.vibration_log]
        
        plt.figure(figsize=(14, 10))
        
        # График общей вибрации
        plt.subplot(2, 1, 1)
        plt.plot(times, total_vib, 'b-', linewidth=1.5)
        plt.xlabel('Время (секунды)')
        plt.ylabel('Вибрация TOTAL (m/s²)')
        plt.title('Общая вибрация от времени')
        plt.grid(True, alpha=0.3)
        
        # График вибрации по осям
        plt.subplot(2, 1, 2)
        plt.plot(times, rms_x, 'r-', label='RMS X', linewidth=1)
        plt.plot(times, rms_y, 'g-', label='RMS Y', linewidth=1)
        plt.plot(times, rms_z, 'b-', label='RMS Z', linewidth=1)
        plt.xlabel('Время (секунды)')
        plt.ylabel('Вибрация по осям (m/s²)')
        plt.title('Вибрация по осям X, Y, Z')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(directory, 'vibration_plot.png')
        plt.savefig(filename, dpi=150)
        print(f"📊 График вибрации сохранён: {filename}")
        plt.close()  # Закрываем график, чтобы не открывался
        return filename


def on_live_data_available(packet, vib_analyzer=None):
    xbus_data = XsDataPacket() 
    DataPacketParser.parse_data_packet(packet, xbus_data)

    # Расчёт вибрации
    if vib_analyzer and xbus_data.accAvailable:
        vib_analyzer.add_data(xbus_data.acc[0], xbus_data.acc[1], xbus_data.acc[2])
        rms = vib_analyzer.get_vibration_rms()
        if rms:
            # Визуальная шкала в терминале
            bar_length = int(rms[3] * 10) if rms[3] < 4 else 40
            bar = "█" * min(bar_length, 40) + "░" * (40 - min(bar_length, 40))
            print(f"\n📊 ВИБРАЦИЯ: X={rms[0]:.4f} Y={rms[1]:.4f} Z={rms[2]:.4f} | TOTAL={rms[3]:.4f} m/s²")
            print(f"   {bar}")

    # Вывод основных данных
    if xbus_data.packetCounterAvailable:
        print(f"packetCounter: {xbus_data.packetCounter}", end=' | ')
    if xbus_data.sampleTimeFineAvailable:
        print(f"sampleTimeFine: {xbus_data.sampleTimeFine}", end=' | ')
    if xbus_data.eulerAvailable:
        print(f"Roll: {xbus_data.euler[0]:.2f}° Pitch: {xbus_data.euler[1]:.2f}° Yaw: {xbus_data.euler[2]:.2f}°", end=' | ')
    if xbus_data.accAvailable:
        print(f"Acc: ({xbus_data.acc[0]:.2f}, {xbus_data.acc[1]:.2f}, {xbus_data.acc[2]:.2f}) m/s²", end=' | ')
    if xbus_data.rotAvailable:
        rot_deg = [xbus_data.rad2deg * r for r in xbus_data.rot]
        print(f"Gyro: ({rot_deg[0]:.2f}, {rot_deg[1]:.2f}, {rot_deg[2]:.2f}) deg/s")
    print()


def main():
    vib_analyzer = None
    simulation_dir = None
    
    try:
        # Создаём папку для текущей симуляции (локально в проекте)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        simulation_name = f"simulation_{timestamp}"
        project_root = os.path.dirname(os.path.abspath(__file__))
        local_sessions_root = os.path.join(project_root, "flight_logs")
        os.makedirs(local_sessions_root, exist_ok=True)
        simulation_dir = os.path.join(local_sessions_root, simulation_name)
        os.makedirs(simulation_dir, exist_ok=True)
        print(f"📁 Создана папка для симуляции: {simulation_dir}")
        
        serial = SerialHandler("/dev/ttyUSB0", 115200)

        go_to_config = bytes.fromhex('FA FF 30 00')
        go_to_measurement = bytes.fromhex('FA FF 10 00')

        serial.send_with_checksum(go_to_config)
        print("Переход в конфигурационный режим...")
        time.sleep(0.1)

        # Конфигурация для MTi-30 (AHRS, углы Эйлера, 100Hz)
        config_mti30 = bytes.fromhex('FA FF C0 20 10 20 FF FF 10 60 FF FF 20 30 00 64 40 20 00 64 40 30 00 64 80 20 00 64 C0 20 00 64 E0 20 FF FF')
        serial.send_with_checksum(config_mti30)
        print("Настройка выходных данных для MTi-30 (углы Эйлера, 100Hz)...")
        time.sleep(0.1)

        serial.send_with_checksum(go_to_measurement)
        print("Переход в измерительный режим...")
        print("Listening for packets... (Ctrl+C для остановки)\n")
        print("=" * 80)

        vib_analyzer = VibrationAnalyzer(window_size=50)

        packet = XbusPacket(on_data_available=lambda p: on_live_data_available(p, vib_analyzer))

        while True:
            byte = serial.read_byte()
            if byte:
                packet.feed_byte(byte)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Остановлено пользователем. Сохраняем данные...")
        
        if vib_analyzer and len(vib_analyzer.vibration_log) > 0:
            # Сохраняем лог и график во временную папку
            vib_analyzer.save_log(simulation_dir)
            vib_analyzer.plot_vibration(simulation_dir)
            
            # Сохраняем информацию о сессии
            info_filename = os.path.join(simulation_dir, 'session_info.txt')
            with open(info_filename, 'w') as f:
                f.write(f"Simulation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total samples: {len(vib_analyzer.vibration_log)}\n")
                f.write(f"Duration: {vib_analyzer.vibration_log[-1][0]:.2f} seconds\n")
                
                total_vibs = [log[1] for log in vib_analyzer.vibration_log]
                f.write(f"Max vibration: {max(total_vibs):.6f} m/s²\n")
                f.write(f"Min vibration: {min(total_vibs):.6f} m/s²\n")
                f.write(f"Avg vibration: {np.mean(total_vibs):.6f} m/s²\n")
            
            print(f"📁 Информация о сессии сохранена")
            
            # Выводим статистику
            total_vibs = [log[1] for log in vib_analyzer.vibration_log]
            print(f"\n📈 Статистика вибрации за сессию:")
            print(f"   Максимальная: {max(total_vibs):.4f} m/s²")
            print(f"   Средняя: {np.mean(total_vibs):.4f} m/s²")
            print(f"   Минимальная: {min(total_vibs):.4f} m/s²")
            print(f"   Всего семплов: {len(vib_analyzer.vibration_log)}")
            
            print(f"💾 Данные сохранены локально: {simulation_dir}")
        else:
            print("Нет данных для сохранения")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    main()
