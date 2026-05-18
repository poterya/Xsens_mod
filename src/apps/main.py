#!/usr/bin/env python3

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from xsens.SerialHandler import SerialHandler
from xsens.XbusPacket import XbusPacket
from xsens.DataPacketParser import DataPacketParser, XsDataPacket
import time
from datetime import datetime
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os

# Класс для расчёта вибрации (с вычитанием гравитации)
class VibrationAnalyzer:
    def __init__(self, window_size=20):
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


class FullXsensLog:
    """Полный снимок распарсенного пакета Xsens (отдельно от vibration_log.csv)."""

    CSV_NAME = "xsens_full_log.csv"

    def __init__(self):
        self._t0 = time.time()
        self.rows = []

    def add_packet(self, d: XsDataPacket):
        t = time.time() - self._t0
        r2d = XsDataPacket.rad2deg

        def f3(v, avail):
            if not avail:
                return ("", "", "")
            return (f"{v[0]:.8f}", f"{v[1]:.8f}", f"{v[2]:.8f}")

        def f4(v, avail):
            if not avail:
                return ("", "", "", "")
            return (f"{v[0]:.8f}", f"{v[1]:.8f}", f"{v[2]:.8f}", f"{v[3]:.8f}")

        ex, ey, ez = f3(d.euler, d.eulerAvailable)
        q0, q1, q2, q3 = f4(d.quat, d.quaternionAvailable)
        ax, ay, az = f3(d.acc, d.accAvailable)
        fax, fay, faz = f3(d.freeAcc, d.freeAccAvailable)
        # rot в рад/с → как в консоли, в °/с
        if d.rotAvailable:
            gx = f"{d.rot[0] * r2d:.8f}"
            gy = f"{d.rot[1] * r2d:.8f}"
            gz = f"{d.rot[2] * r2d:.8f}"
        else:
            gx = gy = gz = ""
        mx, my, mz = f3(d.mag, d.magAvailable)
        lat, lon = (f"{d.latlon[0]:.8f}", f"{d.latlon[1]:.8f}") if d.latlonAvailable else ("", "")
        alt = f"{d.altitude:.8f}" if d.altitudeAvailable else ""
        vx, vy, vz = f3(d.vel, d.velocityAvailable)
        dvx, dvy, dvz = f3(d.deltaV, d.deltaVAvailable)
        dq0, dq1, dq2, dq3 = f4(d.deltaQ, d.deltaQAvailable)

        pc = str(d.packetCounter) if d.packetCounterAvailable else ""
        stf = str(d.sampleTimeFine) if d.sampleTimeFineAvailable else ""
        utc = f"{d.utcTime:.8f}" if d.utcTimeAvailable else ""
        sw = str(d.statusWord) if d.statusWordAvailable else ""
        temp = f"{d.temperature:.8f}" if d.temperatureAvailable else ""
        baro = str(d.baropressure) if d.baropressureAvailable else ""

        self.rows.append(
            (
                f"{t:.6f}",
                pc,
                stf,
                utc,
                ex,
                ey,
                ez,
                q0,
                q1,
                q2,
                q3,
                ax,
                ay,
                az,
                fax,
                fay,
                faz,
                gx,
                gy,
                gz,
                mx,
                my,
                mz,
                lat,
                lon,
                alt,
                vx,
                vy,
                vz,
                sw,
                temp,
                baro,
                dvx,
                dvy,
                dvz,
                dq0,
                dq1,
                dq2,
                dq3,
            )
        )

    def save(self, directory):
        path = os.path.join(directory, self.CSV_NAME)
        header = (
            "time_seconds,packet_counter,sample_time_fine,utc_time,"
            "euler_roll_deg,euler_pitch_deg,euler_yaw_deg,"
            "quat_w,quat_x,quat_y,quat_z,"
            "acc_x,acc_y,acc_z,"
            "free_acc_x,free_acc_y,free_acc_z,"
            "gyro_x_dps,gyro_y_dps,gyro_z_dps,"
            "mag_x,mag_y,mag_z,"
            "lat,lon,altitude,"
            "vel_x,vel_y,vel_z,"
            "status_word,temperature,baropressure,"
            "delta_v_x,delta_v_y,delta_v_z,"
            "delta_q_w,delta_q_x,delta_q_y,delta_q_z\n"
        )
        with open(path, "w") as f:
            f.write(header)
            for row in self.rows:
                f.write(",".join(row) + "\n")
        print(f"📁 Полный лог Xsens: {path}")
        return path


def on_live_data_available(packet, vib_analyzer=None, full_log=None):
    xbus_data = XsDataPacket() 
    DataPacketParser.parse_data_packet(packet, xbus_data)

    if full_log is not None:
        full_log.add_packet(xbus_data)

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
    full_xsens = None
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

        vib_analyzer = VibrationAnalyzer(window_size=20)
        print(f"Окно расчёта вибрации: {vib_analyzer.window_size} сэмплов")

        full_xsens = FullXsensLog()
        packet = XbusPacket(
            on_data_available=lambda p: on_live_data_available(p, vib_analyzer, full_xsens)
        )

        while True:
            byte = serial.read_byte()
            if byte:
                packet.feed_byte(byte)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Остановлено пользователем. Сохраняем данные...")
        
        if full_xsens is not None and len(full_xsens.rows) > 0:
            full_xsens.save(simulation_dir)

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
        elif full_xsens and len(full_xsens.rows) > 0:
            print(f"💾 Полный лог Xsens сохранён (вибрация не накопилась — мало сэмплов в окне): {simulation_dir}")
        else:
            print("Нет данных для сохранения")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    main()
