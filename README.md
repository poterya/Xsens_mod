# Xsens_MTi_Serial_Reader
Use Python Serial library to config/read data from Xsens MTi

Install the dependency libary
Windows:
```
pip install pyserial
```

ubuntu
```
sudo pip3 install pyserial
```
or
```
sudo apt-get update
sudo apt-get -y install python3-serial
```

You need to change the COM and baudrate to your own sensor's COM name and baudrate(default is 115200, but could be configured in MT Manager - Device Settings) in raw_xsens_comms.py:
```
serial = SerialHandler("COM12", 921600) #for ubuntu, change to '/dev/ttyUSB0' #baudrate default is 115200, unless changed in MT Manager
```

#### if you want to configure the sensor, un-comment this line below:
```
serial.send_with_checksum(go_to_config)
set_output_conf(serial)
```

check the set_output_conf() function to send the hex command for setting up to output configuration, you could compose this message at:
MT Manager - Device Data View - GoToConfig - SetOuputConfiguration, and click edit, to add the messages needed.


run the code in the CMD in Windows like this:
```
python main.py
```
for ubuntu:
```
python3 main.py
```

### Build separate training sets

To save clipped CSVs without overwriting existing `Normal_mod` / `Deformed_mod`,
run:

```bash
python3 csv_plot_browser.py Normal --dataset-name set_01
python3 csv_plot_browser.py Deformed --dataset-name set_01
```

This creates:

```text
datasets/set_01/Normal_mod/
datasets/set_01/Deformed_mod/
```

Then train on the selected dataset:

```bash
python3 train_detector.py --base datasets/set_01 --output model_set_01.pkl
python3 train_rf_detector.py --base datasets/set_01 --output model_set_01_rf.pkl
```



This code has been checked with MTi-680 in Windows 11, and MTi-300 in ubuntu 18.04LTS(nVidia Jetson Nano),  not all other MTi models were tested, by they share the same Xbus communication protocol.

For MTi-300 with cable model CA-USB-MTi, if you don't have 'dev/ttyUSB0', 
```
git clone https://github.com/xsens/xsens_mt.git
cd ~/xsens_mt
make HAVE_LIBUSB=1
sudo modprobe usbserial
sudo insmod ./xsens_mt.ko
```
