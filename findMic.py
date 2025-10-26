# find_mic.py
import pyaudio
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
print("Listing Audio Devices:")
for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print(f"Input Device ID {i} - {p.get_device_info_by_host_api_device_index(0, i).get('name')}")
p.terminate()