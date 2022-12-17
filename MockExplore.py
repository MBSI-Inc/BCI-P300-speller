from threading import Thread
import time
import csv
import random

TIME_SCALE_MOD = 1000000
DISPLAYED_CHARS = "123456789".upper()


class MockExplore:
    def __init__(self, log=False) -> None:
        self.log = log
        self.sf = 250
        self.marker_counter = 0
        self.stop_write = False
        self.file_name = "mock"
        self.chars = [x for x in DISPLAYED_CHARS]
        self.chars_counter = 0

    def connect(self, device_name):
        if self.log:
            print("Mock Explore connect:", device_name)
        return

    def record_data(self, file_name, file_type, do_overwrite, block):
        if self.log:
            print("Mock Explore start recording:", file_name, file_type, do_overwrite)
        self.file_name = file_name
        self.stop_write = False
        self.thread = Thread(target=self.write_fake_signal_to_file)
        self.thread.start()
        return

    def set_marker(self, code):
        self.marker_counter += 1
        if self.log:
            print("Marker", self.marker_counter, code)

        if (self.marker_counter == 1):
            with open(self.file_name + "_Marker.csv", newline='', mode='w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(["TimeStamp", "Code"])
        else:
            with open(self.file_name + "_Marker.csv", newline='', mode='a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow([round(time.time() % TIME_SCALE_MOD, 4), "sw_" + self.chars[self.chars_counter]])
                self.chars_counter += 1
        if (self.chars_counter >= len(self.chars)):
            self.chars_counter = 0
            random.shuffle(self.chars)
        return

    def stop_recording(self):
        if self.log:
            print("Mock Explore stop recording")
        self.stop_write = True
        if (self.thread):
            self.thread.join()

    def set_sampling_rate(self, rate):
        self.sf = rate
        return

    def disable_module(self, module_name):
        if self.log:
            print(f'Mock Explore disable module {module_name}')
        return

    def write_fake_signal_to_file(self):
        time_between = 1/self.sf
        with open(self.file_name + "_ExG.csv", newline='', mode='w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["TimeStamp", "ch1", "ch2", "ch3", "ch4"])
            while (not self.stop_write):
                writer.writerow([round(time.time() % TIME_SCALE_MOD, 4),
                                 round(random.uniform(18100, 18200), 2),
                                 round(random.uniform(-600, -400), 2),
                                 round(random.uniform(21300, 21700), 2),
                                 round(random.uniform(2100, 2400), 2)])
                time.sleep(time_between)
        return
