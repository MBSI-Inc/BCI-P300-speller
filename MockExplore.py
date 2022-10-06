class MockExplore:
    def __init__(self, log=False) -> None:
        self.log = log
        self.marker_counter = 0

    def connect(self, device_name):
        if self.log:
            print("Mock Explore connect:", device_name)
        return

    def record_data(self, file_name, file_type, do_overwrite, block):
        if self.log:
            print("Mock Explore start recording:", file_name, file_type, do_overwrite)
        return

    def set_marker(self, code):
        self.marker_counter += 1
        if self.log:
            print("Marker", self.marker_counter, code)
        return

    def stop_recording(self):
        if self.log:
            print("Mock Explore stop recording")

    def set_sampling_rate(self, rate):
        return

    def disable_module(self, module_name):
        if self.log:
            print(f'Mock Explore disable module {module_name}')
        return