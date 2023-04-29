import pickle
import time
import atexit
import threading
from collections import UserDict


class PeriodicPickleVault:
    def __init__(self, file_name, save_interval=None):
        self.file_name = file_name
        self.save_interval = save_interval

        self.data = self.load_data()

        atexit.register(self.save_data)

        if self.save_interval:
            self.stop_event = threading.Event()
            self.periodic_save_thread = threading.Thread(target=self.periodic_save)
            self.periodic_save_thread.start()

    def load_data(self):
        try:
            with open(self.file_name, 'rb') as file:
                data = pickle.load(file)
        except FileNotFoundError:
            data = {}
        return data

    def save_data(self):
        with open(self.file_name, 'wb') as file:
            pickle.dump(self.data, file)

    def periodic_save(self):
        while not self.stop_event.is_set():
            time.sleep(self.save_interval)
            self.save_data()

    def stop(self):
        self.stop_event.set()
        self.periodic_save_thread.join()
        self.save_data()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False  # Do not suppress the exception


class AutoPickleVault(UserDict):
    def __init__(self, file_name, *args, **kwargs):
        self.file_name = file_name
        super().__init__(*args, **kwargs)
        self.data = self.load_data()

    def load_data(self):
        try:
            with open(self.file_name, 'rb') as file:
                data = pickle.load(file)
        except FileNotFoundError:
            data = {}
        return data

    def save_data(self):
        with open(self.file_name, 'wb') as file:
            pickle.dump(self.data, file)
        print('Saved Data')

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.save_data()

    def __delitem__(self, key):
        super().__delitem__(key)
        self.save_data()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.save_data()


def main():

    # TODO: change PeriodicPickleVault to UserDict, add PickleVault class, add PeriodicAutoPickleVault
    data_manager = PeriodicPickleVault('data.pkl')

    try:
        # Your data manipulation and processing code goes here
        pass
    except Exception as e:
        # Handle the exception or log it if needed
        raise
    finally:
        data_manager.stop()

    with PeriodicPickleVault('data.pkl') as data_manager:
        # Your data manipulation and processing code goes here
        pass

    auto_save_dict = AutoPickleVault('data.pkl')

    # Your data manipulation and processing code goes here
    auto_save_dict['key'] = 'value'
    del auto_save_dict['key']


if __name__ == '__main__':
    main()

