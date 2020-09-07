import csv
import logging
import sys
import os
import datetime

from collections import defaultdict


class CSVOutputFormat():
    def __init__(self, filename):
        """
        log to a file, in a CSV format
        :param filename: (str) the file to write the log to
        """

        self.file = open(filename, "w+t")
        self.keys = []
        self.separator = ","

    def write(self, key_values):
        # Add our current row to the history

        for key in sorted(key_values.keys()):
            if key.find('/') > 0:
                key_values[key.split('/')[1]] = key_values[key]
                key_values.pop(key, None)
        extra_keys = key_values.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, key) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(key)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.separator * len(extra_keys))
                self.file.write("\n")
        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            value = key_values.get(key)
            if value is not None:
                self.file.write(str(value))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        """
        closes the file
        """
        self.file.close()


class HumanOutputFormat():
    def __init__(self, filename_or_file):
        """
        log to a file, in a human readable format
        :param filename_or_file: (str or File) the file to write the log to
        """
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "write"), f"Expected file or str, got {filename_or_file}"
            self.file = filename_or_file
            self.own_file = False

    def write(self, key_values):
        key2str = {}
        tag = None
        for key, value in sorted(key_values.items()):
            if isinstance(value, float):
                value_str = f"{value:<8.3g}"
            else:
                value_str = str(value)

            if key.find("/") > 0:  # Find tag and add it to the dict
                tag = key[: key.find("/") + 1]
                key2str[self._truncate(tag)] = ""
            # Remove tag from key
            if tag is not None and tag in key:
                key = str("   " + key[len(tag) :])

            key2str[self._truncate(key)] = self._truncate(value_str)

        # Find max widths
        if len(key2str) == 0:
            warnings.warn("Tried to write empty key-value dict")
            return
        else:
            key_width = max(map(len, key2str.keys()))
            val_width = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (key_width + val_width + 7)
        lines = [dashes]
        for key, value in key2str.items():
            key_space = " " * (key_width - len(key))
            val_space = " " * (val_width - len(value))
            lines.append(f"| {key}{key_space} | {value}{val_space} |")
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")
        self.file.flush()

    @classmethod
    def _truncate(cls, string, max_length = 23):
        return string[: max_length - 3] + "..." if len(string) > max_length else string

    def write_sequence(self, sequence):
        sequence = list(sequence)
        for i, elem in enumerate(sequence):
            self.file.write(elem)
            if i < len(sequence) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.own_file:
            self.file.close()

class Logger(object):

    DEFAULT = None
    CURRENT = None  

    def __init__(self, outputs, folder = './logs'):
        self.name_to_value = defaultdict(float) 
        self.dir = folder  
        self.outputs = outputs

    def record(self, key, value):

        self.name_to_value[key] = value

    def dump(self, step = 0):

        for writer in self.outputs:
            writer.write(self.name_to_value)

        self.name_to_value.clear()

    def log(self, *args, level = 'INFO'):

        if self.level <= level:
            self._do_log(args)


    def set_level(self, level):
        """
        Set logging threshold on current logger.
        :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        """
        self.level = level

    def get_dir(self):
        """
        Get directory that log files are being written to.
        will be None if there is no output directory (i.e., if you didn't call start)
        :return: (str) the logging directory
        """
        return self.dir

    def close(self):
        """
        closes the file
        """
        for _format in self.output_formats:
            _format.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        """
        log to the requested format outputs
        :param args: (list) the arguments to log
        """
        for _format in self.output_formats:
            if isinstance(_format, SeqWriter):
                _format.write_sequence(map(str, args))


Logger.CURRENT = Logger(HumanOutputFormat(sys.stdout))

def dump(step):
    """
    Write all of the diagnostics from the current iteration
    """
    Logger.CURRENT.dump(step)

def record(key, value):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    :param key: (Any) save to log this key
    :param value: (Any) save to log this value
    :param exclude: (str or tuple) outputs to be excluded
    """
    Logger.CURRENT.record(key, value)

def configure(algorithm, environment, log_to_file = False, folder = None):

    if folder is None:
        folder = "./logs"
    
    folder = os.path.join(folder, algorithm, environment)
    
    output = [HumanOutputFormat(sys.stdout)]
    if log_to_file:       
        os.makedirs(folder, exist_ok=True)
        file_name = "run" + datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S-%f") + ".csv"
        file_name = os.path.join(folder, file_name)
        output.append(CSVOutputFormat(file_name))

    Logger.CURRENT = Logger(output, folder=folder)
    print(f"Logging to {folder}")