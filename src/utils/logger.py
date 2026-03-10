import logging
import os

def setup_logger(log_file_path: str):
    """ Sets up the logger for the project. Logs will be saved to 'train.log' with timestamps and messages
    
    Inputs:
    - log_file_path: The path to the directory where the log file will be saved
    
    Outputs:
    - train_logger: A logger object that can be used to log messages throughout the project"""
    
    train_logger = logging.getLogger(__name__)
    train_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )
    file_handler = logging.FileHandler(
        os.path.join(log_file_path, "train.log"),
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    train_logger.addHandler(file_handler)
    train_logger.addHandler(console_handler)

    return train_logger