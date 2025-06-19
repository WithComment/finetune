import logging
import sys

def get_logger(name):
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  logger.propagate = False
  # Create console handler and formatter
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  console_handler.setFormatter(formatter)

  # Add handler to logger (avoid duplicate handlers)
  if not logger.handlers:
    logger.addHandler(console_handler)

  return logger