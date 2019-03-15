import logging
import sys

def log():
    """ Logs Errors, Warnings and ETC as Standard Output
    Args:
    Returns:
        logger: logger object
    """

    logger = logging.getLogger(__name__)
    # DEBUG < INFO < WARNING < ERROR < CRITICAL
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(module)s:%(lineno)d - %(process)d:%(thread)d - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


"""
Usage:
ex) logger.warning("Better to be preprocessed")

logger.critical()
logger.error()
logger.warning()
logger.info()
logger.debug()

"""
