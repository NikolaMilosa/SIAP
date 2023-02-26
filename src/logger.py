import logging

def get_logger(name):
    FORMAT = '[%(name)s] [%(asctime)s] %(levelname)-8s %(message)s'
    logging.basicConfig(format=FORMAT,level=logging.INFO)
    return logging.getLogger(name)
    