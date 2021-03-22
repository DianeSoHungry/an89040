import configparser


def get_cfg(section, key):
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.get(section, key)
