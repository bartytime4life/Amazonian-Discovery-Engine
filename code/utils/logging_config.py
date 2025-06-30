import logging, yaml, os
def setup_logging(default_path='config/logging_config.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    path = os.getenv(env_key, default_path)
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
