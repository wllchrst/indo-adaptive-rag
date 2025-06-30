import yaml
import os
class ConfigurationHelper:
    def __init__(self):
        self.configs = {}
        self.gather_config()
        
    def gather_config(self):
        config_file_paths = os.listdir("configs")

        for file_name in config_file_paths:
            file_path = os.path.join("configs", file_name)
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
                self.configs[file_name.split('.')[0]] = data

configuration_helper = ConfigurationHelper()