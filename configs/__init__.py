def dynamic_load(config_name):
    module_path = f'configs.{config_name}'
    mod = __import__(module_path, fromlist=[''])

    return mod.get_args_parser()
