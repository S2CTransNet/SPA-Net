import yaml
import os,sys
import argparse
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self.get(key, False)
        if isinstance(value, dict):
            value = DotDict(value)
        return value

def get_cfg(args):
    dataset_name = args.dataset_name#.replace("-", "")
    filename = f'{BASE_DIR}/cfgs/{dataset_name}.yaml'
    module_name = f'cfgs.cfgs_{dataset_name}'
    try:
        module = __import__(module_name, fromlist=['create_cfg'])
    except ImportError:
        raise ImportError(f"No configuration module found for {dataset_name}")

    if not os.path.exists(filename) or not args.use_yaml:
        args = module.create_cfg(args)
        with open(filename, 'w') as file:
            yaml.dump(args, file, default_flow_style=False, allow_unicode=True)
    else:
        with open(filename, 'r', encoding='utf-8') as file:
            args = yaml.safe_load(file)
    config = DotDict(args)
    return config

class MainOpt:
    def __init__(self):
        self.p = argparse.ArgumentParser(description="Main options")
        self.p.add_argument('--dataset_name', type=str, default='ShapeNet-34', help='Dataset name.')
        self.p.add_argument('--use_yaml', action='store_true', default=False, help='If use the config.yaml file.')
        self.p.add_argument('--task', type=str, default='test', help='Type of task.')
        self.p.add_argument("--dataset_path", type=str, default=f'{os.path.dirname(__file__)}/data',
                            help='The parent path of dataset.')

    def parse(self, **kwargs):
        self.opt = self.p.parse_args(**kwargs)
        return self.opt

if __name__ == "__main__":
    main_opt = MainOpt()
    opt = main_opt.parse()
    dataset_names = ['KITTI', 'ShapeNet-55', 'ShapeNet-34', 'MVP']
    for name in dataset_names:
        opt.dataset_name = name
        config = get_cfg(opt)
