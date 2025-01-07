import argparse
import os
from utils.runner import test

class MainOpt:
    def __init__(self):
        self.p = argparse.ArgumentParser(description="Main options")
        self.p.add_argument('--dataset_name', type=str, default='ShapeNet-34', help='Dataset name.')
        self.p.add_argument('--use_yaml', action='store_true', default=True, help='If use the config.yaml file.')
        self.p.add_argument('--seen_type', type=str, default='Unseen-21', help='Require from ShapeNet-34.("Seen-34" or "Unseen-21")')

    def parse(self, **kwargs):
        self.opt = self.p.parse_args(**kwargs)
        return self.opt

def runner(opt):
    print('-- Info')
    print("[GPU]Using single GPU.")
    test(opt)

if __name__ == "__main__":
    main_opt = MainOpt()
    opt = main_opt.parse()
    runner(opt)
