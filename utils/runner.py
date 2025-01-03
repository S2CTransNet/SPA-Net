import os
import torch
from datetime import datetime
from utils import tools, builder, evaluate
import torch.nn as nn
import yaml


def get_cfg(args):
    dataset_name = args.dataset_name#.replace("-", "")
    filename = f'cfgs/{dataset_name}.yaml'
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
    config = tools.DotDict(args)
    return config

def test(args=None):
    config = get_cfg(args)
    formatted_now = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    run_name = f"{args.task}_{args.dataset_name}_{formatted_now}"
    data_loader = builder.build_data_loader(args, config)
    visualizer = tools.TensorboardVisualizer(log_dir=os.path.join(config.log_path, run_name))
    module = builder.build_net(config.model)
    if os.path.exists(config.weight_path):
        if config.weight_resume == 'best':
            weight_path = f"{config.weight_path}/{args.dataset_name}_best.pth"
            module.load_state_dict(torch.load(weight_path))
        elif config.weight_resume == 'last':
            weight_path = f"{config.weight_path}/{args.dataset_name}_checkpoint_last.pth"
            checkpoint = torch.load(weight_path)
            module.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError('No weight path found.')
    else:
        raise ValueError('No weight path found.')
    module.to('cuda')
    if args.dataset_name == 'ShapeNet-55' or args.dataset_name ==  'ShapeNet-34':
        nps = [1/4,1/2,3/4]
        for np in nps:
            _ = evaluate.validate(module, data_loader['test'], args, visualizer, np)
    else:
        evaluate.val_forward_KITTI(module, data_loader['test'], visualizer)
