import argparse
import os


def create_cfg(args=None):
    """
    ShapeNet-34 example config
    """
    if args.task == 'test':
        if args.seen_type not in ["Seen-34", "Unseen-21"]:
            KeyError('seen_type is required in testing')
    parser = argparse.ArgumentParser(description='Generate nested configuration dictionary')

    # Dataset parameters
    dataset_defaults = {
        'name': args.dataset_name,
        'data_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataloader', args.dataset_name),
        'n_points': 8192,
        'pc_path': f'{args.dataset_path}/ShapeNet/shapenet_pc/',
        'subset': 'test',
    }
    for subset in ['test']:
        for k, v in dataset_defaults.items():
            default_value = os.path.join(v, args.seen_type) if k == 'data_path' and subset == 'test' else v
            parser.add_argument(f'--dataset.{subset}.{k}', type=type(v),
                                default=default_value if k != 'subset' else subset)

    # Model parameters
    model_defaults = {
        'subset': args.dataset_name,
        'num_points': 6144,
        'num_query': 384,
        'knn_layer': 1,
        'trans_dim': 384,
        'paths_weight': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weight', args.dataset_name),
        'aggregator': 'Transformer',
        'encoder': 'Transformer',
        'decoder': 'Transformer',
        'generator': 'Auto',
        'rebuilder1': 'FoldingNet',
        'rebuilder2': 'FoldingNet',
        'rebuilder3': 'FoldingNet',
        'rate': [1, 2, 2],
        'noise': True if args.dataset_name in ['PCN', 'KITTY'] and args.task == 'train' else False,
    }
    for k, v in model_defaults.items():
        parser.add_argument(f'--model.{k}', type=type(v), default=v)

    # Testing parameters
    testing_defaults = {
        'log_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs'),
        'weight_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weight', args.dataset_name),
        'weight_resume': 'best',  # best or last
    }
    for k, v in testing_defaults.items():
        parser.add_argument(f'--{k}', type=type(v), default=v)

    args = parser.parse_args()
    args_dict = vars(args)
    nested_args = {}
    for key, value in args_dict.items():
        parts = key.split('.')
        d = nested_args
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return nested_args
