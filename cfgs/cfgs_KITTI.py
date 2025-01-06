import argparse
import os

def create_cfg(args=None):
    """
    KITTI example config
    """
    parser = argparse.ArgumentParser(description='Generate nested configuration dictionary')
    if args.task != 'test':
        KeyError('KITTI is only used to test with pretrained weights in other dataset.')

    # Dataset parameters
    dataset_defaults = {
        'name': args.dataset_name,
        'txt_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataloader', f'{args.dataset_name}/test/500-1500.txt'),
        'n_points': 16384,
        'subset': 'test',
    }
    for subset in ['test']:
        for k, v in dataset_defaults.items():
            parser.add_argument(f'--dataset.{subset}.{k}', type=type(v), default=v if k != 'subset' else subset)

    # Model parameters
    model_defaults = {
        'subset': args.dataset_name,
        'num_points': 16384,
        'num_query': 256,
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
        'rate': [1, 2, 4],
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
