from dataloader.ShapeNet import ShapeNet
from dataloader.PCN import PCNv2, PCN
from dataloader.Completion3DDataset import MVP
from dataloader.KITTI import KITTI
from torch.utils.data import DataLoader
from net.base import  net_builder_

def build_data_loader(args, config):
    if args.dataset_name == 'ShapeNet-55' or args.dataset_name == 'ShapeNet-34':
        test_data_loader = ShapeNet(config.dataset.test, args)
    elif  args.dataset_name == "PCN":
        test_data_loader = PCN(config.dataset.test, args)
    elif  args.dataset_name == "MVP":
        test_data_loader = MVP(config.dataset.test, args)
    elif  args.dataset_name == "KITTI":
        test_data_loader = KITTI(config.dataset.test, args)
    else:
        print('No data_loader choose')
        raise NotImplementedError()
    data_loader = {
               'test': DataLoader(test_data_loader, batch_size=1, shuffle=False)
               }

    return data_loader

def build_net(config):
    return net_builder_(config)
