from utils import tools, obj_names
import time
import csv
import os
from datetime import datetime
from tqdm import tqdm

def validate(base_model, test_dataloader, args, visualizer=None, np=None):
    objnames = obj_names.get_obj_name(args.dataset_name)
    gt, dense_points, fps, log_dict, level = val_forward(base_model, np, args, test_dataloader, visualizer)
    total_f1 = total_loss_l1 = total_loss_l2 = total_EMDistance = 0.0
    total_count = total_fps = 0

    print(f'========================= TEST RESULTS =========================')
    print('taxonomy_id', '', '#Count', '', '#F1-score', '', '#CDL1', '', '#CDL2', '', '#EMDistance', '', '#FPS', '', '#Objname')
    formatted_now = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f'results/{args.dataset_name}/{level}_{formatted_now}.csv')
    directory = os.path.dirname(csv_file)
    os.makedirs(directory, exist_ok=True)
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['taxonomy_id', '#Count', '#F1-score', '#CDL1', '#CDL2', '#EMDistance', '#FPS', '#Objname'])

        for taxonomy_id, stats in log_dict.items():
            avg_f1 = stats['F1-score'] / stats['count']
            avg_l1_loss = stats['dense_loss_l1'] / stats['count']
            avg_l2_loss = stats['dense_loss_l2'] / stats['count']
            avg_EMDistance = stats['EMDistance'] / stats['count']
            fps = stats['fps'] / stats['count']
            objname = objnames.get(taxonomy_id, None)
            if args.dataset_name in ['ShapeNet-55','ShapeNet-34']:
                count = stats['count'] // 8
            else:
                count = stats['count']
            print(f"{taxonomy_id:<10} {count:<6} {avg_f1:<7.3f} {avg_l1_loss:<7.3f} {avg_l2_loss:<7.3f} {avg_EMDistance:<7.3f} {fps:<7.3f}{objname:<10}")
            writer.writerow([taxonomy_id, count, avg_f1, avg_l1_loss, avg_l2_loss, avg_EMDistance, fps, objname])

            if visualizer is not None:
                visualizer.visualize_pcd_batch(stats['dense_points'], 1, pcd_name=f"{level}_{objname}_dense_points")
                visualizer.visualize_pcd_batch(stats['sparse_points'], 1, pcd_name=f"{level}_{objname}_sparse_points")
                visualizer.plot_loss(avg_l1_loss, 1, loss_name=f"{level}_{objname}_CDL1")
                visualizer.plot_loss(avg_l2_loss, 1, loss_name=f"{level}_{objname}CDL2")
                visualizer.plot_loss(avg_f1, 1, loss_name=f"{level}_{objname}_F1")
                visualizer.plot_loss(avg_EMDistance, 1, loss_name=f"{level}_{objname}_EMDistance")
                visualizer.plot_loss(total_fps, 1, loss_name=f"{level}_{objname}_FPS_test")

            total_f1 += avg_f1
            total_loss_l1 += avg_l1_loss
            total_loss_l2 += avg_l2_loss
            total_EMDistance += avg_EMDistance
            total_count += 1
            total_fps += fps

    overall_avg_l1_loss = total_loss_l1 / total_count
    overall_avg_l2_loss = total_loss_l2 / total_count
    overall_avg_EMDistance = total_EMDistance / total_count
    overall_f1 = total_f1 / total_count
    total_fps = total_fps / total_count

    print(f"{'Overall':<10} {total_count:<6} {overall_f1:<7.3f} {overall_avg_l1_loss:<7.3f} {overall_avg_l2_loss:<7.3f} {overall_avg_EMDistance:<7.3f} {total_fps:<7.3f}",'-')
    if visualizer is not None:
        visualizer.plot_loss(overall_avg_l1_loss, 1, loss_name=f'{level}_CDL1_avg_test')
        visualizer.plot_loss(overall_avg_l2_loss, 1, loss_name=f'{level}_CDL2_avg_test')
        visualizer.plot_loss(overall_f1, 1, loss_name=f'{level}_F1_avg_test')
        visualizer.plot_loss(total_fps, 1, loss_name=f'{level}_FPS_test')

    return overall_avg_l1_loss, overall_avg_l2_loss, overall_f1, total_fps


def val_forward(base_model,np,args,test_dataloader,visualizer):
    base_model.eval()
    npoints = 8192
    log_dict = {}
    crop = [int(npoints * np), int(npoints * np)] if np is not None else [int(npoints * 0.25)]
    print(f'[Auto validate][level:{crop[0]}]' if np is not None else f'[Auto validate]')
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_dataloader,desc='validate',unit='batch')):
            taxonomy_id = f'{sample["taxonomy_id"][0]}'
            gt = sample["gt"].to('cuda')
            if 'partial' not in sample:
                choice = [torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]),
                          torch.Tensor([-1, 1, 1]), torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]),
                          torch.Tensor([1, -1, -1]), torch.Tensor([-1, -1, -1])]
                partial_list = []
                for item in choice:
                    partial, _ = tools.separate_point_cloud(gt, npoints, crop, fixed_points=item)
                    partial = tools.fps(partial, 2048)
                    partial_list.append(partial.to('cuda'))

                for partial in partial_list:
                    start_time = time.time()
                    ret = base_model(partial)
                    dense_points = ret[-1]
                    end_time = time.time()
                    fps = 1 / (end_time - start_time)
                    _metrics = Metrics.get(dense_points, gt)
                    _metrics = [_metric.item() for _metric in _metrics]

                    if taxonomy_id not in log_dict:
                        log_dict[taxonomy_id] = {'F1-score': 0, 'dense_loss_l1': 0, 'dense_loss_l2': 0, 'EMDistance': 0,
                                                 'count': 0, 'fps': 0, 'dense_points': 0, 'sparse_points': 0}
                    log_dict[taxonomy_id]['F1-score'] += _metrics[0]
                    log_dict[taxonomy_id]['dense_loss_l1'] += _metrics[1]
                    log_dict[taxonomy_id]['dense_loss_l2'] += _metrics[2]
                    log_dict[taxonomy_id]['EMDistance'] += _metrics[3]
                    log_dict[taxonomy_id]['count'] += 1
                    log_dict[taxonomy_id]['fps'] += fps
                    if visualizer is not None:
                        log_dict[taxonomy_id]['dense_points'] = dense_points
                        log_dict[taxonomy_id]['sparse_points'] = ret[0]
            else:
                partial = sample['partial'].to('cuda')
                ret = base_model(partial)
                dense_points = ret[-1]
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                _metrics = Metrics.get(dense_points, gt)
                _metrics = [_metric.item() for _metric in _metrics]

                if taxonomy_id not in log_dict:
                    log_dict[taxonomy_id] = {'F1-score': 0, 'dense_loss_l1': 0, 'dense_loss_l2': 0, 'EMDistance': 0,
                                             'count': 0, 'fps': 0, 'dense_points': 0, 'sparse_points': 0}
                log_dict[taxonomy_id]['F1-score'] += _metrics[0]
                log_dict[taxonomy_id]['dense_loss_l1'] += _metrics[1]
                log_dict[taxonomy_id]['dense_loss_l2'] += _metrics[2]
                log_dict[taxonomy_id]['EMDistance'] += _metrics[3]
                log_dict[taxonomy_id]['count'] += 1
                log_dict[taxonomy_id]['fps'] += fps
                if visualizer is not None:
                    log_dict[taxonomy_id]['dense_points'] = dense_points
                    log_dict[taxonomy_id]['sparse_points'] = ret[0]

        return gt, dense_points, fps, log_dict, crop[0]

def val_forward_KITTI(base_model,test_dataloader,visualizer=None):
    base_model.eval()
    print(f'[Auto validate][KITTI]')
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_dataloader,desc='validate',unit='batch')):
            partial = sample['partial']
            partial = partial.to('cuda')
            ret = base_model(partial)
            dense_points = ret[-1]
            if visualizer is not None:
                visualizer.visualize_pcd_batch(dense_points, 1, pcd_name=f"car_{idx}")

import logging
import open3d
import torch
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from extensions.emd import emd_module as emd

class Metrics(object):
    """References from https://github.com/yuxumin/PoinTr/blob/master/"""
    ITEMS = [{
        'name': 'F-Score',
        'enabled': True,
        'eval_func': 'cls._get_f_score',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'CDL1',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel1',
        'eval_object': ChamferDistanceL1(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'CDL2',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel2',
        'eval_object': ChamferDistanceL2(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'EMDistance',
        'enabled': True,
        'eval_func': 'cls._get_emd_distance',
        'eval_object': emd.emdModule(),
        'is_greater_better': False,
        'init_value': 32767
    }]

    @classmethod
    def get(cls, pred, gt, require_emd=False):
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            if not require_emd and 'emd' in item['eval_func']:
                _values[i] = torch.tensor(0.).to(gt.device)
            else:
                eval_func = eval(item['eval_func'])
                _values[i] = eval_func(pred, gt)

        return _values

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]

    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):

        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        b = pred.size(0)
        device = pred.device
        assert pred.size(0) == gt.size(0)
        if b != 1:
            f_score_list = []
            for idx in range(b):
                f_score_list.append(cls._get_f_score(pred[idx:idx+1], gt[idx:idx+1]))
            return sum(f_score_list)/len(f_score_list)
        else:
            pred = cls._get_open3d_ptcloud(pred)
            gt = cls._get_open3d_ptcloud(gt)

            dist1 = pred.compute_point_cloud_distance(gt)
            dist2 = gt.compute_point_cloud_distance(pred)

            recall = float(sum(d < th for d in dist2)) / float(len(dist2))
            precision = float(sum(d < th for d in dist1)) / float(len(dist1))
            result = 2 * recall * precision / (recall + precision) if recall + precision else 0.
            result_tensor = torch.tensor(result).to(device)
            return result_tensor

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        """pred and gt bs is 1"""
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)

        return ptcloud

    @classmethod
    def _get_chamfer_distancel1(cls, pred, gt):
        chamfer_distance = cls.ITEMS[1]['eval_object']
        return chamfer_distance(pred, gt) * 1000

    @classmethod
    def _get_chamfer_distancel2(cls, pred, gt):
        chamfer_distance = cls.ITEMS[2]['eval_object']
        return chamfer_distance(pred, gt) * 1000

    @classmethod
    def _get_emd_distance(cls, pred, gt, eps=0.005, iterations=100):
        emd_loss = cls.ITEMS[3]['eval_object']
        dist, _ = emd_loss(pred, gt, eps, iterations)
        emd_out = torch.mean(torch.sqrt(dist))
        return emd_out * 1000

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value





