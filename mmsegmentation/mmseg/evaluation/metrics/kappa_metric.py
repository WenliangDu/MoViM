from typing import Sequence, Optional, Dict
from mmengine.evaluator import BaseMetric
import numpy as np
from sklearn.metrics import cohen_kappa_score

from mmseg.registry import METRICS

@METRICS.register_module()
class KappaMetric(BaseMetric):
    def __init__(self,
                 ignore_index: int = 255,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.results = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            # 适配 mmseg 1.2.2 的数据结构
            pred = sample['pred_sem_seg']['data'].squeeze().cpu().numpy().flatten()
            gt = sample['gt_sem_seg']['data'].squeeze().cpu().numpy().flatten()

            # 排除 ignore_index
            valid_mask = gt != self.ignore_index
            pred = pred[valid_mask]
            gt = gt[valid_mask]

            self.results.append((gt, pred))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        # 拼接所有结果
        gt_all = np.concatenate([r[0] for r in results])
        pred_all = np.concatenate([r[1] for r in results])
        kappa = cohen_kappa_score(gt_all, pred_all)
        return dict(kappa=kappa)