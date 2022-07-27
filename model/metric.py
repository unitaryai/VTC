import collections
import time

import faiss
import numpy as np
import pandas as pd
import torch


class MetricTracker:
    def __init__(self, *metrics):
        self.metrics = {}
        for m in metrics:
            self.add_metric(m)
        self.reset()

    def add_metric(self, metric):
        self.metrics[metric.name] = metric

    def set_writer(self, writer):
        for m in self.metrics.values():
            m.set_writer(writer)

    def reset(self):
        for m in self.metrics.values():
            m.reset()

    def update(self, loss, output, meta):
        for m in self.metrics.values():
            m.update(loss, output, meta)

    def avg(self):
        res = {}
        for m in self.metrics.values():
            res[m.name] = m.avg()
        return res

    def result(self):
        res = {}
        for m in self.metrics.values():
            res.update(m.result())
        return res


class BaseMetric:
    def __init__(self, name):
        self.name = name
        self.writer = None
        self.is_train = True
        self.is_val = True

    def set_writer(self, writer):
        self.writer = writer

    def reset(self):
        raise NotImplementedError()

    def update(self, loss, output, meta):
        raise NotImplementedError()

    def avg(self):
        raise NotImplementedError()

    def result(self):
        raise NotImplementedError()


class ScalarPerBatchMetric(BaseMetric):
    def __init__(self, name, metric_fun):
        super().__init__(name)
        self.fun = metric_fun
        self._data = pd.DataFrame(
            index=[self.name], columns=["total", "counts", "average"]
        )
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, loss, output, meta, n=1):
        value = self.fun(loss, output, meta)
        if self.writer is not None:
            self.writer.add_scalar(self.name, value)
        self._data.total[self.name] += value * n
        self._data.counts[self.name] += n
        self._data.average[self.name] = (
            self._data.total[self.name] / self._data.counts[self.name]
        )

    def avg(self):
        return self._data.average[self.name]

    def result(self):
        return dict(self._data.average)


class LossMetric(ScalarPerBatchMetric):
    def __init__(self):
        super().__init__("loss", lambda loss, o, m: loss)


class RecallAtK(BaseMetric):
    def __init__(self, name_a, name_b, k_vals=5):
        super().__init__("recall@k")
        if not isinstance(k_vals, collections.Iterable):
            k_vals = [k_vals]
        self.k_vals = k_vals
        self.name_a = name_a
        self.name_b = name_b
        self.is_train = False
        self.knn_config = faiss.GpuIndexFlatConfig()
        self.knn_config.useFloat16 = False
        self.insert_index = 0
        self.features_a_list = []
        self.features_b_list = []

    def reset(self):
        self.insert_index = 0
        self.features_a_list = []
        self.features_b_list = []

    def update(self, loss, output, meta):
        fa = output[0]
        fb = output[1]
        batch_size = fa.shape[0]
        self.knn_config.device = fa.device.index
        end = self.insert_index + batch_size
        fa = fa.detach().cpu()
        fb = fb.detach().cpu()

        self.features_a_list.append(fa)
        self.features_b_list.append(fb)

        self.insert_index = end

    def compute(self, features_a, features_b):
        num_samples = features_a.shape[0]
        num_dims = features_a.shape[1]
        faiss_search_index = faiss.GpuIndexFlatL2(
            faiss.StandardGpuResources(), num_dims, self.knn_config
        )
        faiss_search_index.add(features_a)
        _, k_closest_points = faiss_search_index.search(
            features_b, int(np.max(self.k_vals) + 1)
        )

        recall_all_k = []
        for k in self.k_vals:
            r_at_k = (
                np.sum(
                    [
                        1
                        for target, rp in enumerate(k_closest_points)
                        if target in rp[:k]
                    ]
                )
                / num_samples
            )
            recall_all_k.append((k, r_at_k))
        return recall_all_k

    def avg(self):
        return None

    def result(self):
        tic = time.time()
        print("RecallAtK: result()...", end=" ", flush=True)

        features_a = torch.cat(self.features_a_list).numpy()
        features_b = torch.cat(self.features_b_list).numpy()

        assert self.insert_index == len(features_a)

        res = {}
        for k, recall in self.compute(features_a, features_b):
            res[f"{self.name_b}_from_{self.name_a}-recall_at_{k}"] = recall
        for k, recall in self.compute(features_b, features_a):
            res[f"{self.name_a}_from_{self.name_b}-recall_at_{k}"] = recall

        if self.writer:
            for name, recall in res.items():
                self.writer.add_scalar(name, recall)

        print("RecallAtK: result() took %.3fs" % (time.time() - tic))

        return res
