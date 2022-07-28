import time

import numpy as np
import torch
import wandb
from torchvision.utils import make_grid

from evaluation.retrieval_evaluation import retrieval_evaluation
from model.metric import LossMetric, MetricTracker
from utils import extract_tensors, inf_loop, is_image_like_batch, move_to

from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metrics, optimizer, lr_scheduler, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.loss_args = config.get("loss_args", {})

        self.train_metrics = MetricTracker(*[m for m in self.metrics if m.is_train])
        self.train_metrics.add_metric(LossMetric())
        self.train_metrics.set_writer(self.writer)
        self.valid_metrics = MetricTracker(*[m for m in self.metrics if m.is_val])
        self.valid_metrics.add_metric(LossMetric())
        self.valid_metrics.set_writer(self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        batch_tic = time.time()
        lr = self.lr_scheduler.get_last_lr()[0]
        hz_list = []
        self.model.writer = self.writer

        for batch_idx, (*data, meta) in enumerate(self.data_loader):
            batch_size = extract_tensors(data)[0].shape[0]

            data = move_to(data, self.device)
            meta = move_to(meta, self.device)

            self.optimizer.zero_grad()
            output = self.model(*data)
            loss = self.criterion(output, meta, **self.loss_args)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update(loss.item(), output, meta)

            toc = time.time() - batch_tic
            hz = batch_size / toc
            hz_list.append(hz)
            hz_list = hz_list[-1000:]
            batch_tic = time.time()

            if batch_idx % self.log_step == 0:
                wandb.log({"loss": loss.item()})
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f} Speed: {:.2f}Hz (av {:.2f}Hz) LR: {:.6f}".format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item(),
                        np.mean(hz_list[-500:]),
                        hz,
                        lr,
                    )
                )
                if is_image_like_batch(data):
                    self.writer.add_image(
                        "input", make_grid(data[0].cpu(), nrow=8, normalize=True)
                    )

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        del output
        del data
        del loss
        torch.cuda.empty_cache()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})
            wandb.log({"val_" + k: v for k, v in val_log.items()})

        self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.logger.debug("Starting validation")
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (*data, meta) in enumerate(self.valid_data_loader):
                data = move_to(data, self.device)
                meta = move_to(meta, self.device)

                output = self.model(*data)
                loss = self.criterion(output, meta, **self.loss_args)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid"
                )
                self.valid_metrics.update(loss.item(), output, meta)
                if is_image_like_batch(data):
                    self.writer.add_image(
                        "input", make_grid(data.cpu(), nrow=8, normalize=True)
                    )

        self.logger.debug("Starting MSRVTT val")

        if hasattr(self.model, "branch_to_adapt_val"):
            original_branch_to_adapt_val = self.model.branch_to_adapt_val
        else:
            original_branch_to_adapt_val = None

        outdf = retrieval_evaluation(
            self.model, "MSRVTT_videos", "full-val", self.device
        )
        vtt, ttv = outdf.loc["R@10"].tolist()

        self.writer.add_scalar("msrvtt_val_vtt", vtt)
        self.writer.add_scalar("msrvtt_val_ttv", ttv)
        wandb.log({"msrvtt_val_vtt": vtt})
        wandb.log({"msrvtt_val_ttv": ttv})

        # Get resuts skipping the adapting
        self.model.branch_to_adapt_val = "skip"
        outdf = retrieval_evaluation(
            self.model, "MSRVTT_videos", "full-val", self.device
        )
        vtt, ttv = outdf.loc["R@10"].tolist()

        self.writer.add_scalar("msrvtt_val_skipadapt_vtt", vtt)
        self.writer.add_scalar("msrvtt_val_skipadapt_ttv", ttv)
        wandb.log({"msrvtt_val_skipadapt_vtt": vtt})
        wandb.log({"msrvtt_val_skipadapt_ttv": ttv})

        # Restore original branch_to_adapt_val from config
        self.model.branch_to_adapt_val = original_branch_to_adapt_val

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
