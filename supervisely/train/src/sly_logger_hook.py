import datetime
import torch
from typing import Optional, Union
from pathlib import Path
import sly_globals as g
from mmengine.hooks import LoggerHook
from mmengine.registry import HOOKS
from sly_train_progress import (
    add_progress_to_request,
    get_progress_cb,
    set_progress,
)

import supervisely as sly

@HOOKS.register_module()
class SuperviselyLoggerHook(LoggerHook):
    def __init__(self,
                 interval: int = 10,
                 ignore_last: bool = True,
                 interval_exp_name: int = 1000,
                 out_dir: Optional[Union[str, Path]] = None,
                 out_suffix: str = '.json',
                 keep_local: bool = False,
                 file_client_args: Optional[dict] = None,
                 log_metric_by_epoch: bool = True,
                 backend_args: Optional[dict] = None):
        super(SuperviselyLoggerHook, self).__init__(interval, ignore_last, interval_exp_name, out_dir, out_suffix, keep_local, file_client_args, log_metric_by_epoch, backend_args)
        self.progress_epoch = None
        self.progress_iter = None
        self._lrs = []
        self.time_sec_tot = 0
        self.start_iter = 0

    def before_run(self, runner) -> None:
        super(SuperviselyLoggerHook, self).before_run(runner)
        self.progress_epoch = sly.Progress("Epochs", runner.max_epochs)
        
        train_dataset_size = len(runner.train_dataloader.dataset)
        batch_size = runner.train_dataloader.batch_size
        iters_per_epoch = train_dataset_size // batch_size
        if train_dataset_size % batch_size != 0:
            iters_per_epoch += 1
        self.progress_iter = sly.Progress("Iterations", iters_per_epoch)
        
        self.start_iter = runner.iter
        
        fields = [
            {"field": "state.isValidation", "payload": False},
        ]
        add_progress_to_request(fields, "Epoch", self.progress_epoch)
        add_progress_to_request(fields, "Iter", self.progress_iter)
        g.api.app.set_fields(g.task_id, fields)

    def before_train_epoch(self, runner) -> None:
        super(SuperviselyLoggerHook, self).before_train_epoch(runner)
        
        self.progress_iter.set_current_value(0)
        fields = [
            {"field": "state.isValidation", "payload": False},
        ]
        add_progress_to_request(fields, "Iter", self.progress_iter)
        g.api.app.set_fields(g.task_id, fields)

    def before_val_epoch(self, runner) -> None:
        super(SuperviselyLoggerHook, self).before_val_epoch(runner)
        
        self.progress_iter.set_current_value(0)
        fields = [
            {"field": "state.isValidation", "payload": True},
        ]
        add_progress_to_request(fields, "Iter", self.progress_iter)
        g.api.app.set_fields(g.task_id, fields)

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        super(SuperviselyLoggerHook, self).after_train_iter(runner, batch_idx, data_batch, outputs)
        if self.every_n_inner_iters(batch_idx, self.interval):
            log_dict = runner.log_processor.get_log_after_iter(runner, batch_idx, 'train')
            self._log_info(log_dict, runner, batch_idx, "train")

    def after_val_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        super(SuperviselyLoggerHook, self).after_val_iter(runner, batch_idx, data_batch, outputs)
        if self.every_n_inner_iters(batch_idx, self.interval):
            log_dict = runner.log_processor.get_log_after_iter(runner, batch_idx, 'val')
            self._log_info(log_dict, runner, batch_idx, "val")

    def _update_progress(self, log_dict, runner):
        fields = []
        if log_dict['mode'] == 'val':
            self.progress_epoch.set_current_value(log_dict["epoch"])
            self.progress_iter.set_current_value(0)
        else:
            current_iter = (log_dict['iter'] - 1) % len(runner.train_dataloader)
            self.progress_iter.set_current_value(current_iter)
            if 'time' in log_dict.keys():
                self.time_sec_tot += log_dict['time']
                time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                fields.append({"field": "data.eta", "payload": eta_str})

        fields.append({"field": "state.isValidation", "payload": log_dict['mode'] == 'val'})
        add_progress_to_request(fields, "Epoch", self.progress_epoch)
        add_progress_to_request(fields, "Iter", self.progress_iter)
        return fields

    def _update_charts(self, log_dict, fields):
        if log_dict['mode'] == 'train':
            fields.extend([
                {"field": "data.chartLR.series[0].data", "payload": [[log_dict["epoch"], round(log_dict["lr"], 6)]], "append": True},
                {"field": "data.chartTrainLoss.series[0].data", "payload": [[log_dict["epoch"], log_dict["loss"]]], "append": True},
            ])
            self._lrs.append(log_dict["lr"])
            fields.append({
                "field": "data.chartLR.options.yaxisInterval",
                "payload": [
                    round(min(self._lrs) - min(self._lrs) / 10.0, 5),
                    round(max(self._lrs) + max(self._lrs) / 10.0, 5)
                ]
            })
            if 'time' in log_dict.keys():
                fields.extend([
                    {"field": "data.chartTime.series[0].data", "payload": [[log_dict["epoch"], log_dict["time"]]], "append": True},
                    {"field": "data.chartDataTime.series[0].data", "payload": [[log_dict["epoch"], log_dict["data_time"]]], "append": True},
                ])
                if torch.cuda.is_available():
                    fields.extend([{"field": "data.chartMemory.series[0].data", "payload": [[log_dict["epoch"], log_dict["memory"]]], "append": True}])
                    
        if log_dict['mode'] == 'val':
            fields.extend([
                {"field": "data.chartValMetrics.series[0].data", "payload": [[log_dict["epoch"], log_dict["precision"]]], "append": True},
                {"field": "data.chartValMetrics.series[1].data", "payload": [[log_dict["epoch"], log_dict["recall"]]], "append": True},
                {"field": "data.chartValMetrics.series[2].data", "payload": [[log_dict["epoch"], log_dict["f1-score"]]], "append": True}
            ])

    def _log_info(self, log_dict, runner, batch_idx, mode):
        if isinstance(log_dict, tuple):
            log_dict, log_str = runner.log_processor.get_log_after_iter(runner, batch_idx, mode)
        
        log_dict['max_iters'] = runner.max_iters
        log_dict['mode'] = mode
                
        fields = self._update_progress(log_dict, runner)
        if mode == 'val':
            mean_val_prediction = self._cal_val_metrics(runner.val_evaluator.metrics)
            log_dict.update(mean_val_prediction)
            
        self._update_charts(log_dict, fields)
        g.api.app.set_fields(g.task_id, fields)


    def _cal_val_metrics(self, metrics):
        results = []
        for batch in metrics:
            batch_res = batch.compute_metrics(batch.results)
            results.append(batch_res)
            
        res_dict = {}
        for key in ["precision", "recall", "f1-score"]: # SingleLabelMetrics/MultiLabelMetrics
            res_dict[key] = sum([res[key] for res in results]) / len(results)
        return res_dict