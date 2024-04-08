import sys
import torch
import numpy as np

from . import Filler
from einops import rearrange


class GCNCycVirtualFiller(Filler):
    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scaled_target=False,
                 whiten_prob=0.05,
                 pred_loss_weight=1.,
                 warm_up=0,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None):
        super(GCNCycVirtualFiller, self).__init__(model_class=model_class,
                                                  model_kwargs=model_kwargs,
                                                  optim_class=optim_class,
                                                  optim_kwargs=optim_kwargs,
                                                  loss_fn=loss_fn,
                                                  scaled_target=scaled_target,
                                                  whiten_prob=whiten_prob,
                                                  metrics=metrics,
                                                  scheduler_class=scheduler_class,
                                                  scheduler_kwargs=scheduler_kwargs)

        self.tradeoff = pred_loss_weight
        self.trimming = (warm_up, warm_up)

        self.known_set = None

    def trim_seq(self, *seq):
        seq = [s[:, self.trimming[0]:s.size(1) - self.trimming[1]] for s in seq]
        if len(seq) == 1:
            return seq[0]
        return seq

    def training_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # To make the model inductive
        # => remove unobserved entries from input data and adjacency matrix
        if self.known_set is None:
            # Get observed entries (nonzero masks across time)
            mask = batch_data["mask"]
            mask = rearrange(mask, "b s n 1 -> (b s) n")
            mask_sum = mask.sum(0)  # n
            known_set = torch.where(mask_sum > 0)[0].detach().cpu().numpy().tolist()
            ratio = float(len(known_set) / mask_sum.shape[0])
            self.ratio = ratio
        else:
            known_set = self.known_set

        batch_data["known_set"] = known_set

        x = batch_data["x"]
        mask = batch_data["mask"]
        y = batch_data.pop("y")
        _ = batch_data.pop("eval_mask")  # drop this, we will re-create a new eval_mask (=mask during training)

        x = x[:, :, known_set, :]  # b s n1 d, n1 = num of observed entries
        mask = mask[:, :, known_set, :]  # b s n1 d
        y = y[:, :, known_set, :]  # b s n1 d

        b, s, n, d = mask.size()

        dynamic_ratio = self.ratio + 0.2 * np.random.random()  # ratio + 0.1
        cur_entry_num = n  # n1
        aug_entry_num = int(cur_entry_num / dynamic_ratio)
        sub_entry_num = aug_entry_num - cur_entry_num  # n2 - n1
        assert sub_entry_num > 0, "The augmented data should have more entries than original data."
        self.sub_entry_num = sub_entry_num
        batch_data["reset"] = True

        sub_entry = torch.zeros(b, s, sub_entry_num, d).to(x.device)
        x = torch.cat([x, sub_entry], dim=2)  # b s n2 d
        mask = torch.cat([mask, sub_entry], dim=2).byte()  # b s n2 d
        y = torch.cat([y, sub_entry], dim=2)  # b s n2 d

        eval_mask = mask  # eval_mask = mask, during training

        batch_data["x"] = x  # b s n2 d
        batch_data["mask"] = mask  # b s n' 1
        batch_data["sub_entry_num"] = sub_entry_num  # number

        # Compute predictions and compute loss
        res = self.predict_batch(batch, preprocess=False, postprocess=False)
        imputation, imputation_cyc, target_cyc = res[0], res[1], res[2]

        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)
        imputation_cyc, target_cyc = self.trim_seq(imputation_cyc, target_cyc)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
            imputation_cyc = self._postprocess(imputation_cyc, batch_preprocessing)

        # partial loss + cycle loss
        loss = self.loss_fn(imputation, target, mask) + \
               1 * self.loss_fn(imputation_cyc, target_cyc, torch.ones_like(imputation_cyc).bool())

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.train_metrics.update(imputation.detach(), y, eval_mask)  # all unseen data
        # self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        mask = batch_data.get('mask')
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        # Compute predictions and compute loss
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False)

        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)

        val_loss = self.loss_fn(imputation, target, eval_mask)

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.val_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        # Compute outputs and rescale
        imputation = self.predict_batch(batch, preprocess=False, postprocess=True)
        test_loss = self.loss_fn(imputation, y, eval_mask)

        # Logging
        self.test_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_loss', test_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return test_loss
