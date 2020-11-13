import numpy as np
import torch
import os.path


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, model_name='unet', mask_name='gaussian2d', mask_perc=30,
                 verbose=False, delta=0, checkpoint_path='./checkpoint_dir', log_path='./log_dir',
                 log_all=False, log_eval=False, trace_func=print):

        self.patience = patience
        self.model_name = model_name
        self.mask_name = mask_name
        self.mask_perc = mask_perc
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.nmse = None
        self.early_stop = False
        self.val_nmse_min = np.Inf
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path
        self.log_all = log_all
        self.log_eval = log_eval
        self.trace_func = trace_func

    def __call__(self, nmse, model_g, model_d, epoch):

        # nmse(+ --> 0) score(- --> 0)
        score = -nmse
        # init score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(nmse, model_g, model_d, epoch)
        # new model is worse
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        # new model is better
        else:
            self.best_score = score
            self.save_checkpoint(nmse, model_g, model_d, epoch)
            self.counter = 0

        log = f'EarlyStopping counter of epoch {epoch + 1} : {self.counter} out of {self.patience}'
        # print(log)
        self.log_all.debug(log)


    def save_checkpoint(self, val_nmse, model_g, model_d, epoch):
        # save model when validation loss decrease.
        if self.verbose:
            log = f'Validation loss decreased ({self.val_nmse_min:.6f} --> {val_nmse:.6f}).  Saving model ...'
            # print(log)
            self.log_all.debug(log)
            self.log_eval.info(log)
        # update the min nmse
        self.val_nmse_min = val_nmse
        torch.save(model_g.state_dict(),
                   os.path.join(self.checkpoint_path,
                                "best_checkpoint_generator_{}_{}_{}_epoch_{}_nmse_{}.pt"
                                .format(self.model_name, self.mask_name, self.mask_perc, epoch + 1, self.val_nmse_min)))
        # torch.save(model_d.state_dict(),
        #            os.path.join(self.checkpoint_path, self.localtime,
        #                         "best_checkpoint_discriminator_{}_{}_{}_epoch_{}_nmse_{}.pt"
        #                         .format(self.model_name, self.mask_name, self.mask_perc, epoch + 1, self.val_nmse_min)))
