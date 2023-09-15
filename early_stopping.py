import numpy as np
from src import utils

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_made = False
        self.val_loss_min = np.Inf
        self.delta = 0
        self.trace_func = print

             
        
    def __call__(self, val_loss, train_batch_losses, val_batch_loss, test_batch_loss, train_batch_acc, val_batch_acc, test_batch_acc, train_batch_auc, val_batch_auc, test_batch_auc, train_auc_indiv_ave, val_auc_indiv_ave, test_auc_indiv_ave, train_batch_prec, val_batch_prec, test_batch_prec, train_prec_indiv_ave, val_prec_indiv_ave, test_prec_indiv_ave, train_batch_recall, val_batch_recall, test_batch_recall, train_recall_indiv_ave, val_recall_indiv_ave, test_recall_indiv_ave, train_batch_f1, val_batch_f1, test_batch_f1, train_f1_indiv_ave, val_f1_indiv_ave, test_f1_indiv_ave):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.checkpoint_made =True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
            self.checkpoint_made =False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0
            self.checkpoint_made =True
        
    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        self.val_loss_min = val_loss # replace the lowest loss value with the new lowest loss value

    def print_checkpoint_metric(self, train_batch_losses, val_batch_loss, test_batch_loss, train_batch_acc, val_batch_acc, test_batch_acc, train_batch_auc, val_batch_auc, test_batch_auc, train_auc_indiv_ave, val_auc_indiv_ave, test_auc_indiv_ave, train_batch_prec, val_batch_prec, test_batch_prec, train_prec_indiv_ave, val_prec_indiv_ave, test_prec_indiv_ave, train_batch_recall, val_batch_recall, test_batch_recall, train_recall_indiv_ave, val_recall_indiv_ave, test_recall_indiv_ave, train_batch_f1, val_batch_f1, test_batch_f1, train_f1_indiv_ave, val_f1_indiv_ave, test_f1_indiv_ave, y_batch_test, test_logits):
        
        checkpoint_train_loss = np.mean(train_batch_losses)
        checkpoint_val_loss = np.mean(val_batch_loss)
        checkpoint_test_loss = np.mean(test_batch_loss)
        checkpoint_train_acc = np.mean(train_batch_acc)
        checkpoint_val_acc = np.mean(val_batch_acc)
        checkpoint_test_acc = np.mean(test_batch_acc)
        
        checkpoint_train_auc = np.mean(train_batch_auc)
        checkpoint_val_auc = np.mean(val_batch_auc)
        checkpoint_test_auc = np.mean(test_batch_auc)
        
        checkpoint_train_auc_indiv = train_auc_indiv_ave
        checkpoint_val_auc_indiv = val_auc_indiv_ave
        checkpoint_test_auc_indiv = test_auc_indiv_ave
        
        checkpoint_train_prec = np.mean(train_batch_prec)
        checkpoint_val_prec = np.mean(val_batch_prec)
        checkpoint_test_prec = np.mean(test_batch_prec)
        
        checkpoint_train_prec_indiv = train_prec_indiv_ave
        checkpoint_val_prec_indiv = val_prec_indiv_ave
        checkpoint_test_prec_indiv = test_prec_indiv_ave
        
        checkpoint_train_recall = np.mean(train_batch_recall)
        checkpoint_val_recall = np.mean(val_batch_recall)
        checkpoint_test_recall = np.mean(test_batch_recall)
        
        checkpoint_train_recall_indiv = train_recall_indiv_ave
        checkpoint_val_recall_indiv = val_recall_indiv_ave
        checkpoint_test_recall_indiv = test_recall_indiv_ave
        
        checkpoint_train_f1 = np.mean(train_batch_f1)
        checkpoint_val_f1 = np.mean(val_batch_f1)
        checkpoint_test_f1 = np.mean(test_batch_f1)
        
        checkpoint_train_f1_indiv = train_f1_indiv_ave
        checkpoint_val_f1_indiv = val_f1_indiv_ave
        checkpoint_test_f1_indiv = test_f1_indiv_ave
        
        checkpoint_y_test = y_batch_test
        checkpoint_logits_test = test_logits
        
        return checkpoint_train_loss, checkpoint_val_loss, checkpoint_test_loss, checkpoint_train_acc, checkpoint_val_acc,  checkpoint_test_acc, checkpoint_train_auc, checkpoint_val_auc,  checkpoint_test_auc, checkpoint_train_auc_indiv, checkpoint_val_auc_indiv, checkpoint_test_auc_indiv, checkpoint_train_prec, checkpoint_val_prec, checkpoint_test_prec, checkpoint_train_prec_indiv, checkpoint_val_prec_indiv, checkpoint_test_prec_indiv, checkpoint_train_recall, checkpoint_val_recall, checkpoint_test_recall, checkpoint_train_recall_indiv, checkpoint_val_recall_indiv, checkpoint_test_recall_indiv, checkpoint_train_f1, checkpoint_val_f1, checkpoint_test_f1, checkpoint_train_f1_indiv, checkpoint_val_f1_indiv, checkpoint_test_f1_indiv, checkpoint_y_test, checkpoint_logits_test
