import math
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
# pytorch
import torch

# Custom Trainer class
# Train and make prediction with the GNN models
class Trainer:
    def __init__(self, model, optimizer, train_loader, valid_loader):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    # training model
    def train_one_epoch(self, epoch):
        # set model on training mode
        self.model.train()

        t_targets = []; p_targets = []; losses = []
        tqdm_iter = tqdm(self.train_loader, total=len(self.train_loader))
        for i, data in enumerate(tqdm_iter):

            tqdm_iter.set_description(f"Epoch {epoch}")
            self.optimizer.zero_grad()
            outputs, loss = self.model(data, data.edge_index, data.batch)
            targets = data.y
            loss.backward()
            self.optimizer.step()

            y_true = self.process_output(targets)  # for one batch
            y_proba = self.process_output(outputs.flatten()) # for one batch

            auc = roc_auc_score(y_true, y_proba)
            # continuous loss/auc update
            tqdm_iter.set_postfix(train_loss=round(loss.item(), 2), train_auc=round(auc, 2), 
                                  valid_loss=None, valid_auc=None)
            
            losses.append(loss.item())
            t_targets.extend(list(y_true))
            p_targets.extend(list(y_proba))

        epoch_auc = roc_auc_score(t_targets, p_targets)
        epoch_loss = sum(losses)/len(losses)
        return epoch_loss, epoch_auc, tqdm_iter


    def process_output(self, out):
        out = out.cpu().detach().numpy()
        return out

    
    def validate_one_epoch(self, progress):
        
        progress_tracker = progress["tracker"]
        train_loss = progress["loss"]
        train_auc = progress["auc"]
        
        # model in eval model
        self.model.eval()
        
        t_targets = []; p_targets = []; losses = []
        for data in self.valid_loader:
            
            outputs, loss = self.model(data, data.edge_index, data.batch)
            outputs, targets = outputs.flatten(), data.y
            
            y_proba = self.process_output(outputs)  # for one batch
            y_true = self.process_output(targets) # for one batch 
            
            t_targets.extend(list(y_true))
            p_targets.extend(list(y_proba))
            losses.append(loss.item())
        
        epoch_auc = roc_auc_score(t_targets, p_targets)
        epoch_loss = sum(losses)/len(losses)
        progress_tracker.set_postfix(train_loss=round(train_loss, 2), train_auc=round(train_auc, 2), 
                                    valid_loss=round(epoch_loss, 2), valid_auc=round(epoch_auc, 2))              
        progress_tracker.close()
        return epoch_loss, epoch_auc
            
    # runs the training and validation trainer for n_epochs
    def run(self, n_epochs=10):
        
        train_scores = []; train_losses = []
        valid_scores = []; valid_losses = []
        for e in range(1, n_epochs+1):
            lt, at, progress_tracker = self.train_one_epoch(e)
            
            train_losses.append(lt)
            train_scores.append(at)
            
            # validate this epoch
            progress = {"tracker": progress_tracker, "loss": lt, "auc": at}  
            lv, av = self.validate_one_epoch(progress)  # pass training progress tracker to validation func
            valid_losses.append(lv)
            valid_scores.append(av)

        return (train_losses, train_scores), (valid_losses, valid_scores)
            
        
    def predict(self, test_loader):
        # set model on evaluation mode
        self.model.eval()
        predictions = []
        tqdm_iter = tqdm(test_loader, total=len(test_loader))
        for data in tqdm_iter:
            tqdm_iter.set_description(f"Making prediction")
            with torch.no_grad():
                o, _ = self.model(data, data.edge_index, data.batch)
                o = self.process_output(o.flatten())
                predictions.extend(list(o))
            tqdm_iter.set_postfix(stage="test dataloader")
        tqdm_iter.close()
        return np.array(predictions)