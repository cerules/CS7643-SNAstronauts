from gnn import train_epoch, test
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import torch
import copy
from tqdm import tqdm

class Trainer():
    def __init__(self, model, optimizer, scheduler, device, name, load_state_path=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.name = name
        self.basePath = f"./output/{name}"

        Path(self.basePath).mkdir(parents=True, exist_ok=True)

        if load_state_path is not None:
            self.model.load_state_dict(torch.load(load_state_path))

    def train(self, data, epochs=101):
        self.best_val_acc = 0
        self.best_test_acc = 0

        self.train_losses = []
        self.train_accs = []
        self.valid_losses = []
        self.valid_accs = []
        self.test_losses = []
        self.train_aucs = []
        self.valid_aucs = []
        self.test_aucs = []
        self.test_accs = []

        best_model_state = None
        
        pbar = tqdm(total=epochs)
        for epoch in range(1, epochs):
            loss, train_auc, train_acc = train_epoch(self.model, self.optimizer, self.device, data)
            val_loss, val_auc, val_acc, test_loss, test_auc, test_acc = test(self.model, self.device, data)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_test_acc = test_acc
                # save state of best model
                best_model_state = copy.deepcopy(self.model.state_dict())

            self.train_losses.append(loss.item())
            self.train_aucs.append(train_auc)
            self.train_accs.append(train_acc)
            self.valid_losses.append(val_loss.item())
            self.valid_aucs.append(val_auc)
            self.valid_accs.append(val_acc)
            self.test_losses.append(test_loss.item())
            self.test_aucs.append(test_auc)
            self.test_accs.append(test_acc)

            self.scheduler.step(val_loss)

            pbar.set_description(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                f'Test: {test_acc:.4f}')

            pbar.update(1)
        pbar.close()
        self.plotLearningCurves()

        # save model
        torch.save(best_model_state, os.path.join(self.basePath, "best.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.basePath, "final.pth"))

        # save loss arrays
        self._save_loss_arrays(self.train_losses, self.train_aucs, self.train_accs, 
                                self.valid_losses, self.valid_aucs, self.valid_accs,
                                self.test_losses, self.test_aucs, self.test_accs)

        print(f"best val acc: {self.best_val_acc}, test acc: {self.best_test_acc}")

    def plotLearningCurves(self):
        self.plotLearningCurve("loss", self.train_losses, self.valid_losses, self.test_losses)
        self.plotLearningCurve("AUC", self.train_aucs, self.valid_aucs, self.test_aucs)
        self.plotLearningCurve("accuracy", train=self.train_accs, valid=self.valid_accs, test=self.test_accs)

    def plotLearningCurve(self, title, train, valid, test):

        train = np.array(train)
        valid = np.array(valid)
        test = np.array(test)
        epochs = np.array(range(len(train)))
        fig, ax = plt.subplots()

        ax.plot(epochs, train, label=f"train {title}")
        ax.plot(epochs, valid, label=f"valid {title}")
        ax.plot(epochs, test, label=f"test {title}")

        ax.set_title(f"{title.capitalize()} per Epoch")
        ax.set_xlabel("epochs")
        ax.set_ylabel(title)
        ax.legend(loc="best")

        fig.savefig(f"./output/{self.name}/{title}.png")
        plt.show()

    def _save_loss_arrays(self, train_losses, train_aucs, train_accs, valid_losses, valid_aucs, valid_accs, test_losses, test_aucs, test_accs):
        np.save('./output/train_losses.npy', (train_losses))
        np.save('./output/train_aucs.npy', (train_aucs))
        np.save('./output/train_accs.npy', (train_accs))
        np.save('./output/valid_losses.npy', (valid_losses))
        np.save('./output/valid_aucs.npy', (valid_aucs))
        np.save('./output/valid_accs.npy', (valid_accs))
        np.save('./output/test_losses.npy', (test_losses))
        np.save('./output/test_aucs.npy', (test_aucs))
        np.save('./output/test_aucs.npy', (test_accs))

