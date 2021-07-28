from gnn import train_epoch, test
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import torch
import copy

class Trainer():
    def __init__(self, model, optimizer, device, name, load_state_path=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.name = name
        self.basePath = f"./output/{name}"

        Path(self.basePath).mkdir(parents=True, exist_ok=True)

        if load_state_path is not None:
            self.model.load_state_dict(torch.load(load_state_path))

    def train(self, data, epochs=101):
        self.best_val_auc = self.test_auc = 0
        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.train_aucs = []
        self.valid_aucs = []
        self.test_aucs = []

        best_model_state = None

        for epoch in range(1, epochs):
            loss, train_auc = train_epoch(self.model, self.optimizer, self.device, data)
            val_loss, val_auc, test_loss, tmp_test_auc = test(self.model, self.device, data)
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.test_auc = tmp_test_auc
                # save state of best model
                best_model_state = copy.deepcopy(self.model.state_dict())

            self.train_losses.append(loss.item())
            self.train_aucs.append(train_auc)
            self.valid_losses.append(val_loss.item())
            self.valid_aucs.append(val_auc)
            self.test_losses.append(test_loss.item())
            self.test_aucs.append(tmp_test_auc)

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_auc:.4f}, Val: {val_auc:.4f}, '
                f'Test: {tmp_test_auc:.4f}')

        
        self.plotLearningCurves()
        
        # save model
        torch.save(best_model_state, os.path.join(self.basePath, "best.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.basePath, "final.pth"))

        print(f"best val auc: {self.best_val_auc}, test auc: {self.test_auc}")

    def plotLearningCurves(self):
        self.plotLearningCurve("loss", self.train_losses, self.valid_losses, self.test_losses)
        self.plotLearningCurve("AUC", self.train_aucs, self.valid_aucs, self.test_aucs)

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

