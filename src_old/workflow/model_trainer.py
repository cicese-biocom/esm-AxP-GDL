from typing import List

from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src_old.workflow.dto_workflow import ModelParameters
from src.modeling.loss_function import LossFunctionContext


class ModelTrainer:
    def __init__(self,
                 model: nn.Module,
                 train_graphs: List,
                 val_graphs: List,
                 batch_size,
                 loss: LossFunctionContext,
                 use_edge_attr,
                 learning_rate,
                 number_of_epochs,
                 metrics_dir,
                 model_parameters: ModelParameters,
                 weight_decay=5e-4,
                 step_size=5,
                 gamma=0.9,
                 device='cpu'
                 ):
        # model
        self.model = model

        # training data
        self.train_dataloader = DataLoader(
            dataset=train_graphs,
            batch_size=batch_size
        )

        # test data
        self.val_dataloader = DataLoader(
            dataset=val_graphs,
            batch_size=batch_size
        )

        # loss
        self.loss = loss.build()

        # optimizer
        self.optimizer = Adam(
            model.parameters(),
            learning_rate,
            weight_decay=weight_decay
        )

        # scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=gamma
        )

        # number_of_epochs
        self.number_of_epochs = number_of_epochs

        # device
        self.device = device

        # use_edge_attr
        self.use_edge_attr = use_edge_attr

        # metrics tensorboardX
        self.writer = SummaryWriter(
            log_dir=metrics_dir,
            filename_suffix="_metrics"
        )

        # model parameters
        self.model_parameters = model_parameters

        # batch evaluator
        self.model_runner = ModelRunner(
            self.model,
            self.use_edge_attr,
            self.device
        )


    def run(self):
        best_mcc = -2
        current_model = {}
        model_with_best_mcc = {}
        metrics_data = []

        bar = tqdm(
            total=self.number_of_epochs,
            desc="Training and Validation:"
        )

        for epoch in range(1, self.number_of_epochs + 1):
            self._run_epoch()

            checkpoint_data = self._get_checkpoint_data(epoch)

            # eval model


    def _run_epoch(self):
        arr_loss = []

        self.model.train()
        for batch in self.train_dataloader:
            self.optimizer.zero_grad()

            out = self.model_runner.predict(batch)

            out_loss = self.loss(out, batch.y)
            arr_loss.append(out_loss.item())

            self.loss.backward()
            self.optimizer.step()

    def _get_checkpoint_data(self, epoch):
        return {
            'epoch': epoch,
            'model': self.model,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'parameters': self.model_parameters.model_dump()
        }
