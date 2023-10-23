import torch
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Mapping, Union, Any, Optional


class BaseTrainer(ABC):
    """
    Base class for all trainers
    """

    def train_epoch(self, log_env: Optional[Dict] = None) -> float:
        train_info = self.on_train_start()
        self.train_mode_on()
        for step, batch in enumerate(self.train_dataloader):
            epoch_info = self.training_step(batch, log_env)
            self.on_epoch_end(train_info=train_info, epoch_info=epoch_info)

        self.on_train_end(train_info)

        return train_info

    def on_train_start(self):
        train_info = {"total_train_loss": 0}
        return train_info

    def on_train_end(self, info: Dict, verbose: Optional[bool] = True):
        info["total_train_loss"] /= len(self.train_dataloader)
        if verbose:
            print(f"\tTrain loss discriminator: {info['total_train_loss']:.3f}")

    def on_epoch_end(self, **kwargs):
        kwargs["train_info"]["total_train_loss"] += kwargs["epoch_info"]["loss"]

    @abstractmethod
    def train_mode_on(self):
        """
        Training step logic
        """
        return NotImplementedError

    @abstractmethod
    def training_step(self):
        """
        Training step logic
        """
        return NotImplementedError

    @torch.no_grad()
    def validation(self, verbose: Optional[bool] = True, log_env: Optional[Dict] = None, **kwargs) -> Dict:

        result_metrics = self.predict(
            model=self.model, data_loader=self.valid_dataloader, label_names=self.config["label_names"]
        )
        self._valid_logging(log_env, info=result_metrics)

        self.on_valid_end(result_metrics, verbose)
        return result_metrics

    @staticmethod
    def on_valid_end(info: Dict, verbose: Optional[bool] = True):
        if verbose:
            print(f"\tTest loss discriminator: {info['loss']:.3f}")
            print(f"\tTest accuracy discriminator: {info['overall_accuracy']:.3f}")
            print(f"\tTest f1 discriminator: {info['overall_f1']:.3f}")

    @abstractmethod
    def predict(self):
        """
        Model predict logic
        """
        return NotImplementedError

    @abstractmethod
    def _train_logging(self):
        """
        Logging logic on training stage
        """
        return NotImplementedError

    @abstractmethod
    def _valid_logging(self):
        """
        Logging logic on validation stage
        """
        return NotImplementedError

    def _prepare_inputs(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_inputs(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_inputs(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.device)
            return data.to(**kwargs)
        return data
