import torch
import poptorch

def convert_to_ipu_model(model, opts, optimizer=None, traininig=True):
    if  traininig:
        _model = poptorch.trainingModel(model, opts, optimizer=optimizer)
        return _model
    else:
        _model = poptorch.inferenceModel(model, opts)
        return _model


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, data, target=None):
        if target is None:
            return out
        out = self.model(data)
        loss = self.loss(out, target)
        return out, loss
