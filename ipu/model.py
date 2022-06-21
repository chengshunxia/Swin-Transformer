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
        poptorch.BeginBlock(self.loss, ipu_id = 7)

    def forward(self, data, target=None):
        if target is None:
            return out
        out = self.model(data)
        loss = self.loss(out, target)
        return out, loss


def pipeline_model(model):
    # for base and large 

    
    ## DEPTHS: [ 2, 2, 18, 2 ]
    ## NUM_HEADS: [ 4, 8, 16, 32 ]

    # depths = config.MODEL.SWIN.DEPTHS
    # pipeline_split_for_swin_base = [2,2,18,2]
    # pipeline_split_for_swin_large = [2,4,4,4,4,4,1,1]


    # patch_embed
    # if model.ape:
    #     model.absolute_pos_embed
    # model.pos_drop
    # for layer in self.layers:
    #     x = layer(x)
    # self.norm
    # self.avgpool

    # x = self.flatten(x, 1)
    # return x
    # x = self.head(x)
    # return x

    poptorch.BeginBlock(model.patch_embed, ipu_id = 0)
    if model.ape:
        poptorch.BeginBlock(model.absolute_pos_embed, ipu_id = 0)
    poptorch.BeginBlock(model.pos_drop, ipu_id = 0)
    # import pdb
    # pdb.set_trace()

    poptorch.BeginBlock(model.layers[0].blocks[0], ipu_id = 0)
    poptorch.BeginBlock(model.layers[0].blocks[1], ipu_id = 0)
    poptorch.BeginBlock(model.layers[0].downsample, ipu_id = 0)

    poptorch.BeginBlock(model.layers[1].blocks[0], ipu_id = 1)
    poptorch.BeginBlock(model.layers[1].blocks[1], ipu_id = 1)
    poptorch.BeginBlock(model.layers[1].downsample, ipu_id = 1)
    poptorch.BeginBlock(model.layers[2].blocks[0], ipu_id = 1)
    poptorch.BeginBlock(model.layers[2].blocks[1], ipu_id = 1)

    poptorch.BeginBlock(model.layers[2].blocks[2], ipu_id = 2)
    poptorch.BeginBlock(model.layers[2].blocks[3], ipu_id = 2)
    poptorch.BeginBlock(model.layers[2].blocks[4], ipu_id = 2)
    poptorch.BeginBlock(model.layers[2].blocks[5], ipu_id = 2)


    poptorch.BeginBlock(model.layers[2].blocks[6], ipu_id = 3)
    poptorch.BeginBlock(model.layers[2].blocks[7], ipu_id = 3)
    poptorch.BeginBlock(model.layers[2].blocks[8], ipu_id = 3)
    poptorch.BeginBlock(model.layers[2].blocks[9], ipu_id = 3)

    poptorch.BeginBlock(model.layers[2].blocks[10], ipu_id = 4)
    poptorch.BeginBlock(model.layers[2].blocks[11], ipu_id = 4)
    poptorch.BeginBlock(model.layers[2].blocks[12], ipu_id = 4)
    poptorch.BeginBlock(model.layers[2].blocks[13], ipu_id = 4)

    poptorch.BeginBlock(model.layers[2].blocks[14], ipu_id = 5)
    poptorch.BeginBlock(model.layers[2].blocks[15], ipu_id = 5)
    poptorch.BeginBlock(model.layers[2].blocks[16], ipu_id = 5)
    poptorch.BeginBlock(model.layers[2].blocks[17], ipu_id = 5)
    poptorch.BeginBlock(model.layers[2].downsample, ipu_id = 5)
  
    poptorch.BeginBlock(model.layers[3].blocks[0], ipu_id = 6)

    poptorch.BeginBlock(model.layers[3].blocks[1], ipu_id = 7)

    poptorch.BeginBlock(model.norm, ipu_id = 7)
    poptorch.BeginBlock(model.avgpool, ipu_id = 7)
    poptorch.BeginBlock(model.flatten, ipu_id = 7)
    poptorch.BeginBlock(model.head, ipu_id = 7)
