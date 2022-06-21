from poptorch.optim import SGD, RMSprop, AdamW


def get_optimizer(args, model):
    regularized_params = []
    non_regularized_params = []

    # Filter biases and norm parameters.
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 1:
                non_regularized_params.append(param)
            else:
                regularized_params.append(param)

    params = [
        {'params': regularized_params, 'weight_decay': args.TRAIN.WEIGHT_DECAY},
        {'params': non_regularized_params, 'weight_decay': 0}
    ]

    optimizer = None
    # if args.TRAIN.OPTIMIZER.NAME == 'sgd':
    #     optimizer = SGD(params, lr=args.lr, momentum=args.momentum, loss_scaling=args.initial_loss_scaling, use_combined_accum=False)
    # elif args.TRAIN.OPTIMIZER.NAME == 'sgd_combined':
    #     optimizer = SGD(params, lr=args.lr, momentum=args.momentum, loss_scaling=args.initial_loss_scaling, velocity_scaling=args.initial_loss_scaling, use_combined_accum=True)
    # elif args.TRAIN.OPTIMIZER.NAME == 'adamw':
    #     ##TODO fix the loss_scaling
    #     optimizer = AdamW(params, lr=args.TRAIN.BASE_LR, loss_scaling=65536, eps=args.TRAIN.OPTIMIZER.EPS)
    # elif args.TRAIN.OPTIMIZER.NAME == 'rmsprop':
    #     optimizer = RMSprop(params, lr=args.lr, alpha=args.rmsprop_decay, momentum=args.momentum, loss_scaling=args.initial_loss_scaling, eps=args.optimizer_eps)
    # elif args.TRAIN.OPTIMIZER.NAME == 'rmsprop_tf':
    #     optimizer = RMSprop(params, lr=args.lr, alpha=args.rmsprop_decay, momentum=args.momentum, loss_scaling=args.initial_loss_scaling, eps=args.optimizer_eps, use_tf_variant=True)
    
    if args.TRAIN.OPTIMIZER.NAME == 'adamw':
        ##TODO fix the loss_scaling, hard code the loss scaling factor to 128
        optimizer = AdamW(params, lr=args.TRAIN.BASE_LR, loss_scaling=128, eps=args.TRAIN.OPTIMIZER.EPS)
    
    return optimizer
