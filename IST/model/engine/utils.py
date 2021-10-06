from torch import optim
from torch.autograd import Variable
import torch
import torch.nn as nn
from model.meta_arch import GramMatrix
from util.logger import setup_logger
from model.meta_arch import GramMSELoss, StyleTransfer

logger = setup_logger('style-transfer', False)

def transform_image(image_transformer, image, device):
    image_transformed = image_transformer.preparation(image)
    image_transformed = Variable(image_transformed.unsqueeze(0).to(device))
    return image_transformed


def optimize(model, content_image, style_image, optimized_image, cfg, max_iterations):
    # compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in model.vgg_model(style_image, cfg.LOSS.STYLE_LAYERS)]
    content_targets = [A.detach() for A in model.vgg_model(content_image, cfg.LOSS.CONTENT_LAYERS)]
    targets = style_targets + content_targets

    # create optimizer
    optimizer = optim.LBFGS([optimized_image])

    log_show_iter = cfg.LOSS.LOG_ITER_SHOW * cfg.LOSS.MAX_ITER
    iterations = [0]
    while iterations[0] < max_iterations:
        def closure():
            optimizer.zero_grad()
            outputs = model.vgg_model(optimized_image, model.loss_layers)
            layer_losses = [model.loss_weights[a] * model.loss_functions[a](A, targets[a]) for a, A in
                            enumerate(outputs)]
            # print(layer_losses)
            loss = sum(layer_losses)
            loss.backward()
            iterations[0] += 1
            # if iterations[0] % log_show_iter == (log_show_iter - 1):
            #     logger.info('Iteration: %d, loss: %f, content loss: %f' % (iterations[0] + 1, loss.data, layer_losses[-1].data))

            return loss

        optimizer.step(closure)
        # break
    return optimized_image

def optimize_new(model, content_image, style_image, optimized_image, cfg, max_iterations, content_only=False, style_only=False, opt="LBFGS"):
    device = torch.device(cfg.MODEL.DEVICE)
    # content only saliency map
    if (content_only == False) & (style_only == False):
        return optimize(model, content_image, style_image, optimized_image, cfg, max_iterations)
    # style only saliency map
    elif content_only:

        content_targets = [A.detach() for A in model.vgg_model(content_image, cfg.LOSS.CONTENT_LAYERS)]
        targets = content_targets
        loss_layers = cfg.LOSS.CONTENT_LAYERS
        loss_functions = [nn.MSELoss()] * len(cfg.LOSS.CONTENT_LAYERS)
        loss_functions = [loss_function.to(device) for loss_function in loss_functions]

        # loss weights settings
        loss_weights = cfg.LOSS.CONTENT_WEIGHTS


    elif style_only:

        style_targets = [GramMatrix()(A).detach() for A in model.vgg_model(style_image, cfg.LOSS.STYLE_LAYERS)]
        targets = style_targets
        loss_layers = cfg.LOSS.STYLE_LAYERS
        loss_functions = [GramMSELoss()] * len(cfg.LOSS.STYLE_LAYERS)
        loss_functions = [loss_function.to(device) for loss_function in loss_functions]

        # loss weights settings
        loss_weights = cfg.LOSS.STYLE_WEIGHTS 
        # pass
    if opt == "LBFGS":
        optimizer = optim.LBFGS([optimized_image])
    else:
        optimizer = optim.Adam([optimized_image])

    # if for saliency map
    # saliency_target = [torch.zeros_like(a) for a in targets]

    log_show_iter = cfg.LOSS.LOG_ITER_SHOW * cfg.LOSS.MAX_ITER
    iterations = [0]
    while iterations[0] < max_iterations:
        def closure():
            optimizer.zero_grad()
            outputs = model.vgg_model(optimized_image, loss_layers)
            layer_losses = [loss_weights[a] * loss_functions[a](A, targets[a]) for a, A in
                            enumerate(outputs)]
            loss = - sum(layer_losses)
            loss.backward()
            iterations[0] += 1
            # if iterations[0] % log_show_iter == (log_show_iter - 1):
            #     logger.info('Iteration: %d, loss: %f, content loss: %f' % (iterations[0] + 1, loss.data, layer_losses[-1].data))

            return loss

        optimizer.step(closure)
        break
    return optimized_image

def saliency(model, content_image, style_image, optimized_image, cfg, max_iterations, content_only=False, style_only=False, opt="LBFGS"):
    device = torch.device(cfg.MODEL.DEVICE)
    # content only saliency map
    if (content_only == False) & (style_only == False):
        return optimize(model, content_image, style_image, optimized_image, cfg, max_iterations)
    # style only saliency map
    elif content_only:

        content_targets = [A.detach() for A in model.vgg_model(content_image, cfg.LOSS.CONTENT_LAYERS)]
        targets = content_targets
        loss_layers = cfg.LOSS.CONTENT_LAYERS
        loss_functions = [nn.MSELoss()] * len(cfg.LOSS.CONTENT_LAYERS)
        loss_functions = [loss_function.to(device) for loss_function in loss_functions]

        # loss weights settings
        loss_weights = cfg.LOSS.CONTENT_WEIGHTS


    elif style_only:

        style_targets = [GramMatrix()(A).detach() for A in model.vgg_model(style_image, cfg.LOSS.STYLE_LAYERS)]
        targets = style_targets
        loss_layers = cfg.LOSS.STYLE_LAYERS
        loss_functions = [GramMSELoss()] * len(cfg.LOSS.STYLE_LAYERS)
        loss_functions = [loss_function.to(device) for loss_function in loss_functions]

        # loss weights settings
        loss_weights = cfg.LOSS.STYLE_WEIGHTS 
        # pass
    if opt == "LBFGS":
        optimizer = optim.LBFGS([optimized_image])
    else:
        optimizer = optim.Adam([optimized_image])

    # if for saliency map
    # saliency_target = [torch.zeros_like(a) for a in targets]

    log_show_iter = cfg.LOSS.LOG_ITER_SHOW * cfg.LOSS.MAX_ITER
    iterations = [0]
    # while iterations[0] < max_iterations:
    #     def closure():
    optimizer.zero_grad()
    outputs = model.vgg_model(optimized_image, loss_layers)
    layer_losses = [loss_weights[a] * torch.sum(A) for a, A in
                    enumerate(outputs)]
    loss = - sum(layer_losses)
    loss.backward()
    iterations[0] += 1
    # if iterations[0] % log_show_iter == (log_show_iter - 1):
    #     logger.info('Iteration: %d, loss: %f, content loss: %f' % (iterations[0] + 1, loss.data, layer_losses[-1].data))

        # return loss

    # optimizer.step(closure)
        # break
    # optimized_image =  (1 / torch.max(optimized_image) - torch.min(optimized_image)) * (optimized_image - torch.min(optimized_image))

    return optimized_image.grad