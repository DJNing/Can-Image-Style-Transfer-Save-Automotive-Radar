from torch.autograd import Variable
import torch
from data import ImageTransform
from model.engine.utils import transform_image, optimize, optimize_new, saliency
from util.logger import setup_logger


logger = setup_logger('style-transfer', False)


def do_transfer_style(
        cfg,
        model,
        content_image,
        style_image,
        device,
        content_only=False,
        style_only=False,
        opt='LBFGS', 
        saliency_map=False
):
    logger.info("Start transferring.")

    image_transformer = ImageTransform(cfg.DATA.IMG_SIZE, cfg.DATA.IMAGENET_MEAN)

    # transform images
    content_image = transform_image(image_transformer, content_image, device)
    style_image = transform_image(image_transformer, style_image, device)
    optimized_image = Variable(content_image.data.clone(), requires_grad=True)
    # optimized_image = Variable(torch.zeros_like(content_image), requires_grad=True)
    # optimize_image = Variable(torch.randn(content_image.size()).type_as(content_image.data),
    #                           requires_grad=True)

#     optimized_image = optimize(model, content_image, style_image, optimized_image, cfg, cfg.LOSS.MAX_ITER)
    if saliency_map:
        optimized_image = saliency(model, content_image, style_image, optimized_image, 
                                        cfg, cfg.LOSS.MAX_ITER, content_only, style_only, opt)
    else:
        optimized_image = optimize_new(model, content_image, style_image, optimized_image, 
                                        cfg, cfg.LOSS.MAX_ITER, content_only, style_only, opt)

    out_image = image_transformer.post_preparation(optimized_image.data[0].cpu().squeeze())
    out_image.save(cfg.OUTPUT.DIR + cfg.OUTPUT.FILE_NAME)
    return out_image
