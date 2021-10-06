import torch

def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel, Pix2PixHDTransferModel, R2LTransfer
        if opt.isTrain:
            if opt.wgan:
                model = R2LTransfer()
            elif opt.transfer:
                model = Pix2PixHDTransferModel()
            else:
                model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    import ipdb
    ipdb.set_trace()
    return model


def create_UDA_model(opt):
    from .udaModel import R2LImageDiscriminator, R2LAE
    if opt.training_module == 'encoder':
        model = R2LAE()
    elif opt.training_module == 'decoder':
        model = R2LImageDiscriminator()
    elif opt.training_module == 'discriminator':
        model = R2LImageDiscriminator()
    else:
        assert('undefined training module: ', opt.training_module)
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model