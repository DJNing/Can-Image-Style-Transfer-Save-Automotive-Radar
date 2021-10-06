import torch.nn as nn
import torch.nn.functional as F
import torch
# from vgg import VGG
import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6, in_features=64):
        super(Generator, self).__init__()

        temp = in_features

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, in_features, 7),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        # in_features = 64
        out_features = in_features*2
        for _ in range(3):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(3):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(temp, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class GeneratorUnet(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6, in_features=64):
        super(GeneratorUnet, self).__init__()

        # temp = in_features

        # Initial convolution block       
        
        self.init_block = nn.Sequential(*[   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, in_features, 7),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True) ])

        # Downsampling
        # in_features = 64
        def down_block(ipc, opc):
            result = [  nn.Conv2d(ipc, opc, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(opc),
                        nn.ReLU(inplace=True) ]
            return nn.Sequential(*result)

        encoder = []
        out_features = in_features*2
        for _ in range(3):
            encoder += [down_block(in_features, out_features)]
            in_features = out_features
            out_features = in_features*2

        self.encoder = nn.ModuleList(encoder)

        # Residual blocks
        res_block = []
        for _ in range(n_residual_blocks):
            res_block += [ResidualBlock(in_features)]

        self.res_block = nn.Sequential(*res_block)

        # Upsampling
        def up_block(ipc, opc):
            result = [  nn.ConvTranspose2d(ipc, opc, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(opc),
                        nn.ReLU(inplace=True) ]
            return nn.Sequential(*result)

        decoder = []
        # for concatenation
        # in_features = in_features * 2
        out_features = in_features//2
        # print(in_features)
        for _ in range(3):
            # print(in_features)
            decoder += [up_block(in_features*2, out_features)]
            in_features = out_features
            out_features = in_features//2
        self.decoder = nn.ModuleList(decoder)

        # Output layer
        self.output_layer = nn.Sequential(*[  nn.ReflectionPad2d(3),
                    nn.Conv2d(out_features*2, output_nc, 7),
                    nn.Tanh() ])

        # self.model = nn.Sequential(*model)

    def forward(self, x):
        # return self.model(x)
        concat = None
        temp = self.init_block(x)
        for layer in self.encoder:
            if concat is None:
                concat = [layer(temp)]
            else:
                concat.append(layer(concat[-1]))
        temp = self.res_block(concat[-1])

        for cnt in range(len(self.decoder)):
            sub = cnt + 1
            idx = len(self.decoder) - sub
            ip_temp = torch.cat((temp, concat[idx]), dim=1)
            temp = self.decoder[cnt](ip_temp)
        
        temp = self.output_layer(temp)
        return temp


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
       
        for x in range(23):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        

    def forward(self, X):
        h_relu1 = self.slice1(X)              
        out = h_relu1
        return out


class contentLoss(nn.Module):
    def __init__(self):
        super(contentLoss, self).__init__()
        self.vgg = Vgg16().cuda()
        # self.encoder.to(torch.device('cuda'))
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        # ip = torch.cat((pred.expand([-1,3,-1,-1]), target.expand([-1,3,-1,-1])), dim=0)
        pred = pred.expand([-1,3,-1,-1])
        target = target.expand([-1,3,-1,-1])
        pred_vgg, target_vgg = self.vgg(pred), self.vgg(target)
        loss = self.loss(pred_vgg, target_vgg)
        return loss




class GeneratorMultiscale(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6, in_features=64):
        super(GeneratorMultiscale, self).__init__()

        # temp = in_features

        # Initial convolution block       
        
        self.init_block = nn.Sequential(*[   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, in_features, 7),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True) ])

        # Downsampling
        # in_features = 64
        def down_block(ipc, opc):
            result = [  nn.Conv2d(ipc, opc, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(opc),
                        nn.ReLU(inplace=True) ]
            return nn.Sequential(*result)

        encoder = []
        out_features = in_features*2
        for _ in range(3):
            encoder += [MultiAtrousConv(in_features, out_features, stride=2)]
            in_features = out_features
            out_features = in_features*2

        self.encoder = nn.ModuleList(encoder)

        # Residual blocks
        res_block = []
        for _ in range(n_residual_blocks):
            res_block += [ResidualBlock(in_features)]

        self.res_block = nn.Sequential(*res_block)


        decoder = []
        # for concatenation
        # in_features = in_features * 2
        out_features = in_features//2
        # print(in_features)
        for _ in range(3):
            # print(in_features)
            decoder += [MultiAtrousTransposeConv(in_features*2, out_features, stride=2)]
            in_features = out_features
            out_features = in_features//2
        self.decoder = nn.ModuleList(decoder)

        # Output layer
        self.output_layer = nn.Sequential(*[  nn.ReflectionPad2d(3),
                    nn.Conv2d(out_features*2, output_nc, 7),
                    nn.Tanh() ])

        # self.model = nn.Sequential(*model)

    def forward(self, x):
        # return self.model(x)

        concat = None
        temp = self.init_block(x)
        for layer in self.encoder:
            if concat is None:
                concat = [layer(temp)]
            else:
                concat.append(layer(concat[-1]))
        temp = self.res_block(concat[-1])

        for cnt in range(len(self.decoder)):
            sub = cnt + 1
            idx = len(self.decoder) - sub
            ip_temp = torch.cat((temp, concat[idx]), dim=1)
            temp = self.decoder[cnt](ip_temp)
        
        temp = self.output_layer(temp)
        return temp

class GeneratorMultiscaleDenseDecoder(GeneratorMultiscale):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6, in_features=64):
        super(GeneratorMultiscaleDenseDecoder, self).__init__(input_nc, output_nc, n_residual_blocks, in_features)

        out_features = in_features*2
        for _ in range(3):
            in_features = out_features
            out_features = in_features*2

        def up_block(ipc, opc):
            result = [  nn.ConvTranspose2d(ipc, opc, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(opc),
                        nn.ReLU(inplace=True) ]
            return nn.Sequential(*result)

        decoder = []
        # for concatenation
        # in_features = in_features * 2
        out_features = in_features//2
        # print(in_features)
        for _ in range(3):
            # print(in_features)
            decoder += [up_block(in_features*2, out_features)]
            in_features = out_features
            out_features = in_features//2
        self.decoder = nn.ModuleList(decoder)



class MultiAtrousConv(nn.Module):
    def __init__(self, ipc, opc, rate_list=[2,4,6,8], stride=1):
        super(MultiAtrousConv, self).__init__()

        if opc%4 != 0 :
            assert('opc must be n x 4')
        # branch_opc = int(opc/4)

        def branch_block(ipc, opc, k, s, p, d):
            layer_list = [
                nn.Conv2d(ipc, opc, k, s, int(p), d), 
                nn.InstanceNorm2d(opc),
                nn.ReLU()
            ]
            return nn.Sequential(*layer_list)

        self.b0 = branch_block(ipc, opc, 3, stride, rate_list[0], rate_list[0])
        self.b1 = branch_block(ipc, opc, 3, stride, rate_list[1], rate_list[1])
        self.b2 = branch_block(ipc, opc, 3, stride, rate_list[2], rate_list[2])
        self.b3 = branch_block(ipc, opc, 3, stride, rate_list[3], rate_list[3])

        # self.relu = nn.ReLU()
        
    def forward(self, ip):
        b0 = self.b0(ip)
        b1 = self.b1(ip)
        b2 = self.b2(ip)
        b3 = self.b3(ip)
        result = b0+b1+b2+b3
        # result = self.relu(result)
        return result

class MultiAtrousTransposeConv(nn.Module):
    def __init__(self, ipc, opc, rate_list=[2, 4, 6, 8], stride=1):
        super(MultiAtrousTransposeConv, self).__init__()

        if opc%4 != 0 :
            assert('opc must be n x 4')
        branch_opc = int(opc/4)

        def branch_block(ipc, opc, k, s, p, d, op):
            layer_list = [
                nn.ConvTranspose2d(ipc, opc, k, stride=s, padding=p, dilation=d, output_padding=op), 
                nn.InstanceNorm2d(opc)
            ]
            return nn.Sequential(*layer_list)


        self.b0 = branch_block(ipc, branch_opc, 3, stride, rate_list[0], rate_list[0], 1)
        self.b1 = branch_block(ipc, branch_opc, 3, stride, rate_list[1], rate_list[1], 1)
        self.b2 = branch_block(ipc, branch_opc, 3, stride, rate_list[2], rate_list[2], 1)
        self.b3 = branch_block(ipc, branch_opc, 3, stride, rate_list[3], rate_list[3], 1)

        self.relu = nn.ReLU()
        
    def forward(self, ip):
        b0 = self.b0(ip)
        b1 = self.b1(ip)
        b2 = self.b2(ip)
        b3 = self.b3(ip)
        result = torch.cat((b0, b1, b2, b3), dim=1)
        result = self.relu(result)
        return result

def calculate_padding(d, ks):
    return int(d * (ks - 1) * 0.5)


class PerceptualLoss():
    def __init__(self, content_layer, style_layer, device, weight_style, weight_content):
        self.net = models.vgg16(pretrained=True).features
        self.net.to(device)
        for param in self.net.parameters():
            param.requires_grad = False
        self.net = list(self.net)
        self.style_layer = style_layer
        self.content_layer = content_layer
        self.weight_style = weight_style
        self.weight_content = weight_content
        self.content_loss_func = [nn.MSELoss()] * len(content_layer)
        self.style_loss_func = [nn.MSELoss()] * len(style_layer)
        # self.__class__ = 'perceptual_loss'
        
    def calculate_loss(self, pred, content, style):
        
        pred_feature, pred_content = self.__get_features(pred, self.content_layer, self.style_layer)
        _, content_target = self.__get_features(content, self.content_layer, self.style_layer)
        style_target, _ = self.__get_features(style, self.content_layer, self.style_layer)
        pred_gram = []
        target_gram = []

        for i in pred_feature:
            pred_gram.append(self.__gram_matrix(i))
        for i in style_target:
            target_gram.append(self.__gram_matrix(i))
            
        style_loss = 0
        content_loss = 0
        
        for i in range(len(self.weight_style)):
            style_loss += self.style_loss_func[i](pred_gram[i], target_gram[i]) * self.weight_style[i]
        
        for i in range(len(self.weight_content)):
            content_loss += self.content_loss_func[i](pred_content[i], content_target[i]) * self.weight_content[i]
        # print(self.weight_style)
        return 1e3 * style_loss + content_loss
    
    def __get_features(self, image, content_layer, style_layer):
        pool_cnt = 1
        relu_cnt = 1
        prev_out = 0
        current_out = None
        output_list = []
        content_out = []
        for layer in self.net:

            name = str(layer.__class__).split('.')[-1]
            if current_out is None:
                
                current_out = layer(image)
            else:
                current_out = layer(prev_out)
            
            if 'Pool' in name:
                pool_cnt += 1
                relu_cnt = 1
            elif 'ReLU' in name:
                layer_name = str(pool_cnt) + ',' + str(relu_cnt)
                
                if layer_name in style_layer:
                    output_list.append(current_out)
#                     print(layer_name)
                if layer_name in content_layer:
                    content_out.append(current_out)
                
                relu_cnt += 1

            prev_out = current_out
            
        return output_list, content_out
    
    def __gram_matrix(self, feature):
        b, c, h, w = feature.size()
        F = feature.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h*w)
        return G
    
    
def get_feature_output(model, image, layers=None, content=False):
    if layers is None:
        if content:
            layers = {
                    '21': 'conv4_2'
                    }
        else:
            layers = {'0': 'conv1_1',
                    '5': 'conv2_1', 
                    '10': 'conv3_1', 
                    '19': 'conv4_1',
                    '21': 'conv4_2',  ## content representation
                    '28': 'conv5_1'
                    }
    feature_output = []
    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            feature_output.append(image)
    return feature_output

def gram_matrix(feature):
    # batch_size is 1
    b, c, h, w = feature.size()
    F = feature.view(b, c, h*w)
    G = torch.bmm(F, F.transpose(1, 2))
    G.div_(h*w)
    return G


class styleTransferLoss():
    def __init__(self, device):
        self.net = models.vgg16(pretrained=True).features
        self.net.to(device)
        for param in self.net.parameters():
            param.requires_grad = False
        self.criterion = torch.nn.MSELoss()
        

    def calculate_loss(self, pred, content, style, layers=None):
        pred_features = get_feature_output(self.net, pred, layers=layers)
        pred_content = get_feature_output(self.net, pred, layers=layers, content=True)
        style_features = get_feature_output(self.net, style, layers=layers)
        content_features = get_feature_output(self.net, content, layers=layers, content=True)
        style_gram =  [gram_matrix(feature) for feature in style_features ]
        pred_gram = [gram_matrix(feature) for feature in pred_features]

        content_loss = self.criterion(pred_content[0], content_features[0])

        style_loss = 0
        for i in range(len(style_gram)):
            style_loss += self.criterion(pred_gram[i], style_gram[i])

        return style_loss, content_loss

class ResidualBlock_atrous(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock_atrous, self).__init__()

        conv_block = [  MultiAtrousConv(in_features, in_features),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorMultiscaleBilinear(nn.Module):
    def __init__(self, input_nc, output_nc, down_sample=3, n_residual_blocks=6, in_features=64):
        super(GeneratorMultiscaleBilinear, self).__init__()
        
        self.init_block = nn.Sequential(*[   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, in_features, 7),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True) ])

        # Downsampling
        def down_block(ipc, opc):
            result = [  nn.Conv2d(ipc, opc, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(opc),
                        nn.ReLU(inplace=True) ]
            return nn.Sequential(*result)

        encoder = []
        out_features = in_features*2
        for _ in range(down_sample):
            encoder += [MultiAtrousConv(in_features, out_features, stride=2)]
            in_features = out_features
            out_features = in_features*2

        self.encoder = nn.ModuleList(encoder)

        # Residual blocks
        res_block = []
        for _ in range(n_residual_blocks):
            res_block += [ResidualBlock_atrous(in_features)]

        self.res_block = nn.Sequential(*res_block)

        def up_block(ipc, opc):
            result = [  nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(ipc, opc, 3, padding=1),
                        nn.InstanceNorm2d(opc),
                        nn.ReLU(inplace=True) ]
            return nn.Sequential(*result)

        decoder = []
        # for concatenation
        # in_features = in_features * 2
        out_features = in_features//2
        # print(in_features)
        for _ in range(down_sample):
            # print(in_features)
            decoder += [up_block(in_features*2, out_features)]
            in_features = out_features
            out_features = in_features//2
        self.decoder = nn.ModuleList(decoder)

        # Output layer
        self.output_layer = nn.Sequential(*[  nn.ReflectionPad2d(3),
                    nn.Conv2d(out_features*2, output_nc, 7),
                    nn.Tanh() ])

        # self.model = nn.Sequential(*model)

    def forward(self, x):
        # return self.model(x)

        concat = None
        temp = self.init_block(x)
        for layer in self.encoder:
            if concat is None:
                concat = [layer(temp)]
            else:
                concat.append(layer(concat[-1]))
        temp = self.res_block(concat[-1])

        for cnt in range(len(self.decoder)):
            sub = cnt + 1
            idx = len(self.decoder) - sub
            ip_temp = torch.cat((temp, concat[idx]), dim=1)
            temp = self.decoder[cnt](ip_temp)
        
        temp = self.output_layer(temp)
        return temp


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        