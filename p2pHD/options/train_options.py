from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', type=bool, default=False, help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')

        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', default=True, action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        # for R2L
        self.parser.add_argument('--inputType', type=str, default="png")
        self.parser.add_argument('--n_scale', type=int, default=3, help='number of skip connection for multiple input scale')
        self.parser.add_argument('--transfer', type=bool, default=False, help='if it is trained for domain transfer')
        self.parser.add_argument('--fine_tune_features', type=bool, default=False, help='change it to True when fine-tuning features encoder for domain invariant features')
        self.parser.add_argument('--AE_type', type=str, default='radar', help='choose to train AE for lidar or radar')
        self.parser.add_argument('--wgan', type=bool, default=False, help='choose to train encoder with WGAN')
        self.parser.add_argument('--n_critic', type=int, default=1, help='freq of updating critics and encoder')
        self.parser.add_argument('--multi_scale', type=bool, default=True, help='whether to use multi scale encoder')
        self.parser.add_argument('--load_pretrain_radar', type=str, default='', help='path to pretrained radar file')
        self.parser.add_argument('--load_pretrain_lidar', type=str, default='', help='path to pretrained lidar file')
        self.parser.add_argument('--load_netDF', type=str, default=' ', help='path to pretrained feature discriminator file')

        # domain adaptation
        self.parser.add_argument('--uda', type=bool, default=False, help='Unsupervised domain adaptation training, highest priority')
        self.parser.add_argument('--training_module', type=str, default='discriminator', help='which module are we going to train, options: encoder, decoder, discriminator')
        self.parser.add_argument('--w_lambda', type=float, default=10, help='coefficient for gradient penalty')
        self.parser.add_argument('--encoder_resblock', type=int, default=0, help='number of res blocks in encoder')
        self.parser.add_argument('--decoder_resblock', type=int, default=0, help='number of res blocks in decoder')
        self.parser.add_argument('--max_ch', type=int, default=256, help='maxinum channels for features')
        self.parser.add_argument('--use_sample_loss', type=bool, default=True, help='if to use sample GAN loss')
        self.isTrain = True
