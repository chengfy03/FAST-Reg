import numpy as np
import torch
from torchvision.transforms import transforms

from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util


class CUT2Model(BaseModel):
    """ 在 CUT 基础上加入了 ITTR 模块 和 resnet34-STN 网络
    生成网络使用非成对数据集,配准网络使用成对数据集
    CUT 网络学习率 lr = 0.0002
    STN 网络学习率 lr = 0.00005 也许能加入 smooth_loss 来限制 STN 的配准结果不要太过离谱
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--deformation_mode',type=str, default="affine", choices='(affine, perspective)',help="netR 的变化方式 仿射变换 或 透视变换")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'R', 'smooth']
        self.visual_names = ['real_A', 'fake_B', 'register_fake_B', 'real_B', 'register_real_A']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['train_B','idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'R']
        else:  # during test time, only load G
            self.model_names = ['G', 'R']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        # stn
        self.netR = networks.define_stn(in_channels=opt.input_nc,net_name=opt.netR,deformation_mode=opt.deformation_mode, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            # STN loss
            self.cosine = torch.nn.CosineEmbeddingLoss().to(self.device)
            self.target = torch.tensor([1]).to(self.device)

            self.lambda_smooth = 1.0
            # self.lambda_smooth = 0.0005

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # STN optimizer parameters
            self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_R)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.train_B = self.train_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()  # calculate gradients for D
            self.compute_G_loss().backward()  # calculate graidents for G
            self.compute_R_loss().backward()
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                    betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self,epoch):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)

        # update G
        # self.set_requires_grad([self.netG, self.netF], True)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
        # self.set_requires_grad([self.netG, self.netF], False)

        # update R
        # self.set_requires_grad([self.netR], True)
        self.optimizer_R.zero_grad()
        # if epoch > 30:
        #     self.loss_R =  self.compute_R_loss()
        # else:
        #     self.loss_R = 0.0 * self.compute_R_loss()
        self.loss_R = self.compute_R_loss()
        self.loss_R.backward()
        self.optimizer_R.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # real_B 为图像配准时的基准图像
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # train_B 为训练生成器的图像
        self.train_B = input['trainB'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # self.trainB_paths = input['trainB_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.train_B),
                              dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        # print("real",self.real.shape)

        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        # fake 为 real_A -> fake_B 和 train_B -> idt_B
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        # fake_B 经STN网络 -> register_fake_B
        # 变形场由fake_B预测得到
        self.register_fake_B, self.register_real_A, self.grid, self.reg_term = self.netR(self.fake_B.detach(), self.real_A, self.real_B)

        # 变形场由real_A预测得到
        # self.register_real_A, self.register_fake_B, self.grid = self.netR(self.real_A,self.fake_B.detach())

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D_fake = loss_D_fake.mean()
        # Real
        pred_real = self.netD(self.train_B)
        loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.train_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def compute_R_loss(self):
        center_crop_224 = transforms.CenterCrop(224)
        loss_R = self.cosine(center_crop_224(self.register_fake_B)[0][0], center_crop_224(self.real_B)[0][0],self.target)
        self.loss_smooth = self.reg_term
        return loss_R + self.lambda_smooth * self.loss_smooth
        # return loss_R
