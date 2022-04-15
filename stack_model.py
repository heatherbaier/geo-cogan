import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
from stack_config import cfg
import torchvision
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


# class CA_NET(nn.Module):
#     # some code is modified from vae examples
#     # (https://github.com/pytorch/examples/blob/master/vae/main.py)
#     def __init__(self, rank):
#         super(CA_NET, self).__init__()
#         self.t_dim = cfg.TEXT.DIMENSION
#         self.c_dim = cfg.GAN.CONDITION_DIM
#         self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
#         self.relu = nn.ReLU()
#         self.rank = rank

#     def encode(self, text_embedding):
#         x = self.relu(self.fc(text_embedding))
#         mu = x[:, :self.c_dim]
#         logvar = x[:, self.c_dim:]
#         return mu, logvar

#     def reparametrize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
# #         if cfg.CUDA:
#         if self.rank is not None:
#             eps = torch.FloatTensor(std.size()).normal_().to(self.rank)
#         else:
#             eps = torch.FloatTensor(std.size()).normal_()
#         eps = Variable(eps)
#         return eps.mul(std).add_(mu)

#     def forward(self, text_embedding):
#         mu, logvar = self.encode(text_embedding)
#         c_code = self.reparametrize(mu, logvar)
#         return c_code, mu, logvar


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition = True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf * 8 + nef, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        # conditioning output
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code
            
#         print("h_c_code: ", h_c_code)

        output = self.outlogits(h_c_code.nan_to_num(0))
        
#         print("OUTPUT: ", output)
        
        return output.view(-1)


# ############# Networks for stageI GAN #############
# class STAGE1_G(nn.Module):
#     def __init__(self, rank):
#         super(STAGE1_G, self).__init__()
#         self.gf_dim = cfg.GAN.GF_DIM * 8
#         self.ef_dim = cfg.GAN.CONDITION_DIM
#         self.z_dim = cfg.Z_DIM
#         self.rank = rank
#         self.define_module()

#     def define_module(self):
#         ninput = self.z_dim + self.ef_dim
#         ngf = self.gf_dim
#         # TEXT.DIMENSION -> GAN.CONDITION_DIM
#         self.ca_net = CA_NET(self.rank)
        
# #         print(ninput)

#         # -> ngf x 4 x 4
#         self.fc = nn.Sequential(
#             nn.Linear(ninput, ngf * 4 * 4, bias=False),
#             nn.BatchNorm1d(ngf * 4 * 4),
#             nn.ReLU(True))

#         # ngf x 4 x 4 -> ngf/2 x 8 x 8
#         self.upsample1 = upBlock(ngf, ngf // 2)
#         # -> ngf/4 x 16 x 16
#         self.upsample2 = upBlock(ngf // 2, ngf // 4)
#         # -> ngf/8 x 32 x 32
#         self.upsample3 = upBlock(ngf // 4, ngf // 8)
#         # -> ngf/16 x 64 x 64
#         self.upsample4 = upBlock(ngf // 8, ngf // 16)
#         # -> 3 x 64 x 64
#         self.img = nn.Sequential(
#             conv3x3(ngf // 16, 3),
#             nn.Tanh())

#     def forward(self, text_embedding, noise):
#         c_code, mu, logvar = self.ca_net(text_embedding)
#         z_c_code = torch.cat((noise, c_code), 1)
# #         print(z_c_code.shape)
#         h_code = self.fc(z_c_code)
    
#         h_code = h_code.nan_to_num(0)

#         h_code = h_code.view(-1, self.gf_dim, 4, 4)
#         h_code = self.upsample1(h_code)
#         h_code = self.upsample2(h_code)
#         h_code = self.upsample3(h_code)
#         h_code = self.upsample4(h_code)
        
#         h_code = h_code.nan_to_num(0)
        
# #         print(h_code)
        
#         # state size 3 x 64 x 64
#         fake_img = self.img(h_code)
        
#         fake_img = fake_img.nan_to_num(0)
        
# #         print()
        
#         return None, fake_img, mu, logvar


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()
#         self.rank = rank

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
#         if cfg.CUDA:
        if self.rank is not None:
            eps = torch.FloatTensor(std.size()).normal_().to(self.rank)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar
    

class STAGE1_G(nn.Module):
    def __init__(self):
        super(STAGE1_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM * 8
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.z_dim = cfg.Z_DIM
        self.LATENT_DIM = 50
#         self.rank = rank
        self.define_module()

    def define_module(self):
        
        ninput = self.LATENT_DIM
        ngf = self.gf_dim
        
        print("NGF: ", ngf)        
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
#         self.ca_net = CA_NET()
        
#         print(ninput)

        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # -> 3 x 64 x 64
        self.img_a = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())
        self.img_b = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())        

    def forward(self, noise):
#         c_code, mu, logvar = self.ca_net(text_embedding)
#         z_c_code = torch.cat((noise, c_code), 1)
#         print(z_c_code.shape)
        
        h_code = self.fc(noise)
            
        h_code = h_code.nan_to_num(0).unsqueeze(2)
        
        h_code = h_code.view(-1, self.gf_dim, 4, 4)
                
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        
        h_code = h_code.nan_to_num(0)
                
        # state size 3 x 64 x 64
        fake_img_a, fake_img_b = self.img_a(h_code), self.img_b(h_code)
                        
        return fake_img_a, fake_img_b


# class STAGE1_D(nn.Module):
#     def __init__(self):
#         super(STAGE1_D, self).__init__()
#         self.df_dim = cfg.GAN.DF_DIM
#         self.ef_dim = cfg.GAN.CONDITION_DIM
#         self.define_module()

#     def define_module(self):
#         ndf, nef = self.df_dim, self.ef_dim
#         self.encode_img = nn.Sequential(
#             nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size (ndf*2) x 16 x 16
#             nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size (ndf*4) x 8 x 8
#             nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             # state size (ndf * 8) x 4 x 4)
#             nn.LeakyReLU(0.2, inplace=True)
#         )

#         print("ndf: ", ndf)
#         print("nef: ", nef)
        
#         self.get_cond_logits = D_GET_LOGITS(ndf, nef)
#         self.get_uncond_logits = None

#     def forward(self, image):
#         img_embedding = self.encode_img(image)

#         return img_embedding
    
    
    
    
class STAGE1_D(nn.Module):
    def __init__(self):
        
        super(STAGE1_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        ndf, nef = self.df_dim, self.ef_dim
        
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size = 4)
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef)
        self.get_uncond_logits = None
        
        self.discrim_fc = torch.nn.Linear(512, 2)
        
        classify = torchvision.models.resnet18()
#         print(self.classify)
        classify.fc = torch.nn.Linear(512, 2)
        classify = classify.cuda()
        self.classify = classify

    def forward(self, image_a, image_b):
        
        img_embedding_a = self.encode_img(image_a)
        img_embedding_b = self.encode_img(image_b)
        
        img_embedding_ab = torch.cat((img_embedding_a, img_embedding_b)).flatten(start_dim = 1)
        
        discrim = self.discrim_fc(img_embedding_ab)

        return discrim, img_embedding_a, img_embedding_b
    
    def classify_a(self, x):
        
        return self.classify(x)
    
    

class Tan2MexCoGANTrainer():
    
    def __init__(self, batch_size = 16, latent_dims = 100):
        super(Tan2MexCoGANTrainer, self).__init__()
        
        self.dis = STAGE1_D().cuda()
        self.gen = STAGE1_G().cuda()
#         self.classify()
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0005)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0005)
        self.true_labels = torch.LongTensor(np.ones(batch_size * 2, dtype=np.int)).cuda()
        self.fake_labels = torch.LongTensor(np.zeros(batch_size * 2, dtype=np.int)).cuda()    
        self.mse_loss_criterion = torch.nn.MSELoss()


    def dis_update(self, images_a, images_b, labels_a, noise):
        
        ################################################################################################
        # Adversarial part for true images
        # THIS IS THE STANDARD FORWARD PASS AND PREDICTS WHETHER THE **TRUE** IMAGES ARE TRUE OR FAKE
        ################################################################################################        
        true_outputs, true_feat_a, true_feat_b = self.dis(images_a, images_b)
        true_loss = nn.functional.cross_entropy(true_outputs, self.true_labels)
        _, true_predicts = torch.max(true_outputs.data, 1)
        true_acc = (true_predicts == 1).sum()/(1.0*true_predicts.size(0))
        
        ################################################################################################
        # Adversarial part for fake images
        # THIS IS THE STANDARD FORWARD PASS AND PREDICTS WHETHER THE **FAKE** IMAGES ARE TRUE OR FAKE
        ################################################################################################        
        fake_images_a, fake_images_b = self.gen(noise)
        fake_outputs, fake_feat_a, fake_feat_b = self.dis(fake_images_a, fake_images_b)
        fake_loss = nn.functional.cross_entropy(fake_outputs, self.fake_labels)
        _, fake_predicts = torch.max(fake_outputs.data, 1)
        fake_acc = (fake_predicts == 0).sum() / (1.0 * fake_predicts.size(0))
        
        
        dummy_tensor = Variable(
            torch.zeros(fake_feat_a.size(0), fake_feat_a.size(1), fake_feat_a.size(2), fake_feat_a.size(3))).cuda()
        mse_loss = self.mse_loss_criterion(fake_feat_a - fake_feat_b, dummy_tensor) * fake_feat_a.size(
            1) * fake_feat_a.size(2) * fake_feat_a.size(3)
        
        
        ################################################################################################
        # Classification loss
        # THIS ACTUALLY PREDICTS THE PASS/FAIL CLASS OF AN IMAGE
        ################################################################################################
        cls_outputs = self.dis.classify_a(images_a)
        cls_loss = nn.functional.cross_entropy(cls_outputs, labels_a)
        _, cls_predicts = torch.max(cls_outputs.data, 1)
        cls_acc = (cls_predicts == labels_a.data).sum() / (1.0 * cls_predicts.size(0))

#         d_loss = true_loss + fake_loss + mse_wei * mse_loss + cls_wei * cls_loss
        
        d_loss = true_loss + fake_loss + mse_loss + cls_loss
        d_loss.backward()
        self.dis_opt.step()
        
        return 0.5 * (true_acc + fake_acc), mse_loss, cls_acc        
        
        
    def gen_update(self, noise):
        
        self.gen.zero_grad()
        fake_images_a, fake_images_b = self.gen(noise)
        fake_outputs, fake_feat_a, fake_feat_b = self.dis(fake_images_a, fake_images_b)
        fake_loss = nn.functional.cross_entropy(fake_outputs, self.true_labels.cuda())
        fake_loss.backward()
        self.gen_opt.step()
        
        return fake_images_a, fake_images_b        
        


# ############# Networks for stageII GAN #############
class STAGE2_G(nn.Module):
    def __init__(self, STAGE1_G, rank):
        super(STAGE2_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.z_dim = cfg.Z_DIM
        self.STAGE1_G = STAGE1_G
        self.rank = rank
        # fix parameters of stageI GAN
        for param in self.STAGE1_G.parameters():
            param.requires_grad = False
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net = CA_NET(self.rank)
        # --> 4ngf x 16 x 16
        self.encoder = nn.Sequential(
            conv3x3(3, ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.hr_joint = nn.Sequential(
            conv3x3(self.ef_dim + ngf * 4, ngf * 4),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.residual = self._make_layer(ResBlock, ngf * 4)
        # --> 2ngf x 32 x 32
        self.upsample1 = upBlock(ngf * 4, ngf * 2)
        # --> ngf x 64 x 64
        self.upsample2 = upBlock(ngf * 2, ngf)
        # --> ngf // 2 x 128 x 128
        self.upsample3 = upBlock(ngf, ngf // 2)
        # --> ngf // 4 x 256 x 256
        self.upsample4 = upBlock(ngf // 2, ngf // 4)
        # --> 3 x 256 x 256
        self.img = nn.Sequential(
            conv3x3(ngf // 4, 3),
            nn.Tanh())

    def forward(self, noise):
        _, stage1_img, _, _ = self.STAGE1_G(text_embedding, noise)
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img)

        c_code, mu, logvar = self.ca_net(text_embedding)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 16, 16)
        i_c_code = torch.cat([encoded_img, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        fake_img = self.img(h_code)
        return stage1_img, fake_img, mu, logvar


class STAGE2_D(nn.Module):
    def __init__(self):
        super(STAGE2_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),  # 128 * 128 * ndf
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # 64 * 64 * ndf * 2
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 32 * 32 * ndf * 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 16
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 32
            conv3x3(ndf * 32, ndf * 16),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),   # 4 * 4 * ndf * 16
            conv3x3(ndf * 16, ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)   # 4 * 4 * ndf * 8
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, image):
        img_embedding = self.encode_img(image)
        return img_embedding
