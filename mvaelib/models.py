from .mvae_utils import AverageMeter

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as opt

import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_shape, encoded_size, arch):
        super(Encoder, self).__init__()

        layers = []

        inputs = input_shape
        for arc in arch:
            layers.append(nn.Linear(inputs, arc))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(arc, momentum=0.8))
            inputs = arc

        layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(arch[-1], encoded_size * 2))
        self.encoder = nn.Sequential(*layers)
        self.encoded_size = encoded_size

    def forward(self, x):
        features = self.encoder(x)
        return features[:, :self.encoded_size], features[:, self.encoded_size:]


class Decoder(nn.Module):
    def __init__(self, input_shape, encoded_size, arch):
        super(Decoder, self).__init__()
        
        layers = []

        inputs = encoded_size
        for arc in reversed(arch):
            layers.append(nn.Linear(inputs, arc))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(arc, momentum=0.8))
            inputs = arc

        layers.append(nn.Linear(arch[0], input_shape))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)

        return pd_mu, pd_logvar

def prior_expert(size, use_cuda=True):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()

    return mu, logvar


class MVAE3(nn.Module):
    def __init__(self, input_shapes, encoded_size, beta, learning_rate, archs):
        super(MVAE3, self).__init__()

        self.encoder1 = Encoder(input_shapes[0], encoded_size, archs[0])
        self.encoder2 = Encoder(input_shapes[1], encoded_size, archs[1])
        self.encoder3 = Encoder(input_shapes[2], encoded_size, archs[2])
        self.decoder1 = Decoder(input_shapes[0], encoded_size, archs[0])
        self.decoder2 = Decoder(input_shapes[1], encoded_size, archs[1])
        self.decoder3 = Decoder(input_shapes[2], encoded_size, archs[2])
        self.experts  = ProductOfExperts()

        self.encoded_size = encoded_size
        self.beta         = beta
        self.lr           = learning_rate

        total_dims = sum(input_shapes)
        self.lambda1 = total_dims - input_shapes[0]
        self.lambda2 = total_dims - input_shapes[1]
        self.lambda3 = total_dims - input_shapes[2]
        sum_lambda = self.lambda1 + self.lambda2 + self.lambda3
        self.lambda1 /= sum_lambda
        self.lambda2 /= sum_lambda
        self.lambda3 /= sum_lambda

        self.optimizer = None

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)

        return mu

    def forward(self, modal1, modal2, modal3):
        mu, logvar   = self.infer(modal1, modal2, modal3)
        z            = self.reparameterize(mu, logvar)
        modal1_recon = self.decoder1(z)
        modal2_recon = self.decoder2(z)
        modal3_recon = self.decoder3(z)

        return modal1_recon, modal2_recon, modal3_recon, mu, logvar

    def infer(self, modal1, modal2, modal3):
        batch_size = modal1.size(0) if modal1 is not None else modal2.size(0) if modal2 is not None else modal3.size(0)
        mu, logvar = prior_expert((1, batch_size, self.encoded_size), next(self.parameters()).is_cuda)

        if modal1 is not None:
            modal1_mu, modal1_logvar = self.encoder1(modal1)
            mu     = torch.cat((mu,     modal1_mu.unsqueeze(0)),     dim=0)
            logvar = torch.cat((logvar, modal1_logvar.unsqueeze(0)), dim=0)

        if modal2 is not None:
            modal2_mu, modal2_logvar = self.encoder2(modal2)
            mu     = torch.cat((mu,     modal2_mu.unsqueeze(0)),     dim=0)
            logvar = torch.cat((logvar, modal2_logvar.unsqueeze(0)), dim=0)

        if modal3 is not None:
            modal3_mu, modal3_logvar = self.encoder3(modal3)
            mu     = torch.cat((mu,     modal3_mu.unsqueeze(0)),     dim=0)
            logvar = torch.cat((logvar, modal3_logvar.unsqueeze(0)), dim=0)

        mu, logvar = self.experts(mu, logvar)

        return mu, logvar

    def elbo_loss(self, modal1_recon, modal1,
                        modal2_recon, modal2,
                        modal3_recon, modal3,
                        mu, logvar, lambda1=1.0, lambda2=1.0, lambda3=1.0):

        if modal1_recon.size() != modal1.size() or\
           modal2_recon.size() != modal2.size() or\
           modal3_recon.size() != modal3.size():
            raise ValueError('Target and input size mismatch')

        modal1_loss, modal2_loss, modal3_loss = 0, 0, 0

        if modal1_recon is not None and modal1 is not None:
            modal1_loss = torch.sum(F.mse_loss(modal1_recon, modal1, reduction='none'), dim=1)
        if modal2_recon is not None and modal2 is not None:
            modal2_loss = torch.sum(F.mse_loss(modal2_recon, modal2, reduction='none'), dim=1)
        if modal3_recon is not None and modal3 is not None:
            modal3_loss = torch.sum(F.mse_loss(modal3_recon, modal3, reduction='none'), dim=1)

        KLD  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        ELBO = torch.mean(lambda1 * modal1_loss + lambda2 * modal2_loss + lambda3 * modal3_loss + self.beta * KLD)

        return ELBO

    def fit(self, epochs, train_generator, test_generator, beta_change=False, log_interval=100, silent=False):
        self.optimizer = opt.Adam(self.parameters(), lr=self.lr)

        beta_step = self.beta / epochs
        if beta_change:
            self.beta = 0

        history = {'loss' : np.zeros((epochs)), 'val_loss' : np.zeros((epochs))}

        for epoch in range(epochs):
            history['loss'][epoch] = self.train_epoch(epoch, train_generator, log_interval, silent)
            if (test_generator is not None):
                history['val_loss'][epoch] = self.test_epoch(epoch, test_generator, silent)

            if beta_change:
                self.beta += beta_step

        return history

    def train_epoch(self, epoch, generator, log_interval, silent=False):
        self.train()
        avg_meter = AverageMeter()

        for batch_idx, (modal1, modal2, modal3) in enumerate(generator):
            modal1     = Variable(modal1.to(next(self.parameters()).device))
            modal2     = Variable(modal2.to(next(self.parameters()).device))
            modal3     = Variable(modal3.to(next(self.parameters()).device))
            batch_size = len(modal1)

            self.optimizer.zero_grad()

            train_loss = torch.tensor(0.0).to(next(self.parameters()).device)
            combinations = [(modal1, modal2, modal3),
                            (modal1, modal2, None),
                            (modal1, None,   modal3),
                            (None,   modal2, modal3),
                            (modal1, None,   None),
                            (None,   modal2, None),
                            (None,   None,   modal3)]

            for c in combinations:
                recon_modal1, recon_modal2, recon_modal3, mu, logvar = self(*c)
                train_loss += self.elbo_loss(
                                    recon_modal1, modal1,
                                    recon_modal2, modal2,
                                    recon_modal3, modal3,
                                    mu, logvar, self.lambda1, self.lambda2, self.lambda3)
            

            avg_meter.update(train_loss.item(), batch_size)

            train_loss.backward()
            self.optimizer.step()

            if not silent and batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.4f}'.format(
                    epoch, batch_idx * len(modal1), len(generator.dataset),
                    100. * batch_idx / len(generator), avg_meter.avg))
        if not silent:
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_meter.avg))

        return avg_meter.avg

    @torch.no_grad()
    def test_epoch(self, epoch, generator, silent):
        self.eval()
        avg_meter = AverageMeter()

        for batch_idx, (modal1, modal2, modal3) in enumerate(generator):
            modal1     = Variable(modal1.to(next(self.parameters()).device))
            modal2     = Variable(modal2.to(next(self.parameters()).device))
            modal3     = Variable(modal3.to(next(self.parameters()).device))
            batch_size = len(modal1)

            test_loss = torch.tensor(0.0).to(next(self.parameters()).device)
            combinations = [(modal1, modal2, modal3),
                            (modal1, modal2, None),
                            (modal1, None,   modal3),
                            (None,   modal2, modal3),
                            (modal1, None,   None),
                            (None,   modal2, None),
                            (None,   None,   modal3)]

            for c in combinations:
                recon_modal1, recon_modal2, recon_modal3, mu, logvar = self(*c)
                test_loss += self.elbo_loss(
                                    recon_modal1, modal1,
                                    recon_modal2, modal2,
                                    recon_modal3, modal3,
                                    mu, logvar, self.lambda1, self.lambda2, self.lambda3)

            avg_meter.update(test_loss.item(), batch_size)

        if not silent:
            print('====> Test loss: {:.4f}'.format(avg_meter.avg))

        return avg_meter.avg

    def save_weights(self, filename):
        torch.save(self.state_dict(), filename)

    def load_weights(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def encode(self, x1, x2, x3):
        mu, logvar = self.infer(x1, x2, x3)
        return self.reparameterize(mu, logvar)

    def decode(self, z):
        return [self.decoder1(z), self.decoder2(z), self.decoder3(z)]


class MultiOutputEncoder(nn.Module):
    def __init__(self, input_shape, encoded_size, arch):
        super(MultiOutputEncoder, self).__init__()

        layers = []

        inputs = input_shape
        for arc in arch:
            layers.append(nn.Linear(inputs, arc))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(arc, momentum=0.8))
            inputs = arc

        layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(arch[-1], encoded_size * 4))
        self.encoder = nn.Sequential(*layers)
        self.encoded_size = encoded_size

    def forward(self, x):
        features = self.encoder(x)
        return features[:, :self.encoded_size], features[:, self.encoded_size:self.encoded_size*2],\
               features[:, self.encoded_size*2:-self.encoded_size], features[:, -self.encoded_size:]

class MultiOutputDecoder(nn.Module):
    def __init__(self, input_shape, encoded_size, arch):
        super(MultiOutputDecoder, self).__init__()
        
        layers = []

        inputs = encoded_size * 2
        for arc in reversed(arch):
            layers.append(nn.Linear(inputs, arc))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(arc, momentum=0.8))
            inputs = arc

        layers.append(nn.Linear(arch[0], input_shape))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class MEMVAE3(nn.Module):
    def __init__(self, input_shapes, encoded_size, beta_shared, beta_single, learning_rate, archs):
        super(MEMVAE3, self).__init__()

        self.encoder1 = MultiOutputEncoder(input_shapes[0], encoded_size, archs[0])
        self.encoder2 = MultiOutputEncoder(input_shapes[1], encoded_size, archs[1])
        self.encoder3 = MultiOutputEncoder(input_shapes[2], encoded_size, archs[2])
        self.decoder1 = MultiOutputDecoder(input_shapes[0], encoded_size, archs[0])
        self.decoder2 = MultiOutputDecoder(input_shapes[1], encoded_size, archs[1])
        self.decoder3 = MultiOutputDecoder(input_shapes[2], encoded_size, archs[2])
        self.experts  = ProductOfExperts()

        self.encoded_size = encoded_size
        self.beta_shared  = beta_shared
        self.beta_single  = beta_single
        self.lr           = learning_rate

        total_dims = sum(input_shapes)
        self.lambda1 = total_dims - input_shapes[0]
        self.lambda2 = total_dims - input_shapes[1]
        self.lambda3 = total_dims - input_shapes[2]
        sum_lambda = self.lambda1 + self.lambda2 + self.lambda3
        self.lambda1 /= sum_lambda
        self.lambda2 /= sum_lambda
        self.lambda3 /= sum_lambda

        self.optimizer = None

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)

        return mu

    def forward(self, modal1, modal2, modal3):
        mu1, logvar1, mu2, logvar2, mu3, logvar3, mu_shared, logvar_shared = self.infer(modal1, modal2, modal3)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        z3 = self.reparameterize(mu3, logvar3)
        z_shared = self.reparameterize(mu_shared, logvar_shared)

        z1 = torch.cat((z1, z_shared), dim=1)
        z2 = torch.cat((z2, z_shared), dim=1)
        z3 = torch.cat((z3, z_shared), dim=1)

        modal1_recon = self.decoder1(z1)
        modal2_recon = self.decoder2(z2)
        modal3_recon = self.decoder3(z3)

        return modal1_recon, modal2_recon, modal3_recon, mu1, logvar1, mu2, logvar2, mu3, logvar3, mu_shared, logvar_shared

    def infer(self, modal1, modal2, modal3):
        batch_size = modal1.size(0) if modal1 is not None else modal2.size(0) if modal2 is not None else modal3.size(0)
        mu_shared, logvar_shared = prior_expert((1, batch_size, self.encoded_size), next(self.parameters()).is_cuda)
        
        mu_single1, logvar_single1 = torch.zeros_like(mu_shared), torch.zeros_like(logvar_shared)
        mu_single2, logvar_single2 = torch.zeros_like(mu_shared), torch.zeros_like(logvar_shared)
        mu_single3, logvar_single3 = torch.zeros_like(mu_shared), torch.zeros_like(logvar_shared)

        if modal1 is not None:
            mu_single1, logvar_single1, mu_shared1, logvar_shared1 = self.encoder1(modal1)
            mu_shared     = torch.cat((mu_shared, mu_shared1.unsqueeze(0)), dim=0)
            logvar_shared = torch.cat((logvar_shared, logvar_shared1.unsqueeze(0)), dim=0)

        if modal2 is not None:
            mu_single2, logvar_single2, mu_shared2, logvar_shared2 = self.encoder2(modal2)
            mu_shared     = torch.cat((mu_shared, mu_shared2.unsqueeze(0)), dim=0)
            logvar_shared = torch.cat((logvar_shared, logvar_shared2.unsqueeze(0)), dim=0)
            
        if modal3 is not None:
            mu_single3, logvar_single3, mu_shared3, logvar_shared3 = self.encoder3(modal3)
            mu_shared     = torch.cat((mu_shared, mu_shared3.unsqueeze(0)), dim=0)
            logvar_shared = torch.cat((logvar_shared, logvar_shared3.unsqueeze(0)), dim=0)
            
        mu_shared, logvar_shared = self.experts(mu_shared, logvar_shared)

        return mu_single1.squeeze(), logvar_single1.squeeze(),\
               mu_single2.squeeze(), logvar_single2.squeeze(),\
               mu_single3.squeeze(), logvar_single3.squeeze(),\
               mu_shared.squeeze(), logvar_shared.squeeze()

    def elbo_loss(self, modal1_recon, modal1,
                        modal2_recon, modal2,
                        modal3_recon, modal3,
                        mu1, logvar1,
                        mu2, logvar2,
                        mu3, logvar3,
                        mu, logvar, lambda1=1.0, lambda2=1.0, lambda3=1.0):

        if modal1_recon.size() != modal1.size() or\
           modal2_recon.size() != modal2.size() or\
           modal3_recon.size() != modal3.size():
            raise ValueError('Target and input size mismatch')

        modal1_loss, modal2_loss, modal3_loss = 0, 0, 0

        if modal1_recon is not None and modal1 is not None:
            modal1_loss  = torch.sum(F.mse_loss(modal1_recon, modal1, reduction='none'), dim=1)
            modal1_loss += -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=1) * self.beta_single
        if modal2_recon is not None and modal2 is not None:
            modal2_loss = torch.sum(F.mse_loss(modal2_recon, modal2, reduction='none'), dim=1)
            modal2_loss += -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp(), dim=1) * self.beta_single
        if modal3_recon is not None and modal3 is not None:
            modal3_loss = torch.sum(F.mse_loss(modal3_recon, modal3, reduction='none'), dim=1)
            modal3_loss += -0.5 * torch.sum(1 + logvar3 - mu3.pow(2) - logvar3.exp(), dim=1) * self.beta_single

        KLD  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        ELBO = torch.mean(torch.mean(lambda1 * modal1_loss + lambda2 * modal2_loss + lambda3 * modal3_loss, dim=0)
                        + torch.mean(self.beta_shared * KLD, dim=0))

        return ELBO

    def fit(self, epochs, train_generator, test_generator, beta_change=False, log_interval=100, silent=False):
        self.optimizer = opt.Adam(self.parameters(), lr=self.lr)

        beta_shared_step = self.beta_shared / epochs
        beta_single_step = self.beta_single / epochs
        if beta_change:
            self.beta_shared = 0
            self.beta_single = 0

        history = {'loss' : np.zeros((epochs)), 'val_loss' : np.zeros((epochs))}

        for epoch in range(epochs):
            history['loss'][epoch] = self.train_epoch(epoch, train_generator, log_interval, silent)
            if (test_generator is not None):
                history['val_loss'][epoch] = self.test_epoch(epoch, test_generator, silent)

            if beta_change:
                self.beta_shared += beta_shared_step
                self.beta_single += beta_single_step

        return history

    def train_epoch(self, epoch, generator, log_interval, silent=False):
        self.train()
        avg_meter = AverageMeter()

        for batch_idx, (modal1, modal2, modal3) in enumerate(generator):
            modal1     = Variable(modal1.to(next(self.parameters()).device))
            modal2     = Variable(modal2.to(next(self.parameters()).device))
            modal3     = Variable(modal3.to(next(self.parameters()).device))
            batch_size = len(modal1)

            self.optimizer.zero_grad()

            train_loss = torch.tensor(0.0).to(next(self.parameters()).device)
            combinations = [(modal1, modal2, modal3),
                            (modal1, modal2, None),
                            (modal1, None,   modal3),
                            (None,   modal2, modal3),
                            (modal1, None,   None),
                            (None,   modal2, None),
                            (None,   None,   modal3)]

            for c in combinations:
                recon_modal1, recon_modal2, recon_modal3, mu1, logvar1, mu2, logvar2, mu3, logvar3, mu_shared, logvar_shared = self(*c)
                train_loss += self.elbo_loss(
                                    recon_modal1, modal1,
                                    recon_modal2, modal2,
                                    recon_modal3, modal3,
                                    mu1, logvar1,
                                    mu2, logvar2,
                                    mu3, logvar3,
                                    mu_shared, logvar_shared,
                                    self.lambda1, self.lambda2, self.lambda3)
            

            avg_meter.update(train_loss.item(), batch_size)

            train_loss.backward()
            self.optimizer.step()

            if not silent and batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.4f}'.format(
                    epoch, batch_idx * len(modal1), len(generator.dataset),
                    100. * batch_idx / len(generator), avg_meter.avg))
        if not silent:
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_meter.avg))

        return avg_meter.avg

    @torch.no_grad()
    def test_epoch(self, epoch, generator, silent):
        self.eval()
        avg_meter = AverageMeter()

        for _, (modal1, modal2, modal3) in enumerate(generator):
            modal1     = Variable(modal1.to(next(self.parameters()).device))
            modal2     = Variable(modal2.to(next(self.parameters()).device))
            modal3     = Variable(modal3.to(next(self.parameters()).device))
            batch_size = len(modal1)

            test_loss = torch.tensor(0.0).to(next(self.parameters()).device)
            combinations = [(modal1, modal2, modal3),
                            (modal1, modal2, None),
                            (modal1, None,   modal3),
                            (None,   modal2, modal3),
                            (modal1, None,   None),
                            (None,   modal2, None),
                            (None,   None,   modal3)]

            for c in combinations:
                recon_modal1, recon_modal2, recon_modal3, mu1, logvar1, mu2, logvar2, mu3, logvar3, mu_shared, logvar_shared = self(*c)
                test_loss += self.elbo_loss(
                                    recon_modal1, modal1,
                                    recon_modal2, modal2,
                                    recon_modal3, modal3,
                                    mu1, logvar1,
                                    mu2, logvar2,
                                    mu3, logvar3,
                                    mu_shared, logvar_shared,
                                    self.lambda1, self.lambda2, self.lambda3)

            avg_meter.update(test_loss.item(), batch_size)

        if not silent:
            print('====> Test loss: {:.4f}'.format(avg_meter.avg))

        return avg_meter.avg

    def save_weights(self, filename):
        torch.save(self.state_dict(), filename)

    def load_weights(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def encode(self, x1, x2, x3):
        mu1, logvar1, mu2, logvar2, mu3, logvar3, mu_shared, logvar_shared = self.infer(x1, x2, x3)
        return self.reparameterize(mu1, logvar1),\
               self.reparameterize(mu2, logvar2),\
               self.reparameterize(mu3, logvar3),\
               self.reparameterize(mu_shared, logvar_shared)

    def decode(self, z_shared, z0):
        z = torch.cat((z0, z_shared), dim=1)
        return [self.decoder1(z), self.decoder2(z), self.decoder3(z)]
