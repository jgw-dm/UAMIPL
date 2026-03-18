import torch
import numpy as np
import torch.optim as optim

from torch.distributions.kl import kl_divergence
from torch_scatter import scatter_softmax, segment_add_csr

from utils import logging, device
from dataloader import setup_scatter
from utils_nn import generate_init_params, FeatureExtractor
from utils_posterior import BayesianPosterior


class UAMIPL(torch.nn.Module):
    def __init__(self, cfg, params=[None, 1, 1]):
        super().__init__()
        self.args = cfg
        self.predict_flag = False
        self.nr_class = self.args.nr_class
        self.nr_samples = self.args.nr_samples
        self.fea_extractor = FeatureExtractor(self.args)

        [init_params, self.dim_xs, nr_fixed_effects] = params

        ln_sigma_u = torch.full((1, self.nr_class), 0.5 * np.log(0.5))
        if init_params is not None:
            *_, var_z, alpha = init_params
            ln_sigma_z = 0.5 * torch.log(var_z)
        else:
            alpha = torch.zeros((nr_fixed_effects, self.nr_class))
            ln_sigma_z = torch.full((1, self.nr_class), 0.5 * np.log(0.5))

        [self.alpha, self.ln_sigma_u, self.ln_sigma_z, self.posterior] = [
            torch.nn.Parameter(alpha),
            torch.nn.Parameter(ln_sigma_u),
            torch.nn.Parameter(ln_sigma_z),
            BayesianPosterior(self.dim_xs, self.nr_class, init_params)
        ]

    def initialize_model(x, fe, s, cfg_params):
        fea_extractor = FeatureExtractor(cfg_params)
        xs = fea_extractor(x)
        xs_array = [i_x.detach().numpy() for i_x in xs]
        init_params = generate_init_params(xs_array, fe, s)
        model_params = [init_params, xs_array[0].shape[1], fe.shape[1]]
        return UAMIPL(cfg_params, model_params)

    @property
    def prior_dist(self):
        [scale_u, scale_z] = [
            self.ln_sigma_u.T * torch.ones([1, self.dim_xs], device=device),
            self.ln_sigma_z.T * torch.ones([1, self.dim_xs], device=device)
        ]
        cov_ldiag = torch.cat([scale_u, scale_z], 1)
        [cov_factor, mu] = [
            torch.zeros_like(cov_ldiag)[:, :, None],
            torch.zeros_like(cov_ldiag)
        ]
        prior_dist = torch.distributions.LowRankMultivariateNormal(mu, cov_factor, torch.exp(2 * cov_ldiag))
        return prior_dist

    def forward(self, xs, current_epoch=0, warmup_epochs=50):
        beta_u, beta_z = self.posterior.get_beta(self.nr_samples, self.predict_flag)
        b = torch.sqrt((beta_z ** 2).mean(0, keepdim=True))
        eta = beta_z / b
        x, i, i_ptr = setup_scatter(xs)
        _w = torch.einsum("iq,qps->ips", x, beta_u)
        w_attn = scatter_softmax(_w, i, dim=0)
        evidence = torch.relu(_w) + 1e-6
        alpha = evidence + 1.0
        uncertainty = self.nr_class / torch.sum(alpha, dim=-1, keepdim=True)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
        mean_unc = uncertainty.mean().item()
        reliability = 1.0 - uncertainty
        # Sanity-check ablation: shuffle reliability within each bag
        if getattr(self.args, "shuffle_reliability", False):
            reliability_shuffled = reliability.clone()
            unique_bags = torch.unique(i)
            for bag_id in unique_bags:
                bag_mask = (i == bag_id)
                bag_rel = reliability[bag_mask]
                perm = torch.randperm(bag_rel.shape[0], device=bag_rel.device)
                reliability_shuffled[bag_mask] = bag_rel[perm]
            reliability = reliability_shuffled

        lambda_base = getattr(self.args, "lambda_edl", 0.2)
        unc_threshold = getattr(self.args, "unc_threshold", 0.0)
        if current_epoch >= warmup_epochs and lambda_base > 0 and mean_unc >= unc_threshold:
            lambda_dyn = lambda_base * mean_unc
            w = (1 - lambda_dyn) * w_attn + lambda_dyn * reliability
        else:
            w = w_attn

        t = torch.einsum("iq,qps->ips", x, eta)
        z_bag = segment_add_csr(w * t, i_ptr)
        mean, std = [z_bag.mean(0), z_bag.std(0)]
        if std.isnan().any():
            std = 1
        z_bag = b * (z_bag - mean) / std
        return z_bag

    def calculate_loss(self, u, fe, s, kld_w):
        logits = fe.mm(self.alpha).unsqueeze(2) + u
        logits_d = logits.permute(0, 2, 1)

        link_func = torch.sum(
            torch.sum(
                logits_d * s.unsqueeze(1).expand(-1, self.nr_samples, -1),
                dim=[1, 2]
            )
        ) / s.shape[0]

        posterior_dist = self.posterior.distribution
        kld = kl_divergence(posterior_dist, self.prior_dist)
        kld_term = kld_w * kld.sum() / s.shape[0]

        loss = - link_func + kld_term

        res_dict = {
            "loss": round(loss.item(), 4),
            "ll": round(link_func.item(), 4),
            "kld": round(kld_term.item(), 4)
        }
        return loss, res_dict

    def calculate_obj(self, data, fe, s_label, ratio_kld, epoch):
        xs = self.fea_extractor(data)
        z_b = self(xs, current_epoch=epoch)
        loss, res = self.calculate_loss(z_b, fe, s_label, kld_w=ratio_kld)
        return loss, res, z_b

    def regenerate_s(self, s, zb, w_conf):
        s_candiate = torch.zeros(s.shape).to(device)
        s_candiate[s > 0] = 1
        s_can_r = s_candiate.unsqueeze(2).expand(-1, -1, self.nr_samples).detach()
        s_pred = torch.mean(torch.softmax(zb.detach(), dim=1) * s_can_r, dim=2)
        s = s.detach()
        new_s = w_conf * s + (1. - w_conf) * s_pred
        return new_s

    def fit(self, train_loader, num_bags, weight_list):
        [lr_value, n_epochs, reg_value] = [self.args.lr, self.args.epochs, self.args.reg]
        optimizer = optim.SGD(self.parameters(), lr=lr_value, momentum=0.9,
                              nesterov=True, weight_decay=reg_value)

        for epoch in range(n_epochs):
            for step, (x, s, _, cov_b, idx) in enumerate(train_loader):
                i_epoch = epoch + 1
                ratio_kld = len(x) / num_bags

                loss, res, z_bag = self.calculate_obj(x, cov_b, s, ratio_kld, i_epoch)
                res["epoch"], res["step"] = i_epoch, step
                if epoch == 0 or (i_epoch) % 10 == 0:
                    logging.info("Loss Dict: {}".format(res))

                alpha_value = weight_list[epoch]
                if not getattr(self.args, "uniform_candidate_weighting", False):
                    new_s = self.regenerate_s(s, z_bag, alpha_value)
                    train_loader.dataset.partial_bag_lab[idx] = new_s

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.predict_flag = True
        return

    @torch.inference_mode()
    def predict(self, xs):
        self.nr_samples = None
        xs = self.fea_extractor(xs)
        s_hat = self(xs).squeeze(2)
        s_logits = torch.softmax(s_hat, dim=1)
        y_pred = torch.max(s_logits.data, 1)[1]
        return y_pred
