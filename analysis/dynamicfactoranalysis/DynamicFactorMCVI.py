"""
Bayesian Monte Carlo Variational Inference (MCVI) for Dynamic Factor Models

Authors: Brian Godwin Lim, Dominic Dayta
"""

import torch
import numpy as np
from torch import nn
from statsmodels.multivariate.pca import PCA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.regression.linear_model import OLS



class InverseWishart(nn.Module):
    def __init__(self, scale, df):
        super().__init__()
        self.register_buffer('scale', scale)
        self.df = df
        self.p = scale.size(-1)
        assert self.df >= self.p
        
        self.register_buffer('scale_logdet', torch.linalg.slogdet(scale).logabsdet)
        self.register_buffer('scale_inv_chol', torch.linalg.cholesky(torch.linalg.inv(scale)))
        
    def log_prob(self, X):
        if min(X.shape) == 0:
            return torch.zeros(X.size(0), device=self.scale.device)
        
        return -0.5 * (
            -self.df * self.scale_logdet +
            self.df * self.p * torch.log(torch.tensor(2, device=self.scale.device)) +
            2 * torch.special.multigammaln(torch.tensor(0.5 * self.df, device=self.scale.device), p=self.p) +
            (self.df + self.p + 1) * torch.linalg.slogdet(X).logabsdet +
            torch.einsum('...ij,...ji->...',
                         self.scale,
                         torch.linalg.inv(X),
                         )
        ).reshape(X.size(0), -1).sum(dim=1)

    def sample(self, batch_size=1):
        if min(self.scale.shape) == 0 or batch_size == 0:
            return torch.empty(batch_size, *self.scale.shape, device=self.scale.device)
        
        Z = torch.randn(batch_size, *self.scale.shape[:-2], self.p, self.df, device=self.scale.device)
        return torch.linalg.inv(
            torch.einsum('...ij,...jk,...kl,...lm->...im',
                         self.scale_inv_chol,
                         Z,
                         Z.transpose(-2, -1),
                         self.scale_inv_chol.transpose(-2, -1),
                         )
            )


class MatrixNormal(nn.Module):
    def __init__(self, mean, left_cov, right_cov):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('left_cov', left_cov)
        self.register_buffer('right_cov', right_cov)
        self.n, self.p = mean.shape[-2:]
        assert left_cov.shape[-2:] == (self.n, self.n)
        assert right_cov.shape[-2:] == (self.p, self.p)

        self.register_buffer('left_cov_inv', torch.linalg.inv(left_cov))
        self.register_buffer('right_cov_inv', torch.linalg.inv(right_cov))
        self.register_buffer('left_cov_logdet', torch.linalg.slogdet(left_cov).logabsdet)
        self.register_buffer('right_cov_logdet', torch.linalg.slogdet(right_cov).logabsdet)
        self.register_buffer('left_cov_chol', torch.linalg.cholesky(left_cov))
        self.register_buffer('right_cov_chol', torch.linalg.cholesky(right_cov))

    def log_prob(self, X):
        if min(X.shape) == 0:
            return torch.zeros(X.size(0), device=self.mean.device)
        
        return -0.5 * (
            self.n * self.p * torch.log(torch.tensor(2 * torch.pi, device=self.mean.device)) +
            self.p * self.left_cov_logdet +
            self.n * self.right_cov_logdet + 
            torch.einsum('...ij,...jk,...kl,...li->...', 
                         self.right_cov_inv, 
                         (X - self.mean).transpose(-2, -1), 
                         self.left_cov_inv, 
                         (X - self.mean),
                         )
        ).reshape(X.size(0), -1).sum(dim=1)
        
    def sample(self, batch_size=1):
        if min(self.mean.shape) == 0 or batch_size == 0:
            return torch.empty(batch_size, *self.mean.shape, device=self.mean.device)
        
        return self.mean + torch.einsum('...ij,...jk,...kl->...il', 
                                        self.left_cov_chol, 
                                        torch.randn(batch_size, *self.mean.shape, device=self.mean.device), 
                                        self.right_cov_chol.transpose(-2, -1),
                                        )

    def conf_int(self, alpha):
        norm = torch.distributions.Normal(loc=torch.tensor(0.0, device=self.mean.device), scale=torch.tensor(1.0, device=self.mean.device))
        left_rand = norm.icdf(torch.tensor(0.5 * alpha, device=self.mean.device))
        right_rand = norm.icdf(torch.tensor(1 - 0.5 * alpha, device=self.mean.device))
        return (
            self.mean + torch.einsum('...ij,...jk,...kl->...il', 
                                     self.left_cov_chol,
                                     left_rand * torch.ones_like(self.mean),
                                     self.right_cov_chol.transpose(-2, -1),
                                     ),
            self.mean + torch.einsum('...ij,...jk,...kl->...il', 
                                     self.left_cov_chol,
                                     right_rand * torch.ones_like(self.mean),
                                     self.right_cov_chol.transpose(-2, -1),
                                     ),
        )


class MultivariateNormal(nn.Module):
    def __init__(self, mean, cov):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('cov', cov)
        self.n = mean.size(-2)
        assert mean.size(-1) == 1
        
        self.register_buffer('cov_inv', torch.linalg.inv(cov))
        self.register_buffer('cov_logdet', torch.linalg.slogdet(cov).logabsdet)
        self.register_buffer('cov_chol', torch.linalg.cholesky(cov))
        
    def log_prob(self, X):
        if min(X.shape) == 0:
            return torch.zeros(X.size(0), device=self.mean.device)
        
        return -0.5 * (
            self.n * torch.log(torch.tensor(2 * torch.pi, device=self.mean.device)) +
            self.cov_logdet +
            torch.einsum('...ij,...jk,...kl->...', 
                         (X - self.mean).transpose(-2, -1), 
                         self.cov_inv, 
                         (X - self.mean),
                         )
        ).reshape(X.size(0), -1).sum(dim=1)

    def sample(self, batch_size=1):
        if min(self.mean.shape) == 0 or batch_size == 0:
            return torch.empty(batch_size, *self.mean.shape, device=self.mean.device)
        
        return self.mean + torch.einsum('...ij,...jk->...ik', 
                                        self.cov_chol, 
                                        torch.randn(batch_size, *self.mean.shape, device=self.mean.device),
                                        )

    def conf_int(self, alpha):
        norm = torch.distributions.Normal(loc=torch.tensor(0.0, device=self.mean.device), scale=torch.tensor(1.0, device=self.mean.device))
        left_rand = norm.icdf(torch.tensor(0.5 * alpha, device=self.mean.device))
        right_rand = norm.icdf(torch.tensor(1 - 0.5 * alpha, device=self.mean.device))
        return (
            self.mean + torch.einsum('...ij,...jk->...ik', 
                                     self.cov_chol, 
                                     left_rand * torch.ones_like(self.mean),
                                     ),
            self.mean + torch.einsum('...ij,...jk->...ik', 
                                     self.cov_chol, 
                                     right_rand * torch.ones_like(self.mean),
                                     ),
        )


class DiagNormal(MultivariateNormal):
    def __init__(self, mean, std):
        super().__init__(mean.unsqueeze(dim=-1), torch.diag_embed(torch.pow(std, 2), dim1=-2, dim2=-1))
        self.register_buffer('std', std)
    
    def log_prob(self, X):
        return super().log_prob(torch.diagonal(X, dim1=-2, dim2=-1).unsqueeze(dim=-1))
    
    def sample(self, batch_size=1):
        return torch.diag_embed(super().sample(batch_size).squeeze(dim=-1), dim1=-2, dim2=-1)

    def conf_int(self, alpha):
        left, right = super().conf_int(alpha)
        return (
            torch.diag_embed(left.squeeze(dim=-1), dim1=-2, dim2=-1),
            torch.diag_embed(right.squeeze(dim=-1), dim1=-2, dim2=-1),
        )


class DiracDelta(nn.Module):
    def __init__(self, point):
        super().__init__()
        self.register_buffer('point', point)
    
    def log_prob(self, X):
        return torch.log(torch.isclose(X, self.point).all(dim=tuple(range(1, X.ndim))).float())
    
    def sample(self, batch_size=1):
        return self.point.unsqueeze(dim=0).repeat_interleave(repeats=batch_size, dim=0)


class DynamicFactorMCVI(nn.Module):
    def __init__(self, endog, k_factors, factor_order, error_order, arch_order, garch_order, stationary_error=False, eps=0):
        super().__init__()
        self.register_buffer('endog', torch.cat([torch.zeros(1, endog.size(1)), endog], dim=0).unsqueeze(dim=-1))
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.error_order = error_order
        self.arch_order = arch_order
        self.garch_order = garch_order
        self.stationary_error = stationary_error
        self.T, self.N = endog.shape
        self.eps = eps
        
        endog = endog.cpu().numpy()
        res_pca = PCA(endog, ncomp=k_factors)
        res_ols = OLS(endog, res_pca.factors).fit()
        endog = endog - np.dot(res_pca.factors, res_pca.loadings.T)
        res_factors = (
            VAR(res_pca.factors).fit(maxlags=factor_order, ic=None, trend='n') if k_factors > 1 else 
            ARIMA(res_pca.factors, order=(factor_order, 0, 0), trend='n').fit(method='burg')
        )
        res_errors = [
            ARIMA(endog[:, i], order=(error_order, 0, 0), trend='n', enforce_stationarity=stationary_error).fit(method='burg')
            for i in range(self.N)
        ]
        self.variational_params = nn.ParameterDict({
            'factor_mean': nn.Parameter(
                torch.cat([torch.zeros(1, k_factors), torch.from_numpy(res_pca.factors.astype(np.float32))], dim=0).unsqueeze(dim=-1)
            ),
            'factor_cov': nn.Parameter(
                self.cov_to_flat(0.2 * torch.eye(k_factors).repeat(self.T + 1, 1, 1))
            ),
            'factor_loadings_mean': nn.Parameter(
                torch.from_numpy(res_ols.params.astype(np.float32)).transpose(-2, -1)
            ),
            'factor_loadings_left_cov': nn.Parameter(
                self.cov_to_flat(0.2 * torch.eye(self.N))
            ),
            'factor_loadings_right_cov': nn.Parameter(
                self.cov_to_flat(0.2 * torch.eye(k_factors))
            ),
            'factor_transition_mean': nn.Parameter(
                torch.from_numpy(
                    res_factors.coefs.astype(np.float32) if k_factors > 1 else 
                    res_factors.params[:factor_order].reshape(-1, 1, 1).astype(np.float32)
                ).flip(dims=[0])
            ),
            'factor_transition_left_cov': nn.Parameter(
                self.cov_to_flat(0.2 * torch.eye(k_factors).repeat(factor_order, 1, 1))
            ),
            'factor_transition_right_cov': nn.Parameter(
                self.cov_to_flat(0.2 * torch.eye(k_factors).repeat(factor_order, 1, 1))
            ),
            'factor_transition_cov_const_scale': nn.Parameter(
                self.cov_to_flat(torch.eye(k_factors)),
                requires_grad=bool(arch_order or garch_order),
            ),
            'factor_transition_cov_arch_mean': nn.Parameter(
                torch.zeros(arch_order, k_factors)
            ),
            'factor_transition_cov_arch_log_std': nn.Parameter(
                torch.log(0.2 * torch.ones(arch_order, k_factors))
            ),
            'factor_transition_cov_garch_mean': nn.Parameter(
                torch.zeros(garch_order, k_factors)
            ),
            'factor_transition_cov_garch_log_std': nn.Parameter(
                torch.log(0.2 * torch.ones(garch_order, k_factors))
            ),
            'error_transition_mean': nn.Parameter(
                self.unconstrain_ar_coeffs(torch.stack([
                    torch.from_numpy(
                        res_error.params[:error_order].astype(np.float32)   
                    ) for res_error in res_errors
                ], dim=1).flip(dims=[0]).unsqueeze(dim=0).unsqueeze(dim=-1)).squeeze(dim=0)
            ),
            'error_transition_log_var': nn.Parameter(
                2 * torch.log(0.2 * torch.ones(error_order, self.N))
            ),
            'error_transition_cov_log_var': nn.Parameter(
                torch.log(torch.tensor([
                    res_error.params[-1].astype(np.float32)
                    for res_error in res_errors
                ]))
            ),
        })
        
        self.variational_param_names = [
            'factor',
            'factor_loadings',
            'factor_transition',
            'factor_transition_cov_const',
            'factor_transition_cov_arch',
            'factor_transition_cov_garch',
            'error_transition',
        ]
        
        self.prior = nn.ModuleDict({
            'factor_loadings': MatrixNormal(
                mean=torch.zeros(self.N, k_factors),
                left_cov=torch.eye(self.N),
                right_cov=torch.eye(k_factors),
            ),
            'factor_transition': MatrixNormal(
                mean=torch.zeros(factor_order, k_factors, k_factors),
                left_cov=torch.eye(k_factors).repeat(factor_order, 1, 1),
                right_cov=torch.eye(k_factors).repeat(factor_order, 1, 1),
            ),
            'factor_transition_cov_const': (
                InverseWishart(
                    scale=torch.eye(k_factors),
                    df=self.k_factors + 2,
                ) if bool(arch_order or garch_order) else
                DiracDelta(
                    point=torch.eye(k_factors),
                )
            ),
            'factor_transition_cov_arch': DiagNormal(
                mean=torch.zeros(arch_order, k_factors),
                std=torch.ones(arch_order, k_factors),
            ),
            'factor_transition_cov_garch': DiagNormal(
                mean=torch.zeros(garch_order, k_factors),
                std=torch.ones(garch_order, k_factors),
            ),
            'error_transition': MultivariateNormal(
                mean=torch.zeros(error_order, self.N, 1),
                cov=torch.eye(self.N).repeat(error_order, 1, 1),
            ),  # Constant parameter
        })
        
    @property
    def device(self):
        return self.endog.device
    
    def safe_pd(self, mat, eps=None):
        eps = eps if eps else self.eps
        return 0.5 * (mat + mat.transpose(-2, -1)) + eps * torch.eye(mat.size(-1), device=mat.device)
    
    def flat_to_cov(self, flat, eps=None):
        dim = int(0.5 * ((1 + 8 * flat.size(-1)) ** 0.5 - 1))
        row, col = torch.tril_indices(dim, dim, device=flat.device)
        L = torch.zeros(*flat.shape[:-1], dim, dim, device=flat.device)
        L[..., row, col] = flat
        return self.safe_pd(L @ L.transpose(-2, -1), eps)
    
    @classmethod
    def cov_to_flat(cls, cov):
        L = torch.linalg.cholesky(cov)
        row, col = torch.tril_indices(cov.size(-1), cov.size(-1), device=cov.device)
        return L[..., row, col]
    
    @classmethod
    def shift(cls, tensor, shifts, dim=0, clip=False):
        return torch.cat([tensor.index_select(dim, index=torch.tensor([0], device=tensor.device)).repeat_interleave(repeats=shifts, dim=dim), 
                          (tensor.narrow(dim, start=0, length=tensor.size(dim) - shifts) if clip else tensor)], dim=dim)

    def constrain_ar_coeffs(self, ar_coeffs):
        if ar_coeffs.size(1) == 0 or not self.stationary_error:
            return ar_coeffs
        
        ar_coeffs = ar_coeffs.squeeze(dim=-1).permute(1, 0, 2).flip(dims=[0])
        n = ar_coeffs.size(0)
        r = ar_coeffs / torch.sqrt(1 + torch.pow(ar_coeffs, 2))
        y = torch.zeros(n, n, *ar_coeffs.shape[1:], dtype=ar_coeffs.dtype, device=ar_coeffs.device)
        
        for k in range(n):
            y = y.index_put((torch.tensor([k]), torch.tensor([k])), r[k])
            for i in range(k):
                y = y.index_put((torch.tensor([k]), torch.tensor([i])), y[k - 1, i] + r[k] * y[k - 1, k - 1 - i])
        return - y[n - 1].flip(dims=[0]).permute(1, 0, 2).unsqueeze(dim=-1)
    
    def unconstrain_ar_coeffs(self, ar_coeffs):
        if ar_coeffs.size(1) == 0 or not self.stationary_error:
            return ar_coeffs
        
        ar_coeffs = ar_coeffs.squeeze(dim=-1).permute(1, 0, 2).flip(dims=[0])
        n = ar_coeffs.size(0)
        y = torch.zeros(n, n, *ar_coeffs.shape[1:], dtype=ar_coeffs.dtype, device=ar_coeffs.device)
        
        y = y.index_put((torch.tensor([n - 1]), ), - ar_coeffs)
        for k in range(n - 1, 0, -1):
            for i in range(k):
                y = y.index_put((torch.tensor([k - 1]), torch.tensor([i])), 
                                (y[k, i] - y[k, k] * y[k, k - 1 - i]) / (1 - torch.pow(y[k, k], 2)))
        r = torch.diagonal(y, dim1=0, dim2=1).permute(2, 0, 1)
        x = r / torch.sqrt(1 - torch.pow(r, 2))
        return x.flip(dims=[0]).permute(1, 0, 2).unsqueeze(dim=-1)
    
    def posterior(self, param):
        if param == 'factor':
            return MultivariateNormal(
                mean=self.variational_params['factor_mean'],
                cov=self.flat_to_cov(self.variational_params['factor_cov']),
            ).to(self.device)
            
        elif param == 'factor_loadings':
            return MatrixNormal(
                mean=self.variational_params['factor_loadings_mean'],
                left_cov=self.flat_to_cov(self.variational_params['factor_loadings_left_cov']),
                right_cov=self.flat_to_cov(self.variational_params['factor_loadings_right_cov']), 
            ).to(self.device)
        
        elif param == 'factor_transition':
            return MatrixNormal(
                mean=self.variational_params['factor_transition_mean'],
                left_cov=self.flat_to_cov(self.variational_params['factor_transition_left_cov']),
                right_cov=self.flat_to_cov(self.variational_params['factor_transition_right_cov']),
            ).to(self.device)
            
        elif param == 'factor_transition_cov_const':
            return (
                InverseWishart(
                    scale=self.flat_to_cov(self.variational_params['factor_transition_cov_const_scale']),
                    df=self.k_factors + 2,
                ) if bool(self.arch_order or self.garch_order) else
                DiracDelta(
                    point=self.flat_to_cov(self.variational_params['factor_transition_cov_const_scale']),
                )
            ).to(self.device)
        
        elif param == 'factor_transition_cov_arch':
            return DiagNormal(
                mean=self.variational_params['factor_transition_cov_arch_mean'],
                std=torch.exp(self.variational_params['factor_transition_cov_arch_log_std']),
            ).to(self.device)
            
        elif param == 'factor_transition_cov_garch':
            return DiagNormal(
                mean=self.variational_params['factor_transition_cov_garch_mean'],
                std=torch.exp(self.variational_params['factor_transition_cov_garch_log_std']),
            ).to(self.device)
        
        elif param == 'error_transition':
            return MultivariateNormal(
                mean=self.variational_params['error_transition_mean'],
                cov=torch.diag_embed(torch.exp(self.variational_params['error_transition_log_var']), dim1=-2, dim2=-1),
            ).to(self.device)

        else:
            raise ValueError(f'Parameter {param} not found')    
    
    def log_likelihood(self, sample_size, factor, factor_loadings, factor_transition, factor_transition_cov_const, 
                       factor_transition_cov_arch, factor_transition_cov_garch, error_transition):
        factor_mean = torch.zeros(sample_size, self.T + 1, self.k_factors, 1, device=self.device)
        if self.factor_order:
            shifted_factor = self.shift(factor, shifts=self.factor_order, dim=1)
            stacked_shifted_factor = shifted_factor.unfold(dimension=1, size=self.factor_order, step=1)[:, :-1].permute(0, 1, 4, 2, 3)
            factor_mean = (
                factor_mean +
                torch.einsum('...pij,...tpjk->...tik',
                             factor_transition,
                             stacked_shifted_factor,
                             )
            )
        
        factor_cov = factor_transition_cov_const.unsqueeze(dim=1).repeat_interleave(repeats=self.T + 1, dim=1)
        if self.arch_order:
            factor_error = factor - factor_mean
            shifted_factor_error = self.shift(factor_error, shifts=self.arch_order, dim=1)
            stacked_shifted_factor_error = shifted_factor_error.unfold(dimension=1, size=self.arch_order, step=1)[:, :-1].permute(0, 1, 4, 2, 3)
            factor_cov[:, 1:] = (
                factor_cov[:, 1:] +
                torch.einsum('...rij,...trjk,...trkl,...rlm->...tim',
                             factor_transition_cov_arch,
                             stacked_shifted_factor_error[:, 1:],
                             stacked_shifted_factor_error[:, 1:].transpose(-2, -1),
                             factor_transition_cov_arch.transpose(-2, -1),
                             )
            )
        if self.garch_order:
            factor_cov_ = [factor_cov[:, 0] for _ in range(self.garch_order + 1)]
            for t in range(1, self.T + 1):
                factor_cov_.append(
                    factor_cov[:, t] +
                    torch.einsum('...sij,...sjk,...skl->...il',
                                 factor_transition_cov_garch,
                                 torch.stack(factor_cov_[t : t + self.garch_order], dim=1),
                                 factor_transition_cov_garch.transpose(-2, -1),
                                 )
                )
            factor_cov = torch.stack(factor_cov_[self.garch_order:], dim=1)
        
        endog_mean = torch.einsum('...ij,...tjk->...tik',
                                  factor_loadings,
                                  factor,
                                  )
        if self.error_order:
            error = self.endog - endog_mean
            shifted_error = self.shift(error, shifts=self.error_order, dim=1)
            stacked_shifted_error = shifted_error.unfold(dimension=1, size=self.error_order, step=1)[:, :-1].permute(0, 1, 4, 2, 3)
            endog_mean = (
                endog_mean +
                torch.einsum('...qij,...tqjk->...tik',
                             torch.diag_embed(error_transition.squeeze(dim=-1), dim1=-2, dim2=-1),
                             stacked_shifted_error,
                             )
            )
            
        endog_cov = torch.diag_embed(
            torch.exp(self.variational_params['error_transition_cov_log_var']), 
            dim1=-2, dim2=-1).repeat(sample_size, self.T + 1, 1, 1)
        
        factor_dist = MultivariateNormal(
            mean=factor_mean,
            cov=self.safe_pd(factor_cov),
        ).to(self.device)
        endog_dist = MultivariateNormal(
            mean=endog_mean[:, 1:],
            cov=self.safe_pd(endog_cov[:, 1:]),
        ).to(self.device)
        
        return (
            endog_dist.log_prob(self.endog[1:].repeat(sample_size, 1, 1, 1)).mean(dim=0) +
            factor_dist.log_prob(factor).mean(dim=0) + 
            self.prior['factor_loadings'].log_prob(factor_loadings).mean(dim=0) + 
            self.prior['factor_transition'].log_prob(factor_transition).mean(dim=0) + 
            self.prior['factor_transition_cov_const'].log_prob(factor_transition_cov_const).mean(dim=0) + 
            self.prior['factor_transition_cov_arch'].log_prob(factor_transition_cov_arch).mean(dim=0) + 
            self.prior['factor_transition_cov_garch'].log_prob(factor_transition_cov_garch).mean(dim=0) + 
            self.prior['error_transition'].log_prob(self.unconstrain_ar_coeffs(error_transition)).mean(dim=0)
        )
    
    def sample_elbo(self, sample_size=1):
        samples = {}
        log_posterior = 0
        for param in self.variational_param_names:
            param_dist = self.posterior(param)
            samples[param] = param_dist.sample(sample_size)
            log_posterior = log_posterior + param_dist.log_prob(samples[param]).mean(dim=0)
        samples['error_transition'] = self.constrain_ar_coeffs(samples['error_transition'])
        
        return self.log_likelihood(sample_size, **samples) - log_posterior
