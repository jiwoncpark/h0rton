import torch

class GaussianNLL:
    def __init__(self, cov_mat, y_dim, out_dim, device):
        self.cov_mat = cov_mat
        self.y_dim = y_dim
        self.out_dim = out_dim
        self.device = device

    def __call__(self, pred, target):
        d = self.y_dim # for readability

        if self.cov_mat == 'diagonal':
            mu = pred[:, :d]
            logvar = pred[:, d:]
            return self.nll_diagonal(target, mu, logvar)

        elif self.cov_mat == 'low_rank':
            mu = pred[:, :d]
            logvar = pred[:, d:2*d]
            F = pred[:, 2*d:]
            return self.nll_lowrank(target, mu, logvar, F, reduce=True)

        elif self.cov_mat == 'mixture':
            # Boring and arbitrary slicing... bear with me...
            mu = pred[:, :d]
            logvar = pred[:, d:2*d]
            F = pred[:, 2*d:4*d]
            mu2 = pred[:, 4*d:5*d]
            logvar2 = pred[:, 5*d:6*d]
            F2 = pred[:, 6*d:8*d]
            alpha = pred[:, -1]
            return self.nll_mixture(target, mu, logvar, F, mu2, logvar2, F2, alpha)

    def nll_diagonal(self, target, mu, logvar):
        precision = torch.exp(-logvar)
        return torch.mean(torch.sum(precision * (target - mu)**2.0 + logvar, dim=1), dim=0)

    def nll_lowrank(self, target, mu, logvar, F, reduce=True):
        # 1/(Y_dim - 1) * (sq_mahalanobis + log(det of \Sigma))
        batch_size, _ = target.shape # self.y_dim = Y_dim - 1
        rank = 2
        F = F.reshape([batch_size, self.y_dim, rank]) # FIXME: hardcoded for rank 2
        inv_var = torch.exp(-logvar) # [batch_size, self.y_dim]
        diag_inv_var = torch.diag_embed(inv_var)  # [batch_size, self.y_dim, self.y_dim]
        diag_prod = F**2.0 * inv_var.reshape([batch_size, self.y_dim, 1]) # [batch_size, self.y_dim, rank] after broadcasting
        off_diag_prod = torch.prod(F, dim=2)*inv_var # [batch_size, self.y_dim]
        #batchdiag = torch.diag_embed(torch.exp(logvar)) # [batch_size, self.y_dim, self.y_dim]
        #batch_eye = torch.eye(rank).reshape(1, rank, rank).repeat(batch_size, 1, 1) # [batch_size, rank, rank]
        #assert batchdiag.shape == torch.Size([batch_size, self.y_dim, self.y_dim])

        # (25), (26) in Miller et al 2016
        log_det = torch.sum(logvar, dim=1) # [batch_size]
        M00 = torch.sum(diag_prod[:, :, 0], dim=1) + 1.0 # [batch_size]
        M11 = torch.sum(diag_prod[:, :, 1], dim=1) + 1.0 # [batch_size]
        M12 = torch.sum(off_diag_prod, dim=1) # [batch_size]
        assert M00.shape == torch.Size([batch_size])
        assert M12.shape == torch.Size([batch_size])
        det_M = M00*M11 - M12**2.0 # [batch_size]
        assert det_M.shape == torch.Size([batch_size])
        assert log_det.shape == torch.Size([batch_size])
        log_det += torch.log(det_M) 
        assert log_det.shape == torch.Size([batch_size])
        #print(det_M)

        inv_M = torch.ones([batch_size, rank, rank], device=self.device)
        inv_M[:, 0, 0] = M11
        inv_M[:, 1, 1] = M00
        inv_M[:, 1, 0] = -M12
        inv_M[:, 0, 1] = -M12
        inv_M /= det_M.reshape(batch_size, 1, 1)

        # (27) in Miller et al 2016
        inv_cov = diag_inv_var - torch.bmm(torch.bmm(torch.bmm(torch.bmm(diag_inv_var, F), inv_M), torch.transpose(F, 1, 2)), diag_inv_var) 
        assert inv_cov.shape == torch.Size([batch_size, self.y_dim, self.y_dim])
        sq_mahalanobis = torch.squeeze(torch.bmm(torch.bmm((mu - target).reshape(batch_size, 1, self.y_dim), inv_cov), (mu - target).reshape(batch_size, self.y_dim, 1)))
        assert sq_mahalanobis.shape == torch.Size([batch_size])

        if reduce==True:
            return torch.mean(sq_mahalanobis + log_det, dim=0)
        else:
            return sq_mahalanobis + log_det

    def nll_mixture(self, target, mu, logvar, F, mu2, logvar2, F2, alpha):
        batch_size, _ = target.shape
        rank = 2
        log_nll = torch.empty([batch_size, rank], device=self.device)
        logsigmoid = torch.nn.LogSigmoid()
        alpha = alpha.reshape(-1)
        log_nll[:, 0] = torch.log(torch.tensor([0.5], device=self.device)) + logsigmoid(-alpha) + self.nll_lowrank(target, mu, logvar, F=F, reduce=False) # [batch_size]
        log_nll[:, 1] = torch.log(torch.tensor([0.5], device=self.device)) + logsigmoid(alpha) + self.nll_lowrank(target, mu2, logvar2, F=F2, reduce=False) # [batch_size]
        sum_two_gaus = torch.logsumexp(log_nll, dim=1) 
        return torch.mean(sum_two_gaus)