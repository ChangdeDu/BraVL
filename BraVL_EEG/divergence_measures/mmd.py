def mmd_loss(z_tilde, z, z_var):
    r"""Calculate maximum mean discrepancy described in the WAE paper.
    Args:
        z_tilde (Tensor): samples from deterministic non-random encoder Q(Z|X).
            2D Tensor(batch_size x dimension).
        z (Tensor): samples from prior distributions. same shape with z_tilde.
        z_var (Number): scalar variance of isotropic gaussian prior P(Z).
    """
    assert z_tilde.size() == z.size()
    assert z.ndimension() == 2

    n = z.size(0)
    im_kernel_z_z = im_kernel_sum(z, z, z_var, exclude_diag=True).div(n*(n-1))
    im_kernel_ztilde_ztilde = im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div(n*(n-1))
    im_kernel_z_ztilde = im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div(n*n).mul(2)
    out = im_kernel_z_z + im_kernel_ztilde_ztilde - im_kernel_z_ztilde
    return out, im_kernel_z_z, im_kernel_ztilde_ztilde, im_kernel_z_ztilde


def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
    r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.
    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2*z_dim*z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()
    return kernel_sum