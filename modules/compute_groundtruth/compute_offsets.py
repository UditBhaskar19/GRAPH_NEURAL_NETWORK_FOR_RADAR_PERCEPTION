# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : normalize and unnormalize gt offsets
# ---------------------------------------------------------------------------------------------------------------

def normalize_gt_offsets(gt_offsets_img, offset_mu, offset_sigma):
    mu_x, mu_y = offset_mu[0], offset_mu[1]
    sigma_x, sigma_y = offset_sigma[0], offset_sigma[1]
    gt_offsets_img[..., 0] = (gt_offsets_img[..., 0] - mu_x) / sigma_x
    gt_offsets_img[..., 1] = (gt_offsets_img[..., 1] - mu_y) / sigma_y
    return gt_offsets_img

def unnormalize_gt_offsets(offsets_img, offset_mu, offset_sigma):
    mu_x, mu_y = offset_mu[0], offset_mu[1]
    sigma_x, sigma_y = offset_sigma[0], offset_sigma[1]
    offsets_img[..., 0] = offsets_img[..., 0] * sigma_x + mu_x
    offsets_img[..., 1] = offsets_img[..., 1] * sigma_y + mu_y
    return offsets_img