import torch.nn.functional as F
import torch.nn as nn





loss_attenion_cri = nn.MSELoss()
def Attention_compare(scoremap1, scoremap2, mask):
    if mask is None:
        return 0
    localization_map_normed_upsample1 = F.upsample_bilinear(scoremap1, mask.shape[1:])
    localization_map_normed_upsample2 = F.upsample_bilinear(scoremap2, mask.shape[1:])
    mask = mask[:, None, ...]
    localization_map_normed_upsample2 = localization_map_normed_upsample1 * (1 - mask) + localization_map_normed_upsample2 * mask
    attention_loss = loss_attenion_cri(localization_map_normed_upsample1, localization_map_normed_upsample2)
    return attention_loss