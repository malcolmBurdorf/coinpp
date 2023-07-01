import coinpp.conversion as conversion
import torch
import torch.nn.functional as F
from einops import rearrange

def gradncp_sample(inputs, func_rep, data_ratio=0.5):
    ratio = data_ratio
    meta_batch_size = inputs.size(0)
    # equivalent to self.grid (but coordinates between -1 and 1)
    coords = rearrange(conversion.shape2coordinates((inputs.shape[1], inputs.shape[2])), 'h w c -> (h w) c')
    coords = coords.clone().detach()[None, ...].repeat((meta_batch_size,) + (1,) * len(coords.shape)).to(inputs.device)
    with torch.no_grad():
        out, feature = func_rep(coords, get_penult_features=True)
        if 'img' in ['img']:
            out = rearrange(out, 'b hw c -> b c hw')
            feature = rearrange(feature, 'b hw f -> b f hw')
            # features are ordered b h w c in coin++, as opposed to b c h w in gradncp (shouldn't be consequential)
            inputs = rearrange(inputs, 'b h w c -> b c (h w)')
        else:
            raise NotImplementedError()
        error = inputs - out  # b c (hw)
        gradient = -1 * feature.unsqueeze(dim=1) * error.unsqueeze(dim=2)  # b c f hw
        gradient_bias = -1 * error.unsqueeze(dim=2)  # b c hw
        gradient = torch.cat([gradient, gradient_bias], dim=2)
        gradient = rearrange(gradient, 'b c f hw -> b (c f) hw')
        gradient_norm = torch.norm(gradient, dim=1)  # b hw
        coords_len = gradient_norm.size(1)

    gradncp_index = torch.sort(
        gradient_norm, dim=1, descending=True
    )[1][:, :int(coords_len * ratio)]  # b int(hw * ratio)

    # for images: dim_in=2 and dim_out=3
    gradncp_coord = torch.gather(
        coords, 1, gradncp_index.unsqueeze(dim=2).repeat(1, 1, 2)
    )
    gradncp_index = gradncp_index.unsqueeze(dim=1).repeat(1, 3, 1)

    return gradncp_coord, gradncp_index


def param_consistency(params, params_bootstrap, bs):
    '''Only one weight layer (the modulations) is modified here''' 
    updated_param = params_bootstrap.detach() - params
    updated_param = updated_param.view(bs, -1)
    param_norm = torch.norm(updated_param, p=2, dim=1).mean()
    return param_norm
