'''
DeepVtx model -- VERBATIM copy of toolkit pyutil/python/SCN/DeepVtx.py
(apply-pointcloud HEAD, 2026-08-14) so a fine-tuned checkpoint is loadable by
the in-tree SCN_Vertex._load_model without any code change on the wire-cell
side.  Do not "improve" the architecture here: the class must stay
state_dict-compatible with wire-cell-data/uboone/scn_vtx/
t48k-m16-l5-lr5d-res0.5-CP24.pth (137 tensors, 7,183,122 params).

load_weights() replicates SCN_Vertex.py:20-24: the uBooNE checkpoint stores
every conv kernel 4-D (27, 1, C_in, C_out) while DeepVtx expects 3-D
(27, C_in, C_out), so squeeze(dim=1) fires on ALL conv tensors -- it is the
normal load path, not a fallback.  A checkpoint saved from this model is in
the squeezed 3-D form, which the in-tree loader accepts as-is (its shape test
is per-tensor).
'''
import torch
import torch.nn as nn
import sparseconvnet as scn

reps = 2 # Conv block repetition factor
m = 16 # Unet number of features
nPlanes = [m, 2*m, 4*m, 8*m, 16*m] # UNet number of features per level

class DeepVtx(nn.Module):
    '''
    spatialSize seems need to be 2^n
    '''
    def __init__(self,  dimension = 3, device = 'cuda', spatialSize = 4096, nIn = 3, nClasses = 2):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension, torch.LongTensor([spatialSize]*3), mode=3)).add(
            scn.SubmanifoldConvolution(dimension, nIn, m, 3, False)).add(
            scn.UNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[2,2])).add(
            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(dimension)).to(device)
        self.inputLayer = scn.InputLayer(dimension, torch.LongTensor([spatialSize]*3), mode=3)
        self.linear = nn.Linear(m, nClasses).to(device)
    def forward(self,x):
        x=self.sparseModel(x)
        x=self.linear(x)
        x=torch.sigmoid(x)
        return x


def make_model(device='cpu'):
    """Production-shape model: nIn=1 (confirmed by the checkpoint's first conv
    in-channels), 2 classes, 4096^3 voxel domain (2048 cm/axis at 0.5 cm)."""
    return DeepVtx(dimension=3, nIn=1, device=device)


def load_weights(model, weights_path):
    """Load a state_dict, squeezing the historical extra dim=1 where present
    (mirror of toolkit SCN_Vertex.py:_load_model)."""
    trained_dict = torch.load(weights_path, weights_only=True, map_location='cpu')
    model_dict = model.state_dict()
    for param_tensor in trained_dict:
        if trained_dict[param_tensor].shape != model_dict[param_tensor].shape:
            trained_dict[param_tensor] = torch.squeeze(trained_dict[param_tensor], dim=1)
    model.load_state_dict(trained_dict)
    return model


def predict_scores(model, coords_np, ft_np, device='cpu'):
    """Run the net; return the per-voxel score sigmoid(vtx)-sigmoid(bg) in
    [-1,+1], exactly as SCN_Vertex.py:78-82.  coords stay on CPU (SCN's
    InputLayer requirement); features go to `device`."""
    coords = torch.LongTensor(coords_np)
    ft = torch.FloatTensor(ft_np).to(device)
    with torch.no_grad():
        prediction = model([coords, ft])
    pred_np = prediction.cpu().numpy()
    return pred_np[:, 1] - pred_np[:, 0]
