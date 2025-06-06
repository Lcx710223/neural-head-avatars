### LCX:render_from_npz.py 根据已有的NPZ数据，在FLAME里重建3D人头并渲染显示。

import numpy as np
import torch
from vht.model.flame import FlameHead, FlameTex
from vht.util.render import SHRenderer
import matplotlib.pyplot as plt

# ========== 1. 加载NPZ参数 ==========
npz_path = "tracked_flame_params.npz"  # 你的NPZ文件路径
params = np.load(npz_path)

# ========== 2. 实例化FLAME模型 ==========
# 读取参数维度
n_shape = params['shape'].shape[-1]
n_expr = params['expr'].shape[-1]
n_tex = params['texture'].shape[-1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 实例化FLAME模型和纹理
flame = FlameHead(n_shape, n_expr).to(device)
flame_tex = FlameTex(n_tex).to(device)

# ========== 3. 取出一帧参数，转为Tensor ==========
frame_idx = 0  # 可选其他帧
def npz_to_tensor(key, idx=None):
    arr = params[key]
    # 如果是多帧的参数，选取一帧
    if arr.ndim > 1:
        arr = arr[idx]
    return torch.from_numpy(arr).float().unsqueeze(0).to(device)

# 按FLAME要求准备参数
shape = npz_to_tensor('shape')
expr = npz_to_tensor('expr', frame_idx)
rotation = npz_to_tensor('rotation', frame_idx)
neck = npz_to_tensor('neck_pose', frame_idx)
jaw = npz_to_tensor('jaw_pose', frame_idx)
eyes = npz_to_tensor('eyes_pose', frame_idx)
trans = npz_to_tensor('translation', frame_idx)
texture = npz_to_tensor('texture')
light = npz_to_tensor('light', frame_idx)

# ========== 4. 前向生成3D人头 ==========
with torch.no_grad():
    verts, lmks = flame(shape, expr, rotation, neck, jaw, eyes, trans)
    albedo = flame_tex(texture)

# ========== 5. 渲染 ==========
renderer = SHRenderer().to(device)
faces = flame.faces.unsqueeze(0)
uvcoords = flame.face_uvcoords.unsqueeze(0)
with torch.no_grad():
    image = renderer.render_single_view(verts, faces, uvcoords, albedo, light)
    # image: (1, H, W, 3) or (1, H, W, 4)

# ========== 6. 显示 ==========
img = image[0].detach().cpu().numpy()
if img.shape[-1] == 4:
    img = img[..., :3] * img[..., 3:4] + 1 - img[..., 3:4]  # 使用Alpha混合到白底
plt.imshow(np.clip(img, 0, 1))
plt.axis("off")
plt.title("FLAME Head Rendered from NPZ")
plt.show()
