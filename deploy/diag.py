import os, sys, glob
sys.path.insert(0, "/home/phamhaisen/OpenPCDet")
os.chdir("/home/phamhaisen/OpenPCDet/tools")
from pcdet.config import cfg, cfg_from_yaml_file
cfg_from_yaml_file("cfgs/nuscenes_models/transfusion_lidar.yaml", cfg)
print("POINT_CLOUD_RANGE:", list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE))
for p in cfg.DATA_CONFIG.DATA_PROCESSOR:
    print("PROC:", {k: p[k] for k in p})
import numpy as np
sys.path.insert(0, "/home/phamhaisen/auto-label-3d")
import run as R
pcds = sorted(glob.glob("/home/phamhaisen/data_day14/pointcloud/*.pcd"))
pts = R.load_pcd(pcds[0])
print("points:", pts.shape, pts.dtype, "nan?", np.isnan(pts).any(), "inf?", np.isinf(pts).any())
print("min:", pts.min(0).tolist(), "max:", pts.max(0).tolist())
import spconv, cumm
print("spconv:", getattr(spconv, "__version__", "?"), spconv.__file__)
print("cumm:", getattr(cumm, "__version__", "?"))
