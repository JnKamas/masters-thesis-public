import open3d as o3d
import cv2
import numpy as np
import os

def load_exr_points(exr_path):
    import OpenEXR, Imath, numpy as np
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    channels = file.channels(["R", "G", "B"], pt)
    xyz = [np.frombuffer(c, dtype=np.float32) for c in channels]
    xyz = np.stack(xyz, axis=-1).reshape(size[1], size[0], 3)
    xyz = xyz.reshape(-1, 3)
    xyz = xyz[~np.isnan(xyz).any(axis=1)]
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))



def read_transform_file(file):
    with open(file, 'r') as f:
        P = f.readline().strip().replace(",", " ").split()
    R = np.array([[float(P[0]), float(P[4]), float(P[8])],
                  [float(P[1]), float(P[5]), float(P[9])],
                  [float(P[2]), float(P[6]), float(P[10])]])
    t = np.array([float(P[12]), float(P[13]), float(P[14])])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

# === CONFIG ===
dataset_root = "/home/k/kamas7/data/complete"
scan_num = "000"   # change to any scan you want
obj_name = "dataset0"  # adjust to match your folder
pred_file = f"/home/k/kamas7/thesis/inference/quartres_resnet34_synth/{obj_name}/prediction_scan_{scan_num}.txt"

stl_path = os.path.join(dataset_root, obj_name, "bin.stl")
exr_path = os.path.join(dataset_root, obj_name, f"scan_{scan_num}_positions.exr")

print("[INFO] Loading data...")
scene = load_exr_points(exr_path)
mesh = o3d.io.read_triangle_mesh(stl_path)
model = mesh.sample_points_uniformly(50000)
model.scale(1000.0, center=(0,0,0))  # STL likely in meters

pred_T = read_transform_file(pred_file)
model.transform(pred_T)

scene.paint_uniform_color([0.7, 0.7, 0.7])
model.paint_uniform_color([1, 0, 0])

print("[INFO] Opening window — gray = scan, red = model.")
o3d.visualization.draw_geometries([scene, model])
