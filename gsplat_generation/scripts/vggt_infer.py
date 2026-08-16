import sys, glob, time
import torch
sys.path.insert(0, "/workspace/vggt")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print(f"device={device} dtype={dtype}")

t0 = time.time()
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()
print(f"model loaded in {time.time()-t0:.1f}s, on {next(model.parameters()).device}")

frames_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/timebox_smoketest/frames"
out_path = sys.argv[2] if len(sys.argv) > 2 else "/workspace/timebox_smoketest/vggt_output.pt"
max_images = int(sys.argv[3]) if len(sys.argv) > 3 else 0
image_names = sorted(
    glob.glob(f"{frames_dir}/*.png")
    + glob.glob(f"{frames_dir}/*.jpg")
    + glob.glob(f"{frames_dir}/*.jpeg")
    + glob.glob(f"{frames_dir}/*.JPG")
    + glob.glob(f"{frames_dir}/*.JPEG")
)
if max_images > 0 and len(image_names) > max_images:
    idx = [round(i * (len(image_names) - 1) / (max_images - 1)) for i in range(max_images)]
    image_names = [image_names[i] for i in idx]
print(f"num images: {len(image_names)}")
if image_names:
    print(f"first: {image_names[0]}")
    print(f"last:  {image_names[-1]}")

images = load_and_preprocess_images(image_names).to(device)
print(f"images tensor: {images.shape}")

t0 = time.time()
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images_b = images[None]  # add batch dim
        aggregated_tokens_list, ps_idx = model.aggregator(images_b)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_b.shape[-2:])
        point_map, point_conf = model.point_head(aggregated_tokens_list, images_b, ps_idx)
print(f"inference done in {time.time()-t0:.1f}s")

print("extrinsic shape:", extrinsic.shape)
print("intrinsic shape:", intrinsic.shape)
print("point_map shape:", point_map.shape)

print("\nfirst camera extrinsic (world-to-cam 3x4):")
print(extrinsic[0, 0])
print("\nfirst camera intrinsic:")
print(intrinsic[0, 0])

cam_positions = -torch.matmul(extrinsic[0, :, :, :3].transpose(-1, -2), extrinsic[0, :, :, 3:4]).squeeze(-1)
print("\ncamera positions (world coords) per frame:")
print(cam_positions.float().cpu().numpy())

torch.save({
    "extrinsic": extrinsic.cpu(),
    "intrinsic": intrinsic.cpu(),
    "point_map": point_map.cpu(),
    "point_conf": point_conf.cpu(),
}, out_path)
print(f"\nsaved to {out_path}")
