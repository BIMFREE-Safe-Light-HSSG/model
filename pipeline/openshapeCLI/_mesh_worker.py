"""
_mesh_worker.py — 메시 복원 전용 워커 (서브프로세스로 실행)
segfault가 발생해도 호출 프로세스가 죽지 않도록 격리한다.

호출:
  python _mesh_worker.py <input.npy> <output.npy> <n_sample>
  input.npy  : [N, 6] float32 XYZ+RGB
  output.npy : [n_sample, 6] float32 (성공 시)  / 저장 안 됨 (실패 시)
"""
import sys
import numpy as np

def run(input_path, output_path, n_sample):
    pts = np.load(input_path)
    xyz = pts[:, :3].copy()
    rgb = pts[:, 3:].copy()
    n_pts = len(pts)

    import open3d as o3d
    from scipy.spatial import cKDTree

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    bbox_diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    radius_n  = max(bbox_diag * 0.08, 0.02)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_n, max_nn=30)
    )

    mesh = None

    # 1차: Alpha Shape (가장 안정적)
    try:
        alpha = bbox_diag * 0.12
        mesh  = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha)
        if len(mesh.triangles) < 50:
            mesh = None
        else:
            print(f"[worker] Alpha shape 성공: {len(mesh.triangles)}개", flush=True)
    except Exception as e:
        print(f"[worker] Alpha shape 실패: {e}", flush=True)
        mesh = None

    # 2차: BPA
    if mesh is None:
        try:
            r = bbox_diag * 0.05
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector([r, r*2, r*4]))
            if len(mesh.triangles) < 50:
                mesh = None
            else:
                print(f"[worker] BPA 성공: {len(mesh.triangles)}개", flush=True)
        except Exception as e:
            print(f"[worker] BPA 실패: {e}", flush=True)
            mesh = None

    if mesh is None:
        print("[worker] 메시 복원 실패", flush=True)
        sys.exit(1)

    # 면적 가중 샘플링
    sampled  = mesh.sample_points_uniformly(number_of_points=n_sample)
    xyz_mesh = np.asarray(sampled.points, dtype=np.float32)

    tree = cKDTree(xyz)
    _, idx = tree.query(xyz_mesh, k=1)
    rgb_mesh = rgb[idx].astype(np.float32)

    result = np.concatenate([xyz_mesh, rgb_mesh], axis=1)
    np.save(output_path, result)
    print(f"[worker] 저장 완료: {output_path}  shape={result.shape}", flush=True)
    sys.exit(0)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python _mesh_worker.py <input.npy> <output.npy> <n_sample>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
