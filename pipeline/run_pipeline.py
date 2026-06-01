"""
SuperHSSG 전체 파이프라인 자동 실행
=====================================
실행 순서:
  0. sam_room_grid.py              → room_grid_{stem}.npz
  1. merge/stage2_s1.py
       + merge/s2_to_l1.py        → merge/stage2_results/l1_{stem}.npz
  2. openshapeCLI/stage3_multiview.py (기본)
     또는 stage3_openshape.py      → openshapeCLI/stage3_results/f_l1_floor.npy
  3. sem_merge/stage4_sem_merge.py → sem_merge/stage4_results/floor_l2*.npy
  4. bottomup/stage5_bottomup.py   → bottomup/stage5_results/floor_h_l*.npy
  5. topdown/stage6_topdown.py     → results/{stem}_scene_graph.json

사용법:
  python run_pipeline.py --src src/floor.npz
  python run_pipeline.py --src src/6_floor.npz
  python run_pipeline.py --from_stage 2
  python run_pipeline.py --only 3
  python run_pipeline.py --skip 0
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parent   # model/pipeline/
MERGE = ROOT / 'merge'
OPEN  = ROOT / 'openshapeCLI'
SEM   = ROOT / 'sem_merge'
BOT   = ROOT / 'bottomup'
TOP   = ROOT / 'topdown'

SRC_NPZ         = None
ROOM_GRID       = None
OUT_SCENE_GRAPH = None
_VIZ_L1         = False
_MULTIVIEW      = True
_CLUTTER_SIM_THR = 0.5
_CLIP_THRESHOLD  = 0.3

STAGE2_DIR  = MERGE / 'stage2_results'
L1_NPZ      = None
STAGE3_DIR  = OPEN  / 'stage3_results'
STAGE4_DIR  = SEM   / 'stage4_results'
STAGE5_DIR  = BOT   / 'stage5_results'
RESULTS_DIR = ROOT  / 'results'

PY = sys.executable


# ── src 파일명 → 출력 이름 유도 ────────────────────────────────────────────────
def _derive_out_name(src_path: Path) -> str:
    stem = src_path.stem
    if stem == 'floor':
        return 'scene_graph.json'
    if stem.endswith('_floor'):
        prefix = stem[:-len('_floor')]
        return f'{prefix}_scene_graph.json'
    return f'{stem}_scene_graph.json'


def _resolve_src(src_arg: str) -> Path:
    p = Path(src_arg)
    if p.is_absolute():
        return p
    candidate = ROOT / 'src' / p
    if candidate.exists():
        return candidate
    return ROOT / p


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def header(stage_num: int, name: str):
    line = "=" * 60
    print(f"\n{line}", flush=True)
    print(f"  STAGE {stage_num}: {name}", flush=True)
    print(f"{line}", flush=True)


def run(cmd: list, cwd: Path = ROOT) -> bool:
    print(f"\n▶ {' '.join(str(c) for c in cmd)}\n", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n✗ 실패 (종료 코드 {result.returncode}, {elapsed:.1f}s)", flush=True)
        return False
    print(f"\n✓ 완료 ({elapsed:.1f}s)", flush=True)
    return True


def check_input(path: Path, label: str) -> bool:
    if not path.exists():
        print(f"  [오류] 입력 파일 없음: {path}  ({label})")
        return False
    return True


# ── 각 스테이지 ────────────────────────────────────────────────────────────────
def stage0_wall_door():
    header(0, "sam_room_grid  (SAM 기반 방 마스크 생성)")
    if not check_input(SRC_NPZ, str(SRC_NPZ)):
        return False
    return run([PY, str(ROOT / 'sam_room_grid.py'),
                '--npz', str(SRC_NPZ),
                '--out_dir', str(ROOT)], cwd=ROOT)


def stage1_merge():
    header(1, "stage2_s1  (Jaccard 슈퍼포인트 병합)")
    if not check_input(SRC_NPZ, str(SRC_NPZ)):
        return False

    edge_path = SRC_NPZ.parent / f'{SRC_NPZ.stem}_edges.npz'
    if not edge_path.exists():
        print(f"\n  [KNN 엣지] {edge_path.name} 없음 → 자동 생성 중...")
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        d = np.load(SRC_NPZ)
        centroids = d['sp_features'][:, 0:3].astype(np.float32)
        n_sp = len(centroids)
        k = min(8, n_sp - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(centroids)
        distances, indices = nn.kneighbors(centroids)
        src_list, dst_list = [], []
        for i, (dists, neighbors) in enumerate(zip(distances, indices)):
            for dist, j in zip(dists[1:], neighbors[1:]):
                if dist <= 1.5:
                    src_list.append(i)
                    dst_list.append(int(j))
        import numpy as np
        edge_src = np.array(src_list, dtype=np.int64)
        edge_dst = np.array(dst_list, dtype=np.int64)
        np.savez_compressed(edge_path, edge_src=edge_src, edge_dst=edge_dst)
        print(f"  [KNN 엣지] {len(edge_src):,}개 생성 → {edge_path.name}")

    s2_out = SRC_NPZ.parent / f'{SRC_NPZ.stem}_s2.npz'
    if not run([PY, str(MERGE / 'stage2_s1.py'), '--npz', str(SRC_NPZ)], cwd=ROOT):
        return False

    return run([
        PY, str(MERGE / 's2_to_l1.py'),
        '--s2',  str(s2_out),
        '--out', str(L1_NPZ),
    ], cwd=ROOT)


def stage2_openshape():
    if _MULTIVIEW:
        header(2, "stage3_multiview  (2D MultiView CLIP + OpenShape 3D 폴백)")
    else:
        header(2, "stage3_openshape  (OpenShape 3D 의미 임베딩)")

    if not check_input(L1_NPZ, "stage2 출력"):
        return False
    if not check_input(SRC_NPZ, str(SRC_NPZ)):
        return False

    if _MULTIVIEW:
        ok = run([
            PY, str(OPEN / 'stage3_multiview.py'),
            '--floor_npz',            str(SRC_NPZ),
            '--l1_npz',               str(L1_NPZ),
            '--out_dir',              str(STAGE3_DIR),
            '--clutter_sim_threshold', str(_CLUTTER_SIM_THR),
            '--no_lvis',
        ], cwd=ROOT)
    else:
        ok = run([
            PY, str(OPEN / 'stage3_openshape.py'),
            '--floor_npz', str(SRC_NPZ),
            '--l1_npz',    str(L1_NPZ),
            '--out_dir',   str(STAGE3_DIR),
            '--threshold', str(_CLIP_THRESHOLD),
        ], cwd=ROOT)

    if not ok:
        return False
    _patch_labels_json(STAGE3_DIR / 'l1_labels_text.json')
    return True


def _patch_labels_json(json_path: Path):
    import json as _json
    if not json_path.exists():
        return
    data = _json.loads(json_path.read_text(encoding='utf-8'))
    n_l1    = data['n_l1']
    objects = data['objects']
    present = {o['l1_id'] for o in objects}
    missing = [i for i in range(n_l1) if i not in present]
    if not missing:
        return
    extras = [{'l1_id': lid, 'label': 'structural',
                'source': 'structural', 'score': None, 'top3': None}
               for lid in missing]
    data['objects']   = sorted(objects + extras, key=lambda x: x['l1_id'])
    data['n_objects'] = len(data['objects'])
    json_path.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  [labels 보충] structural {len(missing)}개 추가 → 총 {n_l1}개")


def stage3_sem_merge():
    header(3, "stage4_sem_merge  (방별 L2 병합)")
    fl1_npy    = STAGE3_DIR / 'f_l1_floor.npy'
    labels_json = STAGE3_DIR / 'l1_labels_text.json'
    for p, lbl in [(fl1_npy, "stage3 f_l1"), (L1_NPZ, "stage2 l1"), (labels_json, "stage3 labels")]:
        if not check_input(p, lbl):
            return False

    cmd = [
        PY, str(SEM / 'stage4_sem_merge.py'),
        '--fl1_npy',     str(fl1_npy),
        '--l1_npz',      str(L1_NPZ),
        '--labels_json', str(labels_json),
        '--out_dir',     str(STAGE4_DIR),
    ]
    if ROOM_GRID and ROOM_GRID.exists():
        cmd += ['--room_grid', str(ROOM_GRID)]
    return run(cmd, cwd=ROOT)


def stage4_bottomup():
    header(4, "stage5_bottomup  (Bottom-up 의미 집계)")
    fl1_npy    = STAGE3_DIR / 'f_l1_floor.npy'
    l2_npy     = STAGE4_DIR / 'floor_l2.npy'
    labels_json = STAGE3_DIR / 'l1_labels_text.json'
    for p, lbl in [(fl1_npy, "stage3 f_l1"), (L1_NPZ, "stage2 l1"),
                   (l2_npy, "stage4 l2"), (labels_json, "stage3 labels")]:
        if not check_input(p, lbl):
            return False

    return run([
        PY, str(BOT / 'stage5_bottomup.py'),
        '--fl1_npy',     str(fl1_npy),
        '--l1_npz',      str(L1_NPZ),
        '--l2_npy',      str(l2_npy),
        '--labels_json', str(labels_json),
        '--out_dir',     str(STAGE5_DIR),
        '--ckpt',        str(STAGE5_DIR / 'stage5_model.pt'),
    ], cwd=ROOT)


def stage5_topdown():
    header(5, "stage6_topdown  (Top-down 정제 + scene_graph 출력)")
    fl1_npy    = STAGE3_DIR / 'f_l1_floor.npy'
    l2_npy     = STAGE4_DIR / 'floor_l2.npy'
    labels_json = STAGE3_DIR / 'l1_labels_text.json'
    l2_geo     = STAGE4_DIR / 'floor_l2_geometry.json'
    for p, lbl in [(fl1_npy, "stage3 f_l1"), (L1_NPZ, "stage2 l1"),
                   (l2_npy, "stage4 l2"), (labels_json, "stage3 labels")]:
        if not check_input(p, lbl):
            return False

    RESULTS_DIR.mkdir(exist_ok=True)
    return run([
        PY, str(TOP / 'stage6_topdown.py'),
        '--fl1_npy',     str(fl1_npy),
        '--l1_npz',      str(L1_NPZ),
        '--l2_npy',      str(l2_npy),
        '--labels_json', str(labels_json),
        '--floor_npz',   str(SRC_NPZ),
        '--l2_geometry', str(l2_geo),
        '--out_dir',     str(TOP / 'stage6_results'),
        '--ckpt',        str(TOP / 'stage6_results' / 'stage6_model.pt'),
        '--out_name',    OUT_SCENE_GRAPH.name,
    ], cwd=ROOT)


STAGES = {
    0: ("sam_room_grid",    stage0_wall_door),
    1: ("stage2_s1",        stage1_merge),
    2: ("stage3_openshape", stage2_openshape),
    3: ("stage4_sem_merge", stage3_sem_merge),
    4: ("stage5_bottomup",  stage4_bottomup),
    5: ("stage6_topdown",   stage5_topdown),
}


def main():
    global SRC_NPZ, ROOM_GRID, OUT_SCENE_GRAPH, L1_NPZ, _VIZ_L1, _MULTIVIEW, _CLUTTER_SIM_THR, _CLIP_THRESHOLD

    parser = argparse.ArgumentParser(description='SuperHSSG 파이프라인 자동 실행')
    parser.add_argument('--src',        default='floor.npz')
    parser.add_argument('--from_stage', type=int, default=0)
    parser.add_argument('--to_stage',   type=int, default=5)
    parser.add_argument('--only',       type=int, default=None)
    parser.add_argument('--skip',       type=int, nargs='+', default=[])
    parser.add_argument('--no_multiview', dest='multiview', action='store_false')
    parser.set_defaults(multiview=True)
    parser.add_argument('--clutter_sim_threshold', type=float, default=0.5)
    parser.add_argument('--threshold',  type=float, default=0.3)
    args = parser.parse_args()

    _MULTIVIEW        = args.multiview
    _CLUTTER_SIM_THR  = args.clutter_sim_threshold
    _CLIP_THRESHOLD   = args.threshold

    SRC_NPZ = _resolve_src(args.src)
    if not SRC_NPZ.exists():
        print(f"[오류] 입력 파일 없음: {SRC_NPZ}")
        sys.exit(1)

    out_name        = _derive_out_name(SRC_NPZ)
    OUT_SCENE_GRAPH = RESULTS_DIR / out_name
    ROOM_GRID       = ROOT / f'room_grid_{SRC_NPZ.stem}.npz'
    L1_NPZ          = STAGE2_DIR / f'l1_{SRC_NPZ.stem}.npz'

    run_stages = [args.only] if args.only is not None else list(range(args.from_stage, args.to_stage + 1))
    run_stages = [s for s in run_stages if s not in args.skip]

    print("\n" + "=" * 60, flush=True)
    print("  SuperHSSG Pipeline Runner", flush=True)
    print("=" * 60, flush=True)
    print(f"  입력: {SRC_NPZ}", flush=True)
    print(f"  출력: {OUT_SCENE_GRAPH}", flush=True)
    print(f"  실행 스테이지: {run_stages}", flush=True)
    print("=" * 60, flush=True)

    t_total = time.time()
    failed_at = None

    for s in run_stages:
        if s not in STAGES:
            continue
        name, fn = STAGES[s]
        if not fn():
            failed_at = s
            break

    elapsed = time.time() - t_total
    print("\n" + "=" * 60, flush=True)
    if failed_at is not None:
        print(f"  ✗ STAGE {failed_at} 에서 실패  (총 {elapsed:.1f}s)", flush=True)
        sys.exit(1)
    else:
        print(f"  ✓ 전체 완료  (총 {elapsed:.1f}s)", flush=True)
        if OUT_SCENE_GRAPH.exists():
            print(f"  출력: {OUT_SCENE_GRAPH}", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
