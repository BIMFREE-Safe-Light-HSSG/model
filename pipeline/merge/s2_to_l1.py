"""
s2_to_l1.py
===========
stage2_s1.py 출력(S2 결과)을 기존 파이프라인의 l1_floor.npz 포맷으로 변환.

기존 파이프라인:
  stage2_colab.py → l1_floor.npz → stage3_openshape.py → stage4_sem_merge.py

새 파이프라인:
  stage2_s1.py → 6_floor_s2.npz → [이 스크립트] → l1_6_floor.npz
                                                          ↓
                                              stage3_openshape.py (그대로 재활용)

변환 규칙:
  l1_labels[sp] = sp_to_s2[sp]   (S2 ID를 L1 ID로 사용)
  구조체 SP (sp_pred ∈ {0,1,2,3}) → l1_labels = -1  (stage3/4 구조체 필터와 동일)

사용법:
  python s2_to_l1.py --s2 ../../src/6_floor_s2.npz
  python s2_to_l1.py --s2 ../../src/5_floor_s2.npz
"""

import argparse
import numpy as np
from pathlib import Path

STRUCTURAL_CLASSES = {0, 1, 2, 3}   # ceiling, floor, wall, beam → l1_labels=-1

if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--s2',  required=True,
                    help='stage2_s1.py 출력 경로 (예: ../../src/6_floor_s2.npz)')
    pa.add_argument('--out', default=None,
                    help='출력 경로 (기본: stage2_results/l1_{stem}.npz)')
    args = pa.parse_args()

    s2_path = Path(args.s2)
    stem    = s2_path.stem.replace('_s2', '')   # "6_floor_s2" → "6_floor"

    # ── 출력 경로 결정 ────────────────────────────────────────
    out_path = Path(args.out) if args.out else \
               Path(__file__).parent / 'stage2_results' / f'l1_{stem}.npz'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── S2 결과 로드 ──────────────────────────────────────────
    print(f"[로드] {s2_path.name}")
    s2 = np.load(s2_path)
    sp_to_s2 = s2['sp_to_s2'].astype(np.int64)   # [N_sp]  SP별 S2 ID
    sp_pred  = s2['sp_pred'].astype(np.int64)     # [N_sp]  SP별 클래스
    N_sp     = len(sp_to_s2)
    N_s2     = int(sp_to_s2.max()) + 1
    print(f"  SP={N_sp:,}  S2={N_s2:,}개")

    # ── l1_labels 생성: 구조체는 -1 ──────────────────────────
    l1_labels = sp_to_s2.copy()
    structural_mask = np.isin(sp_pred, sorted(STRUCTURAL_CLASSES))
    l1_labels[structural_mask] = -1
    n_struct = int(structural_mask.sum())
    n_valid  = int((l1_labels >= 0).sum())
    print(f"  구조체 SP: {n_struct:,}개 → l1_labels=-1")
    print(f"  유효 SP:   {n_valid:,}개  (L1 ID 0~{l1_labels.max()})")

    # ── SP 엣지 로드 (Jaccard 계산 시 사용한 원본 엣지) ────────
    edge_path = s2_path.parent / f'{stem}_edges.npz'
    if edge_path.exists():
        print(f"[엣지] {edge_path.name} 로드")
        ep       = np.load(edge_path)
        edge_src = ep['edge_src'].astype(np.int64)
        edge_dst = ep['edge_dst'].astype(np.int64)
        print(f"  SP 엣지: {len(edge_src):,}개")
    else:
        print(f"[경고] {edge_path.name} 없음 → 빈 엣지로 저장")
        edge_src = np.array([], dtype=np.int64)
        edge_dst = np.array([], dtype=np.int64)

    # ── SP 피처 로드 (원본 floor.npz에서) ────────────────────
    floor_path = s2_path.parent / f'{stem}.npz'
    sp_features = None
    if floor_path.exists():
        print(f"[피처] {floor_path.name} 로드")
        fl          = np.load(floor_path)
        sp_features = fl['sp_features'].astype(np.float32)   # [N_sp, 139]
        print(f"  sp_features: {sp_features.shape}")

    # ── 저장 ─────────────────────────────────────────────────
    save_dict = dict(
        l1_labels   = l1_labels,    # [N_sp]        -1=구조체, 0~=S2 ID
        sp_pred     = sp_pred,      # [N_sp]        SP 클래스 예측
        edge_src    = edge_src,     # [E]           SP 엣지 src
        edge_dst    = edge_dst,     # [E]           SP 엣지 dst
    )
    if sp_features is not None:
        save_dict['sp_features'] = sp_features

    np.savez_compressed(out_path, **save_dict)
    print(f"\n[저장] {out_path}")
    print(f"  → stage3_openshape.py --l1_npz {out_path}")

    # ── 클래스별 L1(=S2) 통계 출력 ───────────────────────────
    CLASS_NAMES = {
        0:'ceiling',1:'floor',2:'wall',3:'beam',4:'column',5:'window',
        6:'door',7:'table',8:'chair',9:'sofa',10:'bookcase',11:'board',12:'clutter'
    }
    s2_cls = np.full(N_s2, -1, dtype=np.int64)
    for sp_id in range(N_sp):
        s2_id = int(sp_to_s2[sp_id])
        if s2_cls[s2_id] < 0:
            s2_cls[s2_id] = int(sp_pred[sp_id])
    print(f"\n  클래스별 L1(S2) 수:")
    for cid, cnt in zip(*np.unique(s2_cls[s2_cls >= 0], return_counts=True)):
        print(f"    {CLASS_NAMES.get(int(cid),'?'):10s}({cid}): {cnt:,}")
