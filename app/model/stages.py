"""
stages.py — 실제 파이프라인 단계 구현
======================================
InputStage   : MinIO에서 scan.zip 다운로드 → NPZ 추출
PipelineStage: run_pipeline.py subprocess 실행 (stdout 파싱 → 진행률 콜백)
OutputStage  : scene_graph.json 읽기 → GraphData 변환
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.config import get_settings
from app.schemas.callback import GraphData
from app.schemas.transform import TransformRequest

# pipeline/ 디렉토리: model/pipeline/
PIPELINE_DIR = Path(__file__).resolve().parents[2] / "pipeline"

# 스테이지 번호 → 진행률 매핑 (stdout의 "STAGE X:" 파싱)
_STAGE_PROGRESS: dict[int, int] = {
    0: 20,
    1: 40,
    2: 60,
    3: 75,
    4: 85,
    5: 95,
}
_STAGE_RE = re.compile(r"STAGE\s+(\d+):", re.IGNORECASE)

ProgressCallback = Callable[[int], Awaitable[None]]


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelInput:
    task_id: str
    building_id: str | None
    npz_path: Path          # pipeline/src/{task_id}.npz


# ── InputStage ────────────────────────────────────────────────────────────────
class InputStage:
    """MinIO에서 scan.zip을 다운로드하고, 내부의 .npz를 추출한다."""

    async def run(self, payload: TransformRequest) -> ModelInput:
        settings = get_settings()
        npz_path = PIPELINE_DIR / "src" / f"{payload.task_id}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        zip_bytes = await self._download_from_minio(
            settings=settings,
            bucket=payload.bucket_name,
            key=payload.object_key,
        )
        self._extract_npz(zip_bytes, npz_path)
        return ModelInput(
            task_id=payload.task_id,
            building_id=payload.building_id,
            npz_path=npz_path,
        )

    async def _download_from_minio(self, settings, bucket: str, key: str) -> bytes:
        """aiobotocore / boto3 fallback으로 MinIO에서 객체 다운로드."""
        try:
            import aiobotocore.session  # type: ignore

            session = aiobotocore.session.get_session()
            async with session.create_client(
                "s3",
                endpoint_url=settings.minio_endpoint,
                aws_access_key_id=settings.minio_access_key,
                aws_secret_access_key=settings.minio_secret_key,
                use_ssl=settings.minio_use_ssl,
            ) as client:
                resp = await client.get_object(Bucket=bucket, Key=key)
                return await resp["Body"].read()

        except ImportError:
            # aiobotocore 없으면 boto3 동기 fallback (asyncio executor)
            import boto3  # type: ignore

            def _sync_download() -> bytes:
                s3 = boto3.client(
                    "s3",
                    endpoint_url=settings.minio_endpoint,
                    aws_access_key_id=settings.minio_access_key,
                    aws_secret_access_key=settings.minio_secret_key,
                )
                buf = io.BytesIO()
                s3.download_fileobj(bucket, key, buf)
                return buf.getvalue()

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _sync_download)

    def _extract_npz(self, zip_bytes: bytes, out_path: Path) -> None:
        """scan.zip 내부의 첫 번째 .npz 파일을 추출한다."""
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            npz_names = [n for n in zf.namelist() if n.endswith(".npz")]
            if not npz_names:
                raise ValueError("scan.zip 내에 .npz 파일이 없습니다.")
            with zf.open(npz_names[0]) as src, open(out_path, "wb") as dst:
                dst.write(src.read())


# ── PipelineStage ─────────────────────────────────────────────────────────────
class PipelineStage:
    """
    run_pipeline.py를 비동기 subprocess로 실행한다.
    stdout에서 'STAGE X:' 패턴을 감지해 진행률 콜백을 호출한다.
    """

    async def run(
        self,
        model_input: ModelInput,
        progress_cb: ProgressCallback | None = None,
    ) -> None:
        settings = get_settings()
        python   = settings.pipeline_python
        script   = str(PIPELINE_DIR / "run_pipeline.py")
        src_arg  = str(model_input.npz_path)

        cmd = [python, script, "--src", src_arg]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PIPELINE_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # stderr → stdout 합침
        )

        async for raw_line in proc.stdout:  # type: ignore[union-attr]
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            print(line, flush=True)          # 서버 로그에 그대로 출력

            if progress_cb:
                m = _STAGE_RE.search(line)
                if m:
                    stage_num = int(m.group(1))
                    pct = _STAGE_PROGRESS.get(stage_num)
                    if pct:
                        await progress_cb(pct)

        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"파이프라인 실패 (exit {proc.returncode}). "
                "서버 로그를 확인하세요."
            )


# ── OutputStage ───────────────────────────────────────────────────────────────
class OutputStage:
    """
    pipeline/results/{stem}_scene_graph.json을 읽어 GraphData로 변환한다.
    scene_graph.json 구조:
      { "nodes": [...], "edges": [...], ... }
    nodes/edges가 없으면 전체를 assets에 넣는다.
    """

    async def run(self, model_input: ModelInput) -> GraphData:
        stem       = model_input.npz_path.stem   # task_id
        out_name   = self._derive_out_name(stem)
        scene_path = PIPELINE_DIR / "results" / out_name

        if not scene_path.exists():
            raise FileNotFoundError(
                f"scene_graph.json 없음: {scene_path}\n"
                "파이프라인이 완료됐는지 확인하세요."
            )

        raw: dict[str, Any] = json.loads(scene_path.read_text(encoding="utf-8"))

        nodes  = raw.get("nodes")  or raw.get("node_list")  or []
        edges  = raw.get("edges")  or raw.get("edge_list")  or []
        assets = {k: v for k, v in raw.items() if k not in ("nodes", "edges", "node_list", "edge_list")}

        # 노드/엣지가 dict 형태인 경우 list로 래핑
        if isinstance(nodes, dict):
            nodes = list(nodes.values())
        if isinstance(edges, dict):
            edges = list(edges.values())

        return GraphData(
            version=str(raw.get("version", "1.0")),
            nodes=nodes,
            edges=edges,
            assets=assets,
        )

    @staticmethod
    def _derive_out_name(stem: str) -> str:
        if stem == "floor":
            return "scene_graph.json"
        if stem.endswith("_floor"):
            prefix = stem[: -len("_floor")]
            return f"{prefix}_scene_graph.json"
        return f"{stem}_scene_graph.json"
