"""路线 B 离线工具：把模型里堆叠的 switch_mlp 专家权重按专家拆成 per-expert 小文件。

拆分后每个文件 layer{L:02d}_expert{E:03d}.safetensors 含扁平 dict：
  gate_proj.weight / gate_proj.scales / gate_proj.biases
  up_proj.weight   / up_proj.scales   / up_proj.biases
  down_proj.weight / down_proj.scales / down_proj.biases
（非量化模型则只有 .weight，可能还有 .bias）

一次拆分、逐专家物化，全程低内存。
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import mlx.core as mx

from mlx_streaming.models.base import ModelDimensions
from mlx_streaming.prep.expert_manifest import (
    ExpertFileRecord,
    ExpertStoreManifest,
    write_manifest,
)

PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]


def _save_expert(path: Path, tensors: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".safetensors",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        mx.save_safetensors(str(temporary), tensors)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def split_switch_glu(switch_glu, out_dir: str, layer: int) -> int:
    """把一个 SwitchGLU 的三组 SwitchLinear 沿专家维拆成 per-expert 文件。返回专家数。"""
    os.makedirs(out_dir, exist_ok=True)
    E = switch_glu.gate_proj.num_experts
    for e in range(E):
        d = {}
        for proj_name in PROJ_NAMES:
            proj = getattr(switch_glu, proj_name)
            for pname, p in proj.parameters().items():
                if isinstance(p, mx.array) and p.ndim >= 1 and p.shape[0] == E:
                    d[f"{proj_name}.{pname}"] = p[e]
        mx.eval(d)   # 只物化这一个专家
        path = Path(out_dir) / f"layer{layer:02d}_expert{e:03d}.safetensors"
        _save_expert(path, d)
    return E


def _file_record(path: Path, root: Path) -> ExpertFileRecord:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return ExpertFileRecord(
        path=path.relative_to(root).as_posix(),
        size=path.stat().st_size,
        sha256=digest.hexdigest(),
    )


def split_model(
    model_path: str,
    out_dir: str,
    *,
    adapter=None,
    source_repository: str | None = None,
    source_revision: str | None = None,
) -> dict:
    """加载模型（lazy）并把所有 MoE 层的专家拆到 out_dir。返回 {dims/统计}。"""
    if adapter is not None:
        if not source_repository or not source_revision:
            raise ValueError(
                "adapter splitting requires source_repository and source_revision"
            )
        loaded = adapter.load(
            model_path,
            revision=source_revision,
            lazy=True,
        )
        model = loaded.model
        specs = adapter.expert_layers(model)
        if not specs:
            raise ValueError("adapter reported no routed expert layers")
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        files = []
        for spec in specs:
            count = split_switch_glu(
                spec.block.switch_mlp,
                out_dir,
                spec.layer_index,
            )
            if count != spec.num_experts:
                raise ValueError(
                    f"layer {spec.layer_index} split {count} experts, "
                    f"expected {spec.num_experts}"
                )
            for expert in range(count):
                path = root / (
                    f"layer{spec.layer_index:02d}_"
                    f"expert{expert:03d}.safetensors"
                )
                files.append(_file_record(path, root))
        first = specs[0]
        gate_proj = first.block.switch_mlp.gate_proj
        bits = int(gate_proj.bits)
        group_size = int(gate_proj.group_size)
        manifest = ExpertStoreManifest(
            schema_version=1,
            source_repository=source_repository,
            source_revision=source_revision,
            architecture=adapter.architecture,
            layer_indices=tuple(spec.layer_index for spec in specs),
            num_experts=first.num_experts,
            top_k=first.top_k,
            hidden_size=first.hidden_size,
            expert_intermediate_size=first.intermediate_size,
            projection_names=tuple(PROJ_NAMES),
            projection_bits={name: bits for name in PROJ_NAMES},
            group_size=group_size,
            quant_mode="affine",
            file_pattern="layer{layer:02d}_expert{expert:03d}.safetensors",
            files=tuple(files),
        )
        dimensions = ModelDimensions(
            architecture=adapter.architecture,
            hidden_size=first.hidden_size,
            num_layers=len(specs),
            num_experts=first.num_experts,
            top_k=first.top_k,
            expert_intermediate_size=first.intermediate_size,
            shared_expert_intermediate_size=None,
            quant_mode="affine",
            quant_bits=bits,
            quant_group_size=group_size,
            max_context=0,
        )
        manifest.validate_against(dimensions, require_complete=True)
        manifest.verify_files(root)
        write_manifest(root / "expert_manifest.json", manifest)
        meta = {
            "out_dir": out_dir,
            "moe_layers": list(manifest.layer_indices),
            "dims": {
                "hidden": manifest.hidden_size,
                "moe_intermediate": manifest.expert_intermediate_size,
                "num_experts": manifest.num_experts,
                "group_size": manifest.group_size,
                "bits": bits,
            },
            "expert_manifest": "expert_manifest.json",
        }
        with (root / "_split_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return meta

    from mlx_lm import load
    model, _ = load(model_path, lazy=True)
    os.makedirs(out_dir, exist_ok=True)
    moe_layers = []
    dims = None
    for l, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            sm = mlp.switch_mlp
            split_switch_glu(sm, out_dir, l)
            moe_layers.append(l)
            if dims is None:
                gp = sm.gate_proj
                dims = {
                    "hidden": gp.input_dims,
                    "moe_intermediate": gp.output_dims,
                    "num_experts": gp.num_experts,
                    "group_size": getattr(gp, "group_size", None),
                    "bits": getattr(gp, "bits", None),
                }
    meta = {"out_dir": out_dir, "moe_layers": moe_layers, "dims": dims}
    with open(os.path.join(out_dir, "_split_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


if __name__ == "__main__":
    mp = sys.argv[1]
    od = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mlx_qwen3_experts"
    m = split_model(mp, od)
    print(json.dumps(m, ensure_ascii=False, indent=2))
