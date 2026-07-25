"""模型 patch：把原生 MoE 块替换成流式块（文件后端 / 常驻切片版），并按需挂跨层预取。"""
from mlx_streaming import config
from mlx_streaming.core.moe.block import FileStreamingMoeBlock, StreamingMoeBlock
from mlx_streaming.models.base import ModelDimensions
from mlx_streaming.core.prefetch.cross_layer import enable_cross_layer_prefetch


def patch_model_filebacked(model, store, hidden=None, moe_inter=None,
                           group_size=None, bits=None,
                           proj_bits: dict | None = None,
                           layer_proj_bits: dict | None = None, *,
                           adapter=None, manifest=None):
    """把每个 MoE 块替换为 FileStreamingMoeBlock，并丢弃常驻的堆叠 switch_mlp。

    store：FileExpertStore（所有 MoE 层共用，按 (layer,expert) 缓存）。
    proj_bits：非空时走混合精度（逐 proj 不同 bit），专家须为对应混合重量化产出。
    layer_proj_bits：{绝对层号: {proj:bits}}，非空时逐层用各自 proj_bits（优先于 proj_bits），
        对应 requantize_dir_layered 产出。各层 QSL 用该层 bit，与流式存盘文件一一对应。
    返回被替换的层数。被替换后原 switch_mlp 不再被引用，惰性权重不会被物化。
    """
    if adapter is not None:
        if manifest is None:
            raise ValueError("adapter-driven patching requires an expert manifest")
        specs = adapter.expert_layers(model)
        if not specs:
            raise ValueError("adapter reported no routed expert layers")
        first = specs[0]
        dimensions = ModelDimensions(
            architecture=adapter.architecture,
            hidden_size=first.hidden_size,
            num_layers=len(specs),
            num_experts=first.num_experts,
            top_k=first.top_k,
            expert_intermediate_size=first.intermediate_size,
            shared_expert_intermediate_size=None,
            quant_mode=manifest.quant_mode,
            quant_bits=manifest.projection_bits["gate_proj"],
            quant_group_size=manifest.group_size,
            max_context=0,
        )
        manifest.validate_against(dimensions, require_complete=True)
        manifest.verify_files(store.root)
        layers = tuple(adapter.layers(model))
        if len(layers) != len(specs):
            raise ValueError(
                f"adapter layer/spec count mismatch: {len(layers)} != {len(specs)}"
            )
        prefetch_model = getattr(adapter.language_model(model), "model", model)
        patched = 0
        for layer, spec in zip(layers, specs):
            if spec.layer_index != patched:
                raise ValueError(
                    f"expert layer index mismatch: expected {patched}, "
                    f"got {spec.layer_index}"
                )
            layer._layer_idx = spec.layer_index
            object.__setattr__(layer, "_prefetch_model_ref", prefetch_model)
            block = spec.block
            layer.mlp = FileStreamingMoeBlock(
                gate=block.gate,
                top_k=spec.top_k,
                norm_topk_prob=spec.normalise_topk,
                store=store,
                layer_idx=spec.layer_index,
                hidden=spec.hidden_size,
                moe_inter=spec.intermediate_size,
                group_size=manifest.group_size,
                bits=manifest.projection_bits["gate_proj"],
                proj_bits=manifest.projection_bits,
                shared_expert=getattr(block, "shared_expert", None),
                shared_expert_gate=getattr(block, "shared_expert_gate", None),
            )
            object.__setattr__(
                layer.mlp, "_prefetch_model_ref", prefetch_model
            )
            patched += 1
        if patched != len(specs):
            raise ValueError(
                f"patched {patched} expert layers, expected {len(specs)}"
            )
        if (
            config.cross_layer_prefetch()
            or getattr(store, "_staging", None) is not None
        ):
            enable_cross_layer_prefetch()
        return patched

    if None in (hidden, moe_inter, group_size, bits):
        raise TypeError(
            "legacy patching requires hidden, moe_inter, group_size and bits"
        )
    patched = 0
    for i, layer in enumerate(model.layers):
        layer._layer_idx = i
        # 用 object.__setattr__ 存反向 model 引用,避免进 nn.Module(=dict)的子模块树:
        # 否则 model→layer→model 成环,wired_limit 等 tree_reduce 遍历会无限递归(RecursionError)。
        object.__setattr__(layer, "_prefetch_model_ref", model)
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            pb = layer_proj_bits.get(i, proj_bits) if layer_proj_bits else proj_bits
            # 捕获共享专家引用使其常驻（Qwen3-Next 有，Qwen3-MoE 无）
            layer.mlp = FileStreamingMoeBlock(
                gate=mlp.gate, top_k=mlp.top_k, norm_topk_prob=mlp.norm_topk_prob,
                store=store, layer_idx=i, hidden=hidden, moe_inter=moe_inter,
                group_size=group_size, bits=bits, proj_bits=pb,
                shared_expert=getattr(mlp, "shared_expert", None),
                shared_expert_gate=getattr(mlp, "shared_expert_gate", None),
            )
            # 同理:不进 dict,避免 mlp→model→...→mlp 成环。供 native-fused-prefetch 取下层 gate。
            object.__setattr__(layer.mlp, "_prefetch_model_ref", model)
            patched += 1
    if config.cross_layer_prefetch() or getattr(store, "_staging", None) is not None:
        enable_cross_layer_prefetch()
    return patched


def patch_model(model, store_factory=None):
    """把模型里每个 MoE 块（含 switch_mlp 与 gate 的块）替换成 StreamingMoeBlock。

    store_factory(layer_idx)->LruExpertStore，可为 None（仅做 uniq 切片、不接磁盘后端）。
    返回被替换的层数，便于校验确实命中了 MoE 层。
    """
    patched = 0
    for i, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            store = store_factory(i) if store_factory is not None else None
            layer.mlp = StreamingMoeBlock(mlp, layer_idx=i, store=store)
            patched += 1
    return patched
