#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def numel(t):
    return int(t.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--base', default=None)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location='cpu')
    base = torch.load(args.base, map_location='cpu') if args.base else None

    groups = {
        'vlm_text_lora': 0,
        'qformer_lora': 0,
        'lawm_lora': 0,
        'query_delta': 0,
        'flow_added_modules': 0,
        'flow_changed_base_tensors': 0,
        'routing_memory': 0,
    }
    changed_flow = []
    for k, v in state.items():
        if k.startswith('policy_backend.vlm.') and (k.endswith('.lora_A') or k.endswith('.lora_B')):
            groups['vlm_text_lora'] += numel(v)
        elif k.startswith('policy_backend.vlm_to_lam.') and (k.endswith('.lora_A') or k.endswith('.lora_B')):
            groups['qformer_lora'] += numel(v)
        elif k.startswith('policy_backend.lam.decoder.') and (k.endswith('.lora_A') or k.endswith('.lora_B')):
            groups['lawm_lora'] += numel(v)
        elif k in {
            'policy_backend.routing_v2_act_query_delta',
            'policy_backend.routing_v2_flow_query_delta',
        }:
            groups['query_delta'] += numel(v)
        elif k.startswith('policy_backend.routing_v2_memory.'):
            groups['routing_memory'] += numel(v)

        if base is not None and k.startswith('policy_backend.flow.'):
            if k not in base:
                groups['flow_added_modules'] += numel(v)
            elif v.shape == base[k].shape and not torch.equal(v, base[k]):
                groups['flow_changed_base_tensors'] += numel(v)
                changed_flow.append(k)

    incremental = (
        groups['vlm_text_lora'] + groups['qformer_lora'] + groups['lawm_lora']
        + groups['query_delta'] + groups['flow_added_modules'] + groups['flow_changed_base_tensors']
        + groups['routing_memory']
    )
    total_model = sum(numel(v) for v in state.values())
    out = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'base': str(Path(args.base).resolve()) if args.base else None,
        'groups': groups,
        'incremental_task_params': incremental,
        'checkpoint_tensor_numel': total_model,
        'incremental_percent_of_checkpoint': 100.0 * incremental / max(1, total_model),
        'changed_flow_tensor_count': len(changed_flow),
        'changed_flow_tensors': changed_flow,
    }
    print('[RoutingV2 checkpoint parameter audit]')
    for k, v in groups.items():
        print(f'  {k:28s}: {v:,}')
    print(f'  {"incremental_task_params":28s}: {incremental:,}')
    print(f'  {"checkpoint_tensor_numel":28s}: {total_model:,}')
    print(f'  {"incremental_percent":28s}: {out["incremental_percent_of_checkpoint"]:.4f}%')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding='utf-8')
        print(f'[OK] wrote {args.output}')


if __name__ == '__main__':
    main()
