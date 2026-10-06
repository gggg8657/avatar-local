"""flash_attn 대역 — 말하기(Wan2.2-S2V) 공식 코드가 flash_attn 을 꼭 import 하는데, 이 서버엔 nvcc 가 없어 빌드할 수 없다.
같은 계산을 torch SDPA(H100 에선 내부적으로 flash 커널)로 한다. app.py 가 S2V 서브프로세스에만 PYTHONPATH 로 넣는다(vendor/Wan2.2 는 수정 안 함)."""
import torch
import torch.nn.functional as F

__version__ = "2.0.0-sdpa-shim"


def _sdpa(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
    """q,k,v [L, H, D] (한 시퀀스) → [L, H, D]. 머리 수가 다르면(GQA) k,v 를 늘린다"""
    if k.shape[1] != q.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1); v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                       dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return o[0].transpose(0, 1)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0,
                           softmax_scale=None, causal=False, window_size=(-1, -1), deterministic=False, **_):
    """[total, H, D] 를 cu_seqlens 경계마다 따로 attention (패딩을 섞지 않음)"""
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    return torch.cat([_sdpa(q[cq[i]:cq[i + 1]], k[ck[i]:ck[i + 1]], v[ck[i]:ck[i + 1]], dropout_p, softmax_scale, causal)
                      for i in range(len(cq) - 1)])


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    """[B, L, H, D] (또는 [L, H, D])"""
    if q.dim() == 3:
        return _sdpa(q, k, v, dropout_p, softmax_scale, causal)
    return torch.stack([_sdpa(q[i], k[i], v[i], dropout_p, softmax_scale, causal) for i in range(q.shape[0])])


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    """[B, L, 3, H, D]"""
    return flash_attn_func(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], dropout_p, softmax_scale, causal)
