"""Blend A weights. w = cloud weight x cover weight; value = floor((1 - w) * clear + w * partner + 0.5).
Snow variants pick the count: none -> CUR_Z, ndsi -> CUR_ZS (used), bit -> CUR_ZB.
Cloud: heavy5 (used) -> count / 8 where count >= 5 else 0; ramp -> count / 8.
Cover: ramp20-50 (used) on the clear-side tree; ramp20-50max on max(clear, mask map); hard40 on clear > 40; none."""
import numpy as np
SNOW = {'none': 'CUR_Z', 'ndsi': 'CUR_ZS', 'bit': 'CUR_ZB'}
CLOUD = ('heavy5', 'ramp')
COVER = ('ramp20-50', 'ramp20-50max', 'hard40', 'none')
PARTNERS = ('V4R', 'V1-V4C')
MAIN = dict(snow='ndsi', cloud='heavy5', cover='ramp20-50')


def cloud_w(c, mode):
    c = c.astype(np.float64)
    return np.where(c >= 5, c / 8, 0.0) if mode == 'heavy5' else c / 8


def cover_w(base, v1v4c, mode):
    if mode == 'ramp20-50':
        return np.clip((base - 20) / 30, 0, 1)
    if mode == 'ramp20-50max':
        return np.clip((np.maximum(base, v1v4c) - 20) / 30, 0, 1)
    if mode == 'hard40':
        return (base > 40).astype(np.float64)
    return np.ones_like(base, dtype=np.float64)


def blend(base, partner, v1v4c, counts, snow, cloud, cover):
    w = cloud_w(counts[SNOW[snow]], cloud) * cover_w(base, v1v4c, cover)
    return np.floor((1 - w) * base + w * partner + 0.5), w


def variants():
    for p in PARTNERS:
        for s in SNOW:
            for c in CLOUD:
                for v in COVER:
                    yield p, s, c, v


def name(p, s, c, v):
    return '%s|snow=%s|cloud=%s|cover=%s' % (p, s, c, v)
