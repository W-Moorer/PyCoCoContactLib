import numpy as np

def triangle_area(a, b, c):
    import ss_compare as _ss
    return _ss.triangle_area(np.asarray(a, float), np.asarray(b, float), np.asarray(c, float))

def triangle_normal(a, b, c):
    import ss_compare as _ss
    return _ss.triangle_normal(np.asarray(a, float), np.asarray(b, float), np.asarray(c, float))

