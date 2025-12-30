import json
import numpy as np

# Dummy constant to maintain compatibility with orjson.OPT_SERIALIZE_NUMPY
OPT_SERIALIZE_NUMPY = 1

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dumps(obj, option=None):
    """
    Drop-in replacement for orjson.dumps.
    Returns bytes.
    Ignores 'option' but uses NumpyEncoder by default to match OPT_SERIALIZE_NUMPY behavior.
    """
    return json.dumps(obj, cls=NumpyEncoder).encode('utf-8')

def loads(obj):
    """
    Drop-in replacement for orjson.loads.
    """
    return json.loads(obj)
