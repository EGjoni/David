import warnings
efficiency_stuff = {}
try:
    import bitsandbytes
    efficiency_stuff['load_in_8bit'] = True
except:
    warnings.warn("bitsandbytes isn't installed, you're missing out on memory savings")

try:
    import accelerate
    efficiency_stuff['device_map'] = "auto"
except:
    warnings.warn("accelerate isn't installed, you're missing out on memory management gains")