import copy
from math import ceil, floor
import torch
import torch.nn as nn
import warnings
class David:
    orig_model = None
    rev_layers = {}
    first_call = True
    @classmethod
    def sling(cls, model, ratio_repeat=None, repeat_map_raw=None):
        r"""
        ratio_repeat -- is a list of tuples, each tuple is expected to have two elements, one is the starting depth 
                        of a sequence of layers, and the other is an ending depth. 
                        The depths are specified in the range 0.0 - 1.0, where 
                        0.0 corresponds to the first layer of the model, and
                        1.0 corresponds to the last layer of the model. 
                        starting depths round down, stopping depths round up.
                        For example, if you have a 7 layer transformer, then
                        ratio_repeat = [(0, 0.5), (0.3, 0.8), (0.5, 1.0)] 
                        would evaluate to a model that sends its values through the layers in the order
                        [0, 1, 2, 3,   2, 3, 4, 5,   3, 4, 5, 6]
                        If your first start value is greater than 0 and/or your end value is less than 1, the
                        layers will automatically fill in to lead up from the first layer and end at the last layer.               
        optionally, you can manually specify the layers to repeat as an ordered list by providing repeat_map_raw
        """
        return_model = model
        if hasattr(return_model, 'model'):
            model = return_model.model
        
        if cls.orig_model is not None and model is not cls.orig_model:
            raise RuntimeError("""This library only handles one model at a time and furthermore is very unimpressed 
                               with you trying to flaunt all your spare VRAM. Rude.""")
        if cls.first_call is False: 
            warnings.warn("Calling sling multiple times on the same model instance can cause memory leaks")
        
        cls.orig_model = model
        orig_layers = cls._recover_or_set_orig_layers(model)
        cls.rev_layers = {}
            
        if repeat_map_raw is None:
            repeat_map_raw = David.generate_from_ratio_repeat(len(orig_layers), ratio_repeat)
        abstract_layers = []
        for i, v in enumerate(repeat_map_raw):
            orig_layer = orig_layers[v]
            if f"{orig_layer.self_attn.layer_idx}" not in cls.rev_layers:
                cls.rev_layers[f"{orig_layer.self_attn.layer_idx}"] = i
            copied_layer = clone_module(orig_layer)
            copied_layer.self_attn.layer_idx = i
            abstract_layers.append(copied_layer)
        cls._cleanup(model)
        model.layers = nn.ModuleList(abstract_layers)
        cls.first_call = False
        return return_model
    
    @classmethod
    def generate_from_ratio_repeat(cls, total_layers, ratio_repeat):
        layers=[]
        for istart, istop in ratio_repeat:
            start = min(max(min(istart, istop),0),1)
            stop = min(max(max(istart, istop),0),1)
            layers += list(range(floor(start*total_layers), ceil(stop*total_layers)))
        
        last_elem = layers[len(layers)-1]
        layers += list(range(last_elem, total_layers-1))
        first_elem = layers[0]
        layers = list(range(0, first_elem)) + layers
        return layers
    
    @classmethod
    def _recover_or_set_orig_layers(cls, model):
        if cls.first_call:
            return model.layers
        else:
            result = nn.ModuleList([])
            sorted_idx = sorted(cls.rev_layers.keys(), key=int)
            for orig_idx in sorted_idx:
                curr_layer = model.layers[int(cls.rev_layers[orig_idx])]
                curr_layer.self_attn.layer_idx = int(orig_idx)
                result.append(curr_layer)
            return result
    
    @classmethod
    def _cleanup(cls, model):
        #The fact that I'm doing this isn't nearly as bad as the fact that it isn't even helping
        for l in model.layers:
            del l.self_attn
            del l
            torch.cuda.empty_cache()
        del model.layers
        torch.cuda.empty_cache()
        
#Yeah it's hacky as hell, but in my defense, I have no defense, and it is wrong to attack a defenseless man.
def clone_module(module):
    cloned_module = copy.deepcopy(module)
    clean_children(module, cloned_module, 'self_attn')
    clean_children(module.self_attn, cloned_module.self_attn)
    return cloned_module

def clean_children(orig_module, cloned_module, exclude = None):
    for subname, submodule in orig_module.named_children():
        if exclude is None or subname != exclude:
            attr = cloned_module.__getattr__(subname)
            del attr
            torch.cuda.empty_cache()
            setattr(cloned_module, subname, submodule)
