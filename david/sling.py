import copy
from math import ceil, floor
import torch
import torch.nn as nn
class David:   
    @classmethod
    def sling(self, model, ratio_repeat=None, repeat_map_raw=None):
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
            
        if repeat_map_raw is None:
            repeat_map_raw = David.generate_from_ratio_repeat(len(model.layers), ratio_repeat)
        abstract_layers = []
        for i, v in enumerate(repeat_map_raw):
            orig_layer = model.layers[v]           
            copied_layer = clone_module(orig_layer)
            copied_layer.self_attn.layer_idx = i
            abstract_layers.append(copied_layer)
        model.layers = nn.ModuleList(abstract_layers)
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
