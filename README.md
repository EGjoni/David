# David

David is a lightweight library for sanely creating / testing / using Goliath-style transformer franken-merges on-the-fly and with a tiny memory footprint.

## Because

You shouldn't be making them so heavy to begin with. It's a franken-merge. Just reuse the information you already have in memory instead of instantiating and storing literally the same exact 500MB of data ever time you want to repeat a layer. 
This code accomplishes that goal in the hackiest possible way to maximize compatibility, and it's still way better than what y'all are doing. 
Please stop uploading massive franken-merges to huggingface. 
Please.

## How

Specify the layers you want to repeat and how you want to repeat them, and now your 120B franken-merge of a 70B model runs just the same but using less than 60% the memory of the naive 120B approach. 
Your kv-cache will, of course still grow in proportion to the number of times you share a layer, and a forward pass through the model will still take just as long to run. 
(Prototyping new franken-merge recipes will be way more convnenient, though)


## Installation

```bash
git clone https://github.com/EGjoni/David
cd David
pip install -e .
```

## Quick Start

```python
from david.sling import David
from transformers import AutoModelForCausalLM

# Load your model once
model = AutoModelForCausalLM.from_pretrained("your-model-name")

# Tell it which layers to reuse and in what order
# if we pretend the original model had 10 layers, then the following specifies
David.sling(model, ratio_repeat=[
    (0, .3),   # [0, 1, 2, 3]
    (.2, .4),  #       [2, 3, 4]
    (.4, .5),  #             [4, 5]
    (.4, 1),   #             [4, 5, 6, 7, 8, 9]
])   
# Final sequence: [0, 1, 2, 3, 2, 3, 4, 4, 5, 4, 5, 6, 7, 8, 9]
```

## Arguments

It's pretty straightforward:

1. You specify which layers you want to reuse using `ratio_repeat`:
   - Each tuple is (start_depth, end_depth)
   - 0.0 means first layer, 1.0 means last layer
   - Start depths round down, end depths round up
   - Any gaps in your `ratio_repeat` sequence get filled automatically:
   - If you explicitly want to skip layers, use `repeat_map_raw`

2. David creates a virtual sequence that reuses your existing layers:
   - No copying of weights
   - No storing a new model
   - Just references to your existing layers in the order you specified



## Examples

For a 7-layer transformer, if you specify:
```python
ratio_repeat = [(0, 0.5), (0.3, 0.8), (0.5, 1.0)]
```
You'll get this sequence: `[0, 1, 2, 3, 2, 3, 4, 5, 3, 4, 5, 6]`

Want to specify exact layers? Just use `repeat_map_raw`:
```python
David.sling(model, repeat_map_raw=[0, 1, 2, 1, 2, 3, 2, 3, 4])
```

## Limitations

- Handles one model at a time (because that's what a franken-merge is)
- `sling()` should only be called on the model once before using it for generation (especially if using kv-cache)
- For inference only, not training
