
from david.sling import David
import sys
from nice_imports import efficiency_stuff
import torch
from transformers import AutoTokenizer, TextStreamer, GenerationConfig
from transformers import AutoModelForCausalLM

#model_id = "cognitivecomputations/dolphin-2.2.1-mistral-7b"
model_id = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    **efficiency_stuff)
model.eval()

David.sling(model, ratio_repeat=[(0, .3), (.2, .4), (.3, .5), 
                            (.4, .7), (.6, .75), (.7, .8), 
                            (.75, .85), (.8, 1)])

streamer = TextStreamer(tokenizer)

initial_input = """Jimmy has a balloon, the balloon string is being held by his left hand. Jimmy also has scissors in his right hand and uses the scissors to cut the balloon string slightly above his left hand, what happens to the balloon? I want you to answer this question in 2 steps, first answer what happens to the balloon, and then incorporate that into a creative story about the situation."""

tokenized_start = tokenizer.apply_chat_template([
    {'role': 'system',
    'content': 'Fulfill the instruction below.'},
    {'role': 'user', 
     'content': initial_input}
], return_tensors='pt')

with torch.no_grad():
    while True:
        generated_tokens = model.generate(
            input_ids=tokenized_start, #.to('cuda:0'), you may want to comment this back in if you don't have the accelerate library installed
            generation_config=GenerationConfig(
                use_cache=True,
                min_new_tokens=2,
                max_new_tokens=800,
                temperature=1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=False,
                output_scores = True
            ),
            streamer=streamer,    
            
        )#, use_cache=True)
        print("\n\nAsk Something:", end="")
        
        await_input = str(input(": "))
        tokenized_start = tokenizer.apply_chat_template([{
            'role': 'user',
            'content': await_input}], return_tensors="pt")
    
    