from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import AutoModelForCausalLM
#  MllamaForConditionalGeneration


def LLM(model):
     
    if model == 'qwen3':
        print('init llm model')       
        # model_name = "Qwen/Qwen2.5-7B-Instruct"
        model_name = "Qwen/Qwen3-8B"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
        return model, tokenizer
    if model == 'qwen2.5':
        print('init llm model')       
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer 
    elif model == 'llama3.2':
        print('init llama model')
        


        
def llm_proposal(model=None,tokenizer=None,prompt=None,model_name='qwen'):
    if model_name =='qwen':
        messages = [ {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True)
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs, max_new_tokens=512)
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response  
    if model_name == 'gpt':
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )

        # 🎯 출력 결과
        reply = response['choices'][0]['message']['content'].strip()
        return reply 




def get_proposal(model, processor, prompt, model_name ='qwen'):
    #temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,do_sample=True, max_new_tokens=1024
    if model_name =='qwen':
        messages = [ {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True)
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs, max_new_tokens=512, eos_token_id =271)
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response  

    elif model_name == 'vllm_qwen':

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            n=1,
            logprobs=1,
            max_tokens=256,
            stop=[],
        )

        output = model.generate(prompt, sampling_params, use_tqdm=False)      
