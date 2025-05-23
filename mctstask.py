from mcts import MCTS
from searchtask import SearchTask
from model import get_proposal, llm_proposal
from utils import extract_summary_from_solution, llm_verify, exact_match_score
from transformers import CLIPModel, AutoProcessor,Qwen2_5_VLForConditionalGeneration, AutoTokenizer
import torch
from PIL import Image
CLIP_MODEL_PATH = "openai/clip-vit-large-patch14-336"
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from prompt import llm_prompt
import re

# Task = MCTS_Task(question, model, processor, args.propose_method, args.value_method, args.branch, args.end_gate,
#                     args.roll_policy, args.roll_branch, args.roll_forward_steps, args.time_limit,
#                     args.iteration_limit, args.exploration_constant, args.alpha, args.inf,
#                     args.temperature, use_case_prompt=args.use_case_prompt, use_reflection=args.use_reflection,
#                     low=args.low, high=args.high, evaluate=args.evaluate,img_path=img)



class MCTS_Task(SearchTask):
    def __init__(self,data, model, processor, prm, propose_method='qwen', value_method='glm', particle_n=3, end_gate=0.9, roll_policy='greedy',
                 roll_branch=1, roll_forward_steps=3, time_limit=None, iteration_limit=3, exploration_constant=0.7,
                 alpha=0.5, inf=1.0, temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=256, use_case_prompt=False, use_reflection='simple', low=0, high=1,
                 evaluate='', sample_value='simple', answer=None, verify_method='string', lang='en', weighted_verify=False,branch=3):
        super().__init__(data, propose_method, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        self.model = model
        self.processor = processor
        self.prm = prm
        self.particle_n = particle_n #particle_num  simulation
        self.branch = branch #

        self.mode = 'mcts'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.end_gate = end_gate
        self.use_reflection = use_reflection
        self.roll_policy = roll_policy
        self.roll_branch = 2
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.roll_forward_steps = roll_forward_steps
        self.alpha = alpha
        self.limit_type = None
        self.INF = inf
        self.node_count = 1
        self.sample_value = sample_value
        self.answer = answer
        self.verify_method = verify_method
        self.reward_model_type = 'vm'
        self.lang = lang
        self.weighted_verify = weighted_verify
       
    def update_count(self):
        self.node_count += 1
        
    def set_limit_type(self):
      if self.time_limit is not None:
          if self.iteration_limit is not None:
              raise ValueError("Cannot have both a time limit and an iteration limit")
          # time taken for each MCTS search in milliseconds
          self.limit_type = 'time'
      else:
          if self.iteration_limit is None:
              raise ValueError("Must have either a time limit or an iteration limit")
          # number of iterations of the search
          if self.iteration_limit < 1:
              raise ValueError("Iteration limit must be greater than one")
          self.limit_type = 'iterations'

    def run(self):
        self.clear_cache()
        self.set_limit_type()
    
        node, finish, root = MCTS(self)
        
        if self.reward_model_type == 'vm':#value_model
            if self.sample_value != 'full':
                if self.evaluate == 'mathvista':  # SciBench style
                    solution = node.y
                
                    if self.lang == 'en':
                        #summary 제공
                        prompt = self.MATH_summary_prompt_wrap(self.question, solution)
                        response = get_proposal(self.model,self.processor,prompt)

                        if not response:
                            print('Failed to get the review!\n')
                            return ''

                        print(f'math_summary_prompt에 의한 결과:{response}\n')
                        summary =response.split("The final answer is")[-1].strip() #get_summary 끝부분
                    final_answer = {'content': self.question, 'solution': solution, 'summary': summary,
                                    'finish': finish}
                    if self.sample_value == 'simple':
                        node.trace_route()
                        new_value_samples = node.get_new_value_samples()
                        final_answer.update({'value_samples': new_value_samples})
                else:  # MATH style self.evaluate == 'scibench"의 else문
                    solution = node.y
                    cnt = 5
                    summ = ''
                    while cnt:
                        if self.verify_method == 'string':
                            summ = self.get_MATH_summary(solution)
                        else:
                            summ = self.get_summary(solution)
                        if summ:
                            node.summary = summ
                            break
                        else:
                            cnt -= 1

                    if not summ:
                        summ = extract_summary_from_solution(solution)
                        node.summary = summ

                    final_answer = {'content': self.question, 'solution': solution, 'summary': summ, 'finish': finish,
                                    'real_answer': self.answer}
                return final_answer, root
        
    
    def get_next_step(self, y, step_n):

        if self.propose_method == 'gpt':
            prompt = self.zero_single_propose_wrap_gpt(self.question, y, step_n, self.lang)
        else:
            prompt = self.zero_single_propose_wrap(self.question, y, step_n, self.lang)

        response =  get_proposal(self.model, self.processor, prompt)
        if not response:
            print('Failed to get the next step!\n')
            return ''
        print(f'response:{response}')
        # if len(response) > 5:
        #     response = response[:5]   
        if response.startswith("Next step: "):  
            stp = response[len("Next step: "):]  # "Next step: " 길이만큼 잘라냄
            if stp in y:
                print('중복！\n')
                return ''  
            revised_ = 'Step ' + str(step_n-1) + ': ' + stp
        else:
            stp = response  # "Next step: "이 없으면 그대로 유지
            if stp in y:
                print('중복！\n')
                return ''  
            revised_ = stp
        print(f'revised 이후의 step: {revised_}\n')
        return revised_ + '\n'



    def get_simple_reflection(self, y, step_n):
        if step_n == 1:
            return '<continue>'
        if self.propose_method in ['local', 'mistral', 'llama'] and self.lang == 'en':
            if 'answer' in y or '\\boxed' in y:
                return '<end>'

        if self.propose_method == 'mistral':
            reflection_prompt = self.single_reflection_wrap_simple_mistral(self.question, y, step_n)
        else:
            reflection_prompt = self.single_reflection_wrap_simple(self.question, y, step_n, self.lang)
        try:
          cnt = 3
          response = []
          while not response and cnt:
            #   response = get_proposal(self.model, self.processor, reflection_prompt,self.img_path)
              response = get_proposal(self.model, self.processor, reflection_prompt)
              cnt -= 1
                  
        except Exception as e:
          print(f'obtain<{self.propose_method}>reflection fail!\nError:{e}\n')
          return ''
        # if not response:
        #     print('获得意见失败！\n')
        #     return '<end>'
        print(f'reflection 결과:{response}')

        
        if 'unsolved' in response or step_n <= 1:
            print('revised된 reflection: <continue>\n')
            return '<continue>'
        elif 'solved' in response:
            print('revised된 reflection: <end>\n')
            return '<end>'
        else:
            print('revised된 reflection: <continue>\n')
            return '<continue>'


    def get_MATH_summary(self, y):
        prompt = self.MATH_summary_prompt_wrap(self.question, y)
        response = llm_proposal(self.model_dict['model'], self.model_dict['tokenizer'], prompt)
        if not response:
            print('Failed to get the review!失败！\n')
            return ''
        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        print(f'Failed to get the review!:{p}\n')
        return p


    def get_step_value(self, y, action):
        print(' get step value 함수 시작\n')
        print(f'y:{y}\n')
        # print(f'action:{action}')
        if y in self.value_cache.keys():
            print('캐시 value 사용됨\n')
            return self.value_cache[y]
        prompt_answer = 'Problem: ' + self.question + '\nSolution:\n' + y

        #response = get_value(self.model,self.processor, prompt_answer, lmm_prompt, action, self.value_method, img_path=self.img_path)
        value = get_value(self.model_dict['model'], self.model_dict['tokenizer'], self.question, y, action, self.value_method)
        
        # value = self.value_outputs_unwrap(response, self.low, 10.0)
            
        print(f'unwrap된 value:{value}\n') #평가받기
        print(' get step value 함수 끝\n')


        self.value_cache.update({y: value})
        # return llm_value* 0.5 + value * 0.5
        return value            



def get_value(model, processor, prm, question, y, action, value_method):
    response = []
    cnt = 2
    trajectory = y + action
    while not response and cnt:
        # value = LLM(llm_prompt, BASE_MODEL_GLM, temperature=temperature, max_tokens=max_tokens, seed=seed)
        #response = get_proposal(model, processor, lmm_prompt, img_path)
        value = prm.score([question], [[trajectory]])[-1][-1][-1]
        
        cnt -= 1
        response.append(value)
    print(f'value로 생성된 값:{response}')
    return response[0]


    particle_intermediate_storage = []
    particles_tracker = []
    current_timestep = 1
    
    
    
    stepwise_particle_tracker_before = []
    stepwise_particle_tracker_after = []

    # Initialize particles
    if reference_particle is None:
        particles = [Particle(temperature=llm_sampling_temp) for _ in range(args.particles)]
    else:
        particles = [Particle(temperature=llm_sampling_temp) for _ in range(args.particles - 1)]

    print(f"Initialized {args.particles} particles.")
    
          for idx, particle in enumerate(particles):
              
              if not particle.is_active():
                  continue                   
                           
         
              response_to_pass_for_score = "\n\n".join(particle.trajectory) + "\n\n" + action
              print(f'particle{idx}_suba:{suba}\n')
              print('--------------')
              reward = prm.score([question], [[response_to_pass_for_score]])[-1][-1][-1]
              
              particle.add_step(subanswers[0], reward, stop)
    
          torch.cuda.empty_cache()
 
          step = step + 1
          stepwise_particle_tracker_before.append([p.deepcopy() for p in particles])
      if reference_particle is None:
          current_particles = particles
          tracker_before = stepwise_particle_tracker_before
          tracker_after =  stepwise_particle_tracker_after
        #    particles, stepwise_particle_tracker_before, stepwise_particle_tracker_after 
      else:
          current_particles = particles + [reference_particle] 
      #def gibbs kernel end
      particles_tracker.append(current_particles)
      particle_intermediate_storage.append([copy.deepcopy(current_particles)])
  

      os.makedirs(config.output_dir, exist_ok=True)
  
      save_path = os.path.join(config.output_dir, f"question{row}.pkl")
      with open(save_path, "wb") as f:
          pickle.dump(particles_tracker, f)
  
      intermediate_dir = os.path.join(config.output_dir, f"question{row}_intermediate.pkl")
      
      with open(intermediate_dir, "wb") as f:
          pickle.dump(particle_intermediate_storage, f)
          
  
      logger.info(f"Saved particles to: {save_path}")
  
      rewards = [x.rewards[-1] for x in current_particles]
      best_particle = current_particles[np.argmax(rewards)]    