

llm_prompt = """Evaluate the given solution (not necessarily complete) based on the clarity, coherence, and correctness of the logical reasoning process. Assess whether the steps follow a structured, step-by-step approach, ensuring that intermediate steps are neither skipped nor incorrect. Consider whether assumptions are clearly stated, conclusions logically follow from premises, and the response adheres to formal rules in mathematical proofs or coding logic. 

Provide only decimal score between 0 and 10. The output format is limited to: "Score:..." where ... represents the omitted output content, which you need to fill in.
Here is the input you should score.
Input: 
Problem:"""

zero_single_proposal_prompt_en = '''
Given the problem and an existing partial solution (not a complete answer), generate the correct next step to solve the problem.
Please keep your response concise and limited to one reasoning step.

The output format is strictly:
"Next step: ..."
Please follow the restricted format and be brief.
If solvable, include the final answer.
'''
zero_single_proposal_prompt_llama = '''
Given the problem and an existing partial solution (not a complete answer), generate the correct next step to solve the problem.
Please keep your response concise and limited to one reasoning step.'''


critic_simplified = '''Given the problem, and reasoning steps, evaluate whether the reasoning steps are sufficient to solve the problem.
If there is no correct answer, never assign a score higher than 9.
Output format must be: "Score: ...", where ... is the decimal score.
Do not include any additional explanation or analysis. Follow the output format exactly.'''


single_reflection_prompt_simple_en = '''
Given a problem and reasoning steps (not necessarily complete), you need to determine whether the given steps have completely solved the problem.
If the given steps have provided the final answer to the question, then you should generate "Problem solved" and nothing else.
If the given steps have not yet calculated the answer to the question or have not finished reasoning, then please generate "Problem unsolved" 
Do not generate additional reasoning step or anaylsis, please generate either "Problem solved" or "Problem unsolved". 
'''


# image_description_prompt = '''
# your task is to provide the image description based on a given image and problem, the output format is limited to "Image Description:..." where ... represents the output result, which you should fill in.
# Here is the input, please follow the restricted output format.
# Given problem:
# '''


