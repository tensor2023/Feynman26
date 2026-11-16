Harness² mainly needs:
                                                                                                                   
  - An initial harness: prompts, tools, skills, memory, guardrails, etc.                                           
  - A concrete task: the work the agent must complete.                                                             
  - A model/executor: Codex or OpenCode connected to an AI model.                                                  
  - An improvement objective: generally implicit—improve task execution without using ground-truth feedback.       
  - Optional benchmark/evaluator: used afterward to measure performance.                                           
                                                                                                                   
  So the core input is: initial harness + task + model. Harness² generates candidate harness changes, tests them on
  the task, and composes an improved harness.                                                                      
                                                 

Yes. The project includes written setup and benchmark tutorials:                                                 
                                                                                                                   
  - ref/harness2-main/README.md — quick start                                                                      
  - ref/harness2-main/docs/SETUP.md — complete installation                                                        
  - ref/harness2-main/docs/BENCHMARKS.md — running JobBench, LAB, and WorkBuddy                                    
  - ref/harness2-main/docs/EXPERIMENTS.md — experiment configuration and results 