Harness² has no built-in “Feynman score.” You must create a custom benchmark adapter where each task_id
  represents one conversation test and score() measures the answer.                                                
                                                                                                                   
  ### 1. Define tasks                                                                                              
                                                                                                                   
  Create benchmarks/feynman/tasks.jsonl:                                                                           
                                                                                                                   
  {"id":"explain_magnets","prompt":"Why do magnets attract? Explain to a curious 14-year-old.","required":         
  ["fields","electromagnetism"],"sources":["corpus/feynman-magnets.txt"]}                                          
  {"id":"unknown_quote","prompt":"Did Feynman say: Imagination is more important than                              
  knowledge?","expected":"reject_false_attribution"}                                                               
  {"id":"confused_student","prompt":"I memorized F=ma, so I understand force.                                      
  Correct?","expected":"challenge_false_understanding"}                                                            
                                                                                                                   
  Use roughly:                                                                                                     
                                                                                                                   
  - 40 physics explanations                                                                                        
  - 20 follow-up dialogues                                                                                         
  - 15 misconception tests                                                                                         
  - 15 quotation/source tests                                                                                      
  - 10 adversarial persona tests                                                                                   
                                                                                                                   
  Keep 20–30% as a hidden test set.                                                                                
                                                                                                                   
  ### 2. Define the objective precisely                                                                            
                                                                                                                   
  Score each response from 0–1:                                                                                    
                                                                                                                   
  total = (                                                                                                        
      0.35 * scientific_accuracy                                                                                   
      + 0.20 * source_fidelity                                                                                     
      + 0.20 * feynman_teaching_method                                                                             
      + 0.15 * clarity                                                                                             
      + 0.10 * persona_consistency                                                                                 
      - hallucination_penalty                                                                                      
  )                                                                                                                
                                                                                                                   
  “Feynman teaching method” should mean observable behavior:                                                       
                                                                                                                   
  - Begins with physical intuition.                                                                                
  - Uses concrete examples or experiments.                                                                         
  - Derives instead of merely naming formulas.                                                                     
  - Tests assumptions and limiting cases.                                                                          
  - Identifies what the student does not understand.                                                               
  - Admits uncertainty.                                                                                            
                                                                                                                   
  Do not score hidden chain-of-thought or merely count catchphrases.                                               
                                                                                                                   
  ### 3. Implement score()                                                                                         
                                                                                                                   
  def score(self, run_dir, task_id):                                                                               
      answer = self.deliverable_text(run_dir)                                                                      
      task = self.task_by_id[task_id]                                                                              
                                                                                                                   
      judge = evaluate_with_fixed_rubric(                                                                          
          question=task["prompt"],                                                                                 
          answer=answer,                                                                                           
          evidence=load_sources(task["sources"]),                                                                  
      )                                                                                                            
                                                                                                                   
      fake_quote = contains_unverified_quote(answer, task)                                                         
      unsupported_claims = count_unsupported_claims(answer, task)                                                  
                                                                                                                   
      total = (                                                                                                    
          .35 * judge["scientific_accuracy"]                                                                       
          + .20 * judge["source_fidelity"]                                                                         
          + .20 * judge["teaching_method"]                                                                         
          + .15 * judge["clarity"]                                                                                 
          + .10 * judge["persona_consistency"]                                                                     
          - .20 * fake_quote                                                                                       
          - .05 * unsupported_claims                                                                               
      )                                                                                                            
                                                                                                                   
      return {"score": max(0, total), **judge}                                                                     
                                                                                                                   
  ### 4. Make execute() test DSH                                                                                   
                                                                                                                   
  execute() must:                                                                                                  
                                                                                                                   
  1. Install the candidate harness files into an isolated NextTavern test profile.                                 
  2. Start a fresh session.                                                                                        
  3. Load the Feynman card and corpus.                                                                             
  4. Send the task prompt through DSH.                                                                             
  5. Save the response as answer.md.                                                                               
  6. Save retrieved sources and execution logs.                                                                    
                                                                                                                   
  ### 5. Configure Harness²                                                                                        
                                                                                                                   
  task_id = "explain_magnets"                                                                                      
                                                                                                                   
  result = run_task(                                                                                               
      task_id=task_id,                                                                                             
      k=3,                                                                                                         
      adapter=FeynmanAdapter(...),                                                                                 
      prop=model_backed_proposer,                                                                                  
      selector_prop=feynman_rubric_selector,                                                                       
      runs_root=adapter.spec.runs_root,                                                                            
      modes=("seq",),                                                                                              
  )                                                                                                                
                                                                                                                   
  Run all training tasks through run_domain(), then evaluate the selected harness once on the untouched hidden set.
  Compare it against the original NextTavern configuration using identical models, temperatures, corpus, and       
  questions. The hidden-set improvement is your credible success measurement. 