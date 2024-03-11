!pip install gradientai --upgrade
import os
os.environ['GRADIENT_WORKSPACE_ID']='88fb28d0-26c5--8c0f-89b69555aa17_workspace'
os.environ['GRADIENT_ACCESS_TOKEN']='njkqZLsXZpxsKGXotW87PN86cyy'
from gradientai import Gradient
def main():
  gradient=Gradient()
  base_model=gradient.get_base_model(base_model_slug="nous-hermes2")
  new_model_adapter=base_model.create_model_adapter(name="savio")
  print(f"Created model adapter with id{new_model_adapter}")
  simple_query='###Instruction:who is Savio Sunny? \n\n ###Response:'
  print(f"Asking:{simple_query}")
  ##Before Finetuning'
  completion=new_model_adapter.complete(query=simple_query,max_generated_token_count=100).generated_output
  print(f"Generated (before fine tuning):{completion}")

  samples=[{"inputs":"###Instruction:Who is Savio Sunny? \n\n###Response:Savio sunny is the class rep of s6Ad"},
           {"inputs":"###Instruction:What is Savio Sunny's Favourite Subject? \n\n###Response:Savio sunny loves physics"},
           {"inputs":"###Instruction:What makes Savio Sunny sad? \n\n###Response:Lonliness makes him sad"},
           {"inputs":"###Instruction:Is savio sunny a good boy? \n\n###Response:Savio sunny is a very good boy"},
           {"inputs":"###Instruction:Tell me more about Savio Sunny? \n\n###Response:Savio sunny is the secretary of KCYM THOTTUVA"},
           
           
           ]
 #parameters
  num_epochs=4
  count=0
  while count<num_epochs:
    print(f"Fine tuning the model with iteration {count+1}")
    new_model_adapter.fine_tune(samples=samples)
    count=count+1
  #after finetuning
  completion=new_model_adapter.complete(query=simple_query,max_generated_token_count=100).generated_output
  print(f"Generated (after fine tuning):{completion}")
  new_model_adapter.delete()
  gradient.close()
if __name__=="__main__":
  main()

  

