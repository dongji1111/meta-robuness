# meta-robuness
This is the code for roubust meta-learning project

To train the model, run train.py
different args:

   --epoch: epoch number', default=1000
   
   --n_way: n way in meta model, default=10
   
   --k_spt: k shot for support set in meta model, default=1
   
   --k_qry: k shot for query setin meta model, default=10
   
   --task_num: meta batch size, namely task num, default=4
   
   --meta_lr: meta-level outer learning rate, default=1e-3
   
   --update_lr: task-level inner update learning rate, default=0.01
   
   --update_step: task-level inner update steps, default=5
   
   --update_step_test: update steps for finetunning', default=10
   
   
   meta.py: code for meta-learner
   learner.py: code for base-learner of the meta model
   xxx.py and xxx_attack.py : the attack we used as the subtask of the meta model
