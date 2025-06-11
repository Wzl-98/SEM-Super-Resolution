网络训练：python sr.py -p train -c config/SEM_sr3_ddpm_linear.json<br>将config/SEM_sr3_ddpm_linear.json文件中的"path"→"resume_state"改为保存的模型参数地址<br>
网络推理：python sr.py -p val -c config/SEM_sr3_ddpm_linear.json 
