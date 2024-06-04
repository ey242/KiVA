import os 

records_dir = "records"

for concept in ["2DRotation"]:
    command = f"qsub -v args='--concept {concept} --model llava' -o {records_dir}/{concept} run.sh" 
    # command = f"qsub -v args='--concept {concept} --model gpt4' -o {records_dir}/{concept} run_gpt4.sh" 
    os.system(command)