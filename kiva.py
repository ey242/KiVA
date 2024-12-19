import os
from kiva_stream import KiVAStream
from tracker import Tracker


SUPPORTED_PAIRS = [("gpt4", "single"), ("gpt4o", "single"), ("llava", "single"), 
                ("gpt4", "multi"), ("gpt4o", "multi"), ("mantis", "multi")]


def eval_response(response, correct_answers, incorrect_answers, all_choices): 
	
	all_choices_option = [x.split(" ")[0] for x in all_choices]	
	correct_answers_option = [x.split(" ")[0] for x in correct_answers] 
	incorrect_answers_option = [x.split(" ")[0] for x in incorrect_answers]

	all_response_options = {}
	for choice in all_choices_option:
		if choice in response:
			all_response_options[choice] = response.index(choice)

	if len(all_response_options) == 0:
		return False
	
	extracted_choice = min(all_response_options, key=all_response_options.get)
	

	if extracted_choice in correct_answers_option:
		result_boolean = "1"
		result_text = all_choices[all_choices_option.index(extracted_choice)]
	elif extracted_choice in incorrect_answers_option:
		result_boolean = "0"
		result_text = all_choices[all_choices_option.index(extracted_choice)]
	else: 
		result_boolean = "Null"
		result_text = "Null"

	return result_boolean, result_text


def main():

    #set up args parse 
    import argparse
    parser = argparse.ArgumentParser(description='Run KiVA')
    parser.add_argument('--concept', type=str,  default="Reflect", help='Concept to evaluate')
    parser.add_argument('--model_name', type=str,  default="gpt4o", help='Model to use')
    parser.add_argument('--mode', type=str,  default = "KiVA", help='Mode to use')
    parser.add_argument('--image_num', type=str,  default = "single", help='Number of images to use')

    args = parser.parse_args()

    stimuli_directory = f"transformed_objects/{args.mode}/{args.concept}" # Insert object file directory
    text_files_dir = f"transformed_objects/{args.mode}/trial_tracker/"
    output_directory = f"models_data/{args.mode}/{args.model_name}_{args.image_num}_image/output_{args.concept}"
    os.makedirs(output_directory, exist_ok=True)

    system_prompt = open(f"prompts/{args.image_num}/system_prompt.txt", "r").read()
    api_key = ""

    if args.model_name == "llava":
        from models.llava_model import LLavaModel
        chat_model =  LLavaModel(system_prompt, max_token = 300)

    if args.model_name == "gpt4":
        from models.gpt4_model import GPT4Model
        assert api_key is not None, "API key is required for GPT4 model"
        chat_model = GPT4Model(system_prompt, api_key=api_key, max_token=300)

    elif args.model_name == "gpt4o":
        from models.gpt4o_model import GPT4OModel
        assert api_key is not None, "API key is required for GPT4 model"
        chat_model = GPT4OModel(system_prompt, api_key=api_key, max_token=300)

    else: 
        raise ValueError("Model name not recognized.")
	

    assert (args.model_name, args.image_num) in SUPPORTED_PAIRS, f"Model {args.model_name} with args.image_num {args.image_num} not supported."
        
    data = KiVAStream(args.concept, args.image_num, args.mode, stimuli_directory, text_files_dir, args.model_name)
    tracker = Tracker(output_directory)

    for items in data: 

        tracker_data_point = {}

        images = items["images"]
        exps_details = items["exps_details"]

        for exp in exps_details:
            tracker_data_point[exp] = exps_details[exp]
        

        if args.image_num == "single":
            out_chatmodel = chat_model.run_model(items["prompts"]["general_cross_rule_prompt"], items["images"][0])
        elif args.image_num == "multi": 
            out_chatmodel = chat_model.run_model_indiv(items["prompts"]["general_cross_rule_prompt"], items["images"][0])

        tracker_data_point["Full#1"] = out_chatmodel["response"]

        print("Cross Domain Response: ", out_chatmodel["response"])

        answers_cross = items["answers"]["general_cross"]
        result_boolean, result_text = eval_response(out_chatmodel["response"], 
                                                    answers_cross["correct_answers"], 
                                                    answers_cross["incorrect_answers"],
                                                    answers_cross["all_answers"])
        
        tracker_data_point["MCResponse#1"] = result_boolean 
        tracker_data_point["Response#1"] = result_text

        if result_boolean == "1":
            
            if args.image_num == "single":
                out_chatmodel = chat_model.run_model(items["prompts"]["general_within_rule_prompt"])
            elif args.image_num == "multi":
                out_chatmodel = chat_model.run_model_indiv(items["prompts"]["general_within_rule_prompt"])

            tracker_data_point["Full#2"] = out_chatmodel["response"]

            print("Within Domain Response: ", out_chatmodel["response"])

            answers_within = items["answers"]["general_within"]
            result_boolean, result_text = eval_response(out_chatmodel["response"], 
                                                        answers_within["correct_answers"], 
                                                        answers_within["incorrect_answers"],
                                                        answers_within["all_answers"])
            
            tracker_data_point["MCResponse#2"] = result_boolean
            tracker_data_point["Response#2"] = result_text

        else: 
            tracker_data_point["Full#2"] = "[Skipped mcq#2, incorrect previous response]"
            tracker_data_point["MCResponse#2"] = ""
            tracker_data_point["Response#2"] = ""
        

        if args.image_num == "single":
            out_chatmodel = chat_model.run_model(items["prompts"]["extrapolation_prompt"])
        if args.image_num == "multi":
            out_chatmodel = chat_model.run_model_multi(items["prompts"]["extrapolation_prompt"], images[1:])    
        
        tracker_data_point["Full#3"] = out_chatmodel["response"]
        answers_extrapolation = items["answers"]["extrapolation"]

        print("Extrapolation Response: ", out_chatmodel["response"])

        result_boolean, result_text = eval_response(out_chatmodel["response"],
                                                    answers_extrapolation["correct_answers"], 
                                                    answers_extrapolation["incorrect_answers"],
                                                    answers_extrapolation["all_answers_with_params"])

        tracker_data_point["MCResponse#3"] = result_boolean
        tracker_data_point["Response#3"] = result_text


        tracker.update(tracker_data_point)

    tracker.save_df()


if __name__ == "__main__":
    main()