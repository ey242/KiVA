
import pandas as pd

class Tracker():
	
	def __init__(self, output_path):
		self.tracker = self.init_tracker()
		self.output_path = output_path	

	def init_tracker(self):
		data_point = {
				"Variation": [],
				"Regeneration": [],
				"Train_input_param": [],
				"Train_output_param": [],
				"Test_input_param": [],
				"Test_output_param": [],
				"Full#1": [],
				"Full#2": [],
				"Full#3": [],
				"MCResponse#1": [],
				"MCResponse#2": [],
				"MCResponse#3": [],
				"Response#1": [],
				"Response#2": [],
				"Response#3": [], 
				"Param": [],
			}
		return data_point
	
	def update(self, data_point): 
		for key in self.tracker.keys():
			assert key in data_point.keys(), f"Key {key} not found in data_point"

		for key in self.tracker.keys():
			self.tracker[key].append(data_point[key])
		

	def save_df(self): 
		#save the data with unique param into its own df 
		df = pd.DataFrame(self.tracker)
		for param in df['Param'].unique():
			df_param = df[df['Param'] == param]
			df_param = df_param.drop(columns=['Param'])
			df_param.to_csv(f"{self.output_path}/{param}.csv", index=False)


