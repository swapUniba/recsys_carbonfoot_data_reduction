import sys
import os
import pandas as pd
import csv
import json
import time

from recbole.quick_start import run_recbole
from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction
from recbole.utils import get_trainer

from codecarbon import EmissionsTracker


def run_model_dataset(model, dataset):


	# 'check.tsv' is a useful file used to check if the model has been executed with no problem
	# this is partcularly useful when using remote server
	done = False
	log_file = f'epo200_checks/{model}/{dataset}/check.tsv'

	if os.path.exists(log_file):
		with open(log_file, 'r') as fin:
			line = fin.readline().strip()
			if line == 'Done':
				done = True

	# if the model has been already trained, just skip it
	if done:
		return

	parameter_dict = {
						'epochs': 200,
						'checkpoint_dir': f'epo200_checks/{model}/{dataset}/',
						'benchmark_filename': ['train', 'valid', 'test'],
						'metrics':  [ 'Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 
									'Precision', 'GAUC', 'ItemCoverage', 'AveragePopularity', 'GiniIndex',
									'ShannonEntropy', 'TailPercentage']
	}

	# start the carbon footprint tracking 
	with EmissionsTracker(project_name=f'{model}_{dataset}', output_file='epo200_emissions.tsv') as tracker:
		
		try:

			# start the carbon footprint tracking, train the model, stop the tracking
			tracker.start()
			run_recbole(model=f'{model}', dataset=f'{dataset}', config_dict=parameter_dict)
			tracker.stop()

			# read all gathered data and save to carbon file
			codecarbon_results = vars(tracker)

			with open(log_file, 'w') as fout:
				fout.write('Done')

		except Exception as e:

			# if something happens, write the exception on the check.tsv file 
			# this will allow the model to be executed again
			print(f"ERROR: {e}")
			with open(log_file, 'w') as fout:
				fout.write(f"ERROR: {e}")

	# find the .pth model file that has been trained - it is not an elegant solution,
	# but it is the only one adoperable in RecBole, since it does not allow to customize
	# the name of the .pth file will be saved
	all_models = os.listdir(f'epo200_checks/{model}/{dataset}/')
	for m in all_models:
		if model in m:
			model_path =  f'epo200_checks/{model}/{dataset}/{m}'
			break

	os.makedirs(f"epo200_results/{model}/{dataset}/")


	# compute recommendation metrics
	config, model_pth, dataset_pth, train_data, valid_data, test_data = load_data_and_model(model_path)
	trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model_pth)
	trainer.eval_collector.data_collect(train_data)
	trainer.saved_model_file = model_path
	metrics = trainer.evaluate(test_data)

	# define output file names
	out_emissions = f"epo200_results/{model}/{dataset}/emissions_{model_path.split('/')[-1].replace('.pth','.tsv')}"
	out_recbole = f"epo200_results/{model}/{dataset}/recbole_{model_path.split('/')[-1].replace('.pth','.tsv')}"

	# save both the results
	with open(out_emissions, "w") as outfile:
		csvwriter = csv.writer(outfile, delimiter='\t')
		csvwriter.writerow(dict(codecarbon_results))
		csvwriter.writerow(dict(codecarbon_results).values())

	with open(out_recbole, 'w') as outfile:
		csvwriter = csv.writer(outfile, delimiter='\t')
		csvwriter.writerow(dict(metrics))
		csvwriter.writerow(dict(metrics).values())



if __name__ == "__main__":

	# get dataset and model name
	args = sys.argv[1:]

	# just print them to be sure everything is fine :)
	print(args[0])
	print(args[1])

	model = args[0]
	dataset = args[1]
	
	# run the model on that dataset
	run_model_dataset(model=model, dataset=dataset)

	sys.exit(0)