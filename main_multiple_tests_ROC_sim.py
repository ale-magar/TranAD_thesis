import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
# from beepy import beep

def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	#print('Shape of data in Convert to Windows', np.shape(data))
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
	return torch.stack(windows)

def load_dataset(dataset, oneFileFlag, num_of_train_files, num_of_test_files):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	labels_loader = []
	training_loader = np.array([])
	training_loader = []
	testing_loader = []
	validation_loader = np.array([])
	validation_loader = []
	num_of_validation_files = int(0.4*num_of_train_files)
	#num_of_train_files = int(0.6*num_of_train_files)
	print(num_train_files)
	if oneFileFlag == True:
		for file in ['train', 'test', 'labels']:
			if dataset == 'GAUSS': file = '1_GAUSS_Anomaly_Gaussnoise_' + file
			if dataset == 'SPIKE': file = '1_SPIKE_Anomaly_Gaussnoise_' + file
			loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	else:
		for i in np.arange(0,num_of_train_files,1): #0, num_of_train_files, 1): 100,num_of_train_files+99,1): 
			for file in ['train']:
				if dataset == 'GAUSS':
					file = f'{i}_GAUSS_Anomaly_Gaussnoise_'+file
				if dataset == 'SPIKE':
					file = f'{i}_SPIKE_Anomaly_Gaussnoise_'+file
				if dataset == '2STDNOISE':
					file = f'{i}_2STDNOISE_Anomaly_Gaussnoise_'+file    
				if dataset == 'GW':
					file = f'{i}_GW_Anomaly_Gaussnoise_'+file 
				if dataset == 'SG_COR_HIGHFA':
					file = f'{i}_SG_COR_HIGHFA_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF':
					file = f'{i}_SG_COR_LOWF_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF':
					file = f'{i}_SG_COR_MEDF_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_HIGHF_shift':
					file = f'{i}_SG_COR_HIGHF_shift_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_shift':
					file = f'{i}_SG_COR_LOWF_shift_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF_shift':
					file = f'{int(i)}_SG_COR_MEDF_shift_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_A125_10s_shifts':
					file = f'{i}_SG_COR_LOWF_A125_10s_shifts_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF_SNR5':
					file = f'{int(i)}_SG_COR_MEDF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF_SNR10':
					file = f'{int(i)}_SG_COR_MEDF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF_SNR15':
					file = f'{int(i)}_SG_COR_MEDF_SNR15_Anomaly_Gaussnoise_'+file
				if dataset == 'MEDF_SNR5':
					file = f'{int(i)}_SG_COR_MEDF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'MEDF_SNR10':
					file = f'{int(i)}_SG_COR_MEDF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'MEDF_SNR15':
					file = f'{int(i)}_SG_COR_MEDF_SNR15_Anomaly_Gaussnoise_'+file  
				if dataset == 'LOWF_SNR5':
					file = f'{int(i)}_SG_COR_LOWF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'LOWF_SNR10':
					file = f'{int(i)}_SG_COR_LOWF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'LOWF_SNR15':
					file = f'{int(i)}_SG_COR_LOWF_SNR15_Anomaly_Gaussnoise_'+file
				if dataset == 'HIGHF_SNR5':
					file = f'{int(i)}_SG_COR_HIGHF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'HIGHF_SNR10':
					file = f'{int(i)}_SG_COR_HIGHF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'HIGHF_SNR15':
					file = f'{int(i)}_SG_COR_HIGHF_SNR15_Anomaly_Gaussnoise_'+file  
				if dataset == 'SG_COR_LOWF_SNR5':
					file = f'{int(i)}_SG_COR_LOWF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_SNR10':
					file = f'{int(i)}_SG_COR_LOWF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_SNR15':
					file = f'{int(i)}_SG_COR_LOWF_SNR15_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_SNR20':
					file = f'{int(i)}_SG_COR_LOWF_SNR15_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_SNR25':
					file = f'{int(i)}_SG_COR_LOWF_SNR15_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_HIGHF_SNR5':
					file = f'{int(i)}_SG_COR_HIGHF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_HIGHF_SNR10':
					file = f'{int(i)}_SG_COR_HIGHF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_HIGHF_SNR15':
					file = f'{int(i)}_SG_COR_HIGHF_SNR15_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr25':
					file = f'{int(i)}_gw_cor_f200_snr20_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr20':
					file = f'{int(i)}_gw_cor_f200_snr20_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr15':
					file = f'{int(i)}_gw_cor_f200_snr15_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr10':
					file = f'{int(i)}_gw_cor_f200_snr10_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr5':
					file = f'{int(i)}_gw_cor_f200_snr5_Anomaly_Gaussnoise_'+file
				if dataset == 'GW_sim_200_snr15':
					file = f'{int(i)}_GW_sim_200_snr15_'+file    
				if dataset == 'GW_sim_200_snr10':
					file = f'{int(i)}_GW_sim_200_snr10_'+file    
				if dataset == 'GW_sim_200_snr5':
					file = f'{int(i)}_GW_sim_200_snr5_'+file    
				if dataset == 'GW_sim_900_snr15':
					file = f'{int(i)}_GW_sim_900_snr15_'+file  
				if dataset == 'GW_sim_900_snr10':
					file = f'{int(i)}_GW_sim_900_snr10_'+file  
				if dataset == 'GW_sim_900_snr5':
					file = f'{int(i)}_GW_sim_900_snr5_'+file
				if dataset == 'GW_sim_3000_snr15':
					file = f'{int(i)}_GW_sim_3000_snr15_'+file  
				if dataset == 'GW_sim_3000_snr10':
					file = f'{int(i)}_GW_sim_3000_snr10_'+file  
				if dataset == 'GW_sim_3000_snr8':
					file = f'{int(i)}_GW_sim_3000_snr8_'+file  
				if dataset == 'GW_sim_3000_snr5':
					file = f'{int(i)}_GW_sim_3000_snr5_'+file 
				if dataset == 'GW_sim_3000_snr15_ng':
					file = f'{int(i)}_GW_sim_3000_snr15_'+file  
				if dataset == 'GW_sim_3000_snr10_ng':
					file = f'{int(i)}_GW_sim_3000_snr10_'+file  
				if dataset == 'GW_sim_3000_snr8_ng':
					file = f'{int(i)}_GW_sim_3000_snr8_'+file  
				if dataset == 'GW_sim_3000_snr5_ng':
					file = f'{int(i)}_GW_sim_3000_snr5_'+file  
				if dataset == 'GW_noiseglitch3e-23':
					file = f'{int(i)}_GW_sim_3000_snr8_'+file 
				if dataset == 'GW_noiseglitch1.5e-23':
					file = f'{int(i)}_GW_sim_3000_snr8_'+file 
				if dataset == 'GW_noiseglitch0.5e-23':
					file = f'{int(i)}_GW_sim_3000_snr8_'+file 
				if dataset == 'GW_real':
					file = f'{int(i)}_gw_cor_f200_snr15_Anomaly_Gaussnoise_'+file 
				file = np.load(os.path.join(folder, f'{file}.npy'))

			#	training_loader = np.append(training_loader, [file])
				training_loader.append(file)
		# for i in np.arange(1,num_of_test_files+1,1): # if GW
 
		for i in np.arange(0,num_of_test_files,1):   
			for file in ['test']:
				if dataset == 'GAUSS':
					file = f'{i}_GAUSS_Anomaly_Gaussnoise_'+file
				if dataset == 'SPIKE':
					file = f'{i}_SPIKE_Anomaly_Gaussnoise_'+file
				if dataset == '2STDNOISE':
					file = f'{i}_2STDNOISE_Anomaly_Gaussnoise_'+file 
				if dataset == 'GW':
					file = f'100_sig_GW_Anomaly_Gaussnoise_'+file 
					print(i, file)
				if dataset == 'SG_COR_HIGHFA':
					file = f'{i}_SG_COR_HIGHFA_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF':
					file = f'{i}_SG_COR_LOWF_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF':
					file = f'{i}_SG_COR_MEDF_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_HIGHFA_shift':
					file = f'{i}_SG_COR_HIGHFA_shift_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_shift':
					file = f'{i}_SG_COR_LOWF_shift_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF_shift':
					file = f'{i}_SG_COR_MEDF_shift_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_A125_10s_shifts':
					file = f'{i}_SG_COR_LOWF_A125_10s_shifts_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF_SNR5':
					file = f'{int(i)}_SG_COR_MEDF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF_SNR10':
					file = f'{int(i)}_SG_COR_MEDF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'MEDF_SNR5':
					file = f'{int(i)}_SG_COR_MEDF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'MEDF_SNR10':
					file = f'{int(i)}_SG_COR_MEDF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'MEDF_SNR15':
					file = f'{int(i)}_SG_COR_MEDF_SNR15_Anomaly_Gaussnoise_'+file  
				if dataset == 'LOWF_SNR5':
					file = f'{int(i)}_SG_COR_LOWF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'LOWF_SNR10':
					file = f'{int(i)}_SG_COR_LOWF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'LOWF_SNR15':
					file = f'{int(i)}_SG_COR_LOWF_SNR15_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_SNR20':
					file = f'{int(i)}_SG_COR_LOWF_SNR20_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_SNR25':
					file = f'{int(i)}_SG_COR_LOWF_SNR25_Anomaly_Gaussnoise_'+file
				if dataset == 'HIGHF_SNR5':
					file = f'{int(i)}_SG_COR_HIGHF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'HIGHF_SNR10':
					file = f'{int(i)}_SG_COR_HIGHF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'HIGHF_SNR15':
					file = f'{int(i)}_SG_COR_HIGHF_SNR15_Anomaly_Gaussnoise_'+file  
				if dataset == 'SG_COR_MEDF_SNR15':
					file = f'{int(i)}_SG_COR_MEDF_SNR15_Anomaly_Gaussnoise_'+file  
				if dataset == 'SG_COR_LOWF_SNR5':
					file = f'{int(i)}_SG_COR_LOWF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_SNR10':
					file = f'{int(i)}_SG_COR_LOWF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_SNR15':
					file = f'{int(i)}_SG_COR_LOWF_SNR15_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_HIGHF_SNR5':
					file = f'{int(i)}_SG_COR_HIGHF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_HIGHF_SNR10':
					file = f'{int(i)}_SG_COR_HIGHF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_HIGHF_SNR15':
					file = f'{int(i)}_SG_COR_HIGHF_SNR15_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr25':
					file = f'{int(i)}_gw_cor_f200_snr25_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr20':
					file = f'{int(i)}_gw_cor_f200_snr20_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr15':
					file = f'{int(i)}_gw_cor_f200_snr15_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr10':
					file = f'{int(i)}_gw_cor_f200_snr10_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr5':
					file = f'{int(i)}_gw_cor_f200_snr5_Anomaly_Gaussnoise_'+file
				if dataset == 'GW_sim_200_snr15':
					file = f'{int(i)}_GW_sim_200_snr15_'+file    
				if dataset == 'GW_sim_200_snr10':
					file = f'{int(i)}_GW_sim_200_snr10_'+file    
				if dataset == 'GW_sim_200_snr5':
					file = f'{int(i)}_GW_sim_200_snr5_'+file    
				if dataset == 'GW_sim_900_snr15':
					file = f'{int(i)}_GW_sim_900_snr15_'+file  
				if dataset == 'GW_sim_900_snr10':
					file = f'{int(i)}_GW_sim_900_snr10_'+file  
				if dataset == 'GW_sim_900_snr5':
					file = f'{int(i)}_GW_sim_900_snr5_'+file
				if dataset == 'GW_sim_3000_snr15':
					file = f'{int(i)}_GW_sim_3000_snr15_'+file  
				if dataset == 'GW_sim_3000_snr10':
					file = f'{int(i)}_GW_sim_3000_snr10_'+file  
				if dataset == 'GW_sim_3000_snr8':
					file = f'{int(i)}_GW_sim_3000_snr8_'+file  
				if dataset == 'GW_sim_3000_snr5':
					file = f'{int(i)}_GW_sim_3000_snr5_'+file  
				if dataset == 'GW_sim_3000_snr15_ng':
					file = f'{int(i)}_GW_sim_3000_snr15_'+file  
				if dataset == 'GW_sim_3000_snr10_ng':
					file = f'{int(i)}_GW_sim_3000_snr10_'+file  
				if dataset == 'GW_sim_3000_snr8_ng':
					file = f'{int(i)}_GW_sim_3000_snr8_'+file     
				if dataset == 'GW_sim_3000_snr5_ng':
					file = f'{int(i)}_GW_sim_3000_snr5_'+file    
				if dataset == 'GW_noiseglitch3e-23':
					file = f'{int(i)}_GW_sim_3000_snr15_'+file 
				if dataset == 'GW_noiseglitch1.5e-23':
					file = f'{int(i)}_GW_sim_3000_snr15_'+file 
				if dataset == 'GW_noiseglitch0.5e-23':
					file = f'{int(i)}_GW_sim_3000_snr15_'+file    
				if dataset == 'GW_real':
					file = f'{int(i)}_GW_real_'+file                 
				file = np.load(os.path.join(folder, f'{file}.npy'))
				testing_loader.append(file)
		# for i in np.arange(1,num_of_test_files+1,1): # if GW
		for i in np.arange(0,num_of_test_files,1):
			for file in ['labels']:
				if dataset == 'GAUSS':
					file = f'{i}_GAUSS_Anomaly_Gaussnoise_'+file
				if dataset == 'SPIKE':
					file = f'{i}_SPIKE_Anomaly_Gaussnoise_'+file
				if dataset == '2STDNOISE':
					file = f'{i}_2STDNOISE_Anomaly_Gaussnoise_'+file 
				if dataset == 'GW':
					file = f'100_sig_GW_Anomaly_Gaussnoise_'+file 
				if dataset == 'SG_COR_HIGHFA':
					file = f'{i}_SG_COR_HIGHFA_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF':
					file = f'{i}_SG_COR_LOWF_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF':
					file = f'{i}_SG_COR_MEDF_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_HIGHFA_shift':
					file = f'{i}_SG_COR_HIGHFA_shift_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_shift':
					file = f'{i}_SG_COR_LOWF_shift_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF_shift':
					file = f'{i}_SG_COR_MEDF_shift_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_A125_10s_shifts':
					file = f'{i}_SG_COR_LOWF_A125_10s_shifts_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF_SNR5':
					file = f'{int(i)}_SG_COR_MEDF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF_SNR10':
					file = f'{int(i)}_SG_COR_MEDF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF_SNR15':
					file = f'{int(i)}_SG_COR_MEDF_SNR15_Anomaly_Gaussnoise_'+file  
				if dataset == 'MEDF_SNR5':
					file = f'{int(i)}_SG_COR_MEDF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'MEDF_SNR10':
					file = f'{int(i)}_SG_COR_MEDF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'MEDF_SNR15':
					file = f'{int(i)}_SG_COR_MEDF_SNR15_Anomaly_Gaussnoise_'+file
				if dataset == 'LOWF_SNR5':
					file = f'{int(i)}_SG_COR_LOWF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'LOWF_SNR10':
					file = f'{int(i)}_SG_COR_LOWF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'LOWF_SNR15':
					file = f'{int(i)}_SG_COR_LOWF_SNR15_Anomaly_Gaussnoise_'+file
				if dataset == 'HIGHF_SNR5':
					file = f'{int(i)}_SG_COR_HIGHF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'HIGHF_SNR10':
					file = f'{int(i)}_SG_COR_HIGHF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'HIGHF_SNR15':
					file = f'{int(i)}_SG_COR_HIGHF_SNR15_Anomaly_Gaussnoise_'+file    
				if dataset == 'SG_COR_LOWF_SNR5':
					file = f'{int(i)}_SG_COR_LOWF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_SNR10':
					file = f'{int(i)}_SG_COR_LOWF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_SNR15':
					file = f'{int(i)}_SG_COR_LOWF_SNR15_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_SNR20':
					file = f'{int(i)}_SG_COR_LOWF_SNR20_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_SNR25':
					file = f'{int(i)}_SG_COR_LOWF_SNR25_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_HIGHF_SNR5':
					file = f'{int(i)}_SG_COR_HIGHF_SNR5_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_HIGHF_SNR10':
					file = f'{int(i)}_SG_COR_HIGHF_SNR10_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_HIGHF_SNR15':
					file = f'{int(i)}_SG_COR_HIGHF_SNR15_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr25':
					file = f'{int(i)}_gw_cor_f200_snr25_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr20':
					file = f'{int(i)}_gw_cor_f200_snr20_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr15':
					file = f'{int(i)}_gw_cor_f200_snr15_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr10':
					file = f'{int(i)}_gw_cor_f200_snr10_Anomaly_Gaussnoise_'+file
				if dataset == 'gw_cor_f200_snr5':
					file = f'{int(i)}_gw_cor_f200_snr5_Anomaly_Gaussnoise_'+file
				if dataset == 'GW_sim_200_snr15':
					file = f'{int(i)}_GW_sim_200_snr15_'+file    
				if dataset == 'GW_sim_200_snr10':
					file = f'{int(i)}_GW_sim_200_snr10_'+file    
				if dataset == 'GW_sim_200_snr5':
					file = f'{int(i)}_GW_sim_200_snr5_'+file    
				if dataset == 'GW_sim_900_snr15':
					file = f'{int(i)}_GW_sim_900_snr15_'+file  
				if dataset == 'GW_sim_900_snr10':
					file = f'{int(i)}_GW_sim_900_snr10_'+file  
				if dataset == 'GW_sim_900_snr5':
					file = f'{int(i)}_GW_sim_900_snr5_'+file  
				if dataset == 'GW_sim_3000_snr15':
					file = f'{int(i)}_GW_sim_3000_snr15_'+file  
				if dataset == 'GW_sim_3000_snr10':
					file = f'{int(i)}_GW_sim_3000_snr10_'+file  
				if dataset == 'GW_sim_3000_snr8':
					file = f'{int(i)}_GW_sim_3000_snr8_'+file 
				if dataset == 'GW_sim_3000_snr5':
					file = f'{int(i)}_GW_sim_3000_snr5_'+file 
				if dataset == 'GW_sim_3000_snr15_ng':
					file = f'{int(i)}_GW_sim_3000_snr15_'+file  
				if dataset == 'GW_sim_3000_snr10_ng':
					file = f'{int(i)}_GW_sim_3000_snr10_'+file  
				if dataset == 'GW_sim_3000_snr8_ng':
					file = f'{int(i)}_GW_sim_3000_snr8_'+file  
				if dataset == 'GW_sim_3000_snr5_ng':
					file = f'{int(i)}_GW_sim_3000_snr5_'+file 
				if dataset == 'GW_noiseglitch3e-23':
					file = f'{int(i)}_GW_sim_3000_snr15_'+file 
				if dataset == 'GW_noiseglitch1.5e-23':
					file = f'{int(i)}_GW_sim_3000_snr15_'+file 
				if dataset == 'GW_noiseglitch0.5e-23':
					file = f'{int(i)}_GW_sim_3000_snr15_'+file 
				if dataset == 'GW_real':
					file = f'{int(i)}_GW_real_'+file 
				file = np.load(os.path.join(folder, f'{file}.npy'))
				#print(np.shape(file))
				labels_loader.append(file)
		

	if args.less: loader[0] = cut_array(0.2, loader[0])

 # possibly works 
	training_data_full = np.array((training_loader))
	training_data = training_data_full # [0:num_of_train_files]
	training_data = np.ravel(training_data)
	training_data = np.reshape(training_data, (np.shape(training_data)[0], 1))
	
	#training_data = np.concatenate((training_loader[0], training_loader[1], training_loader[2], training_loader[3], training_loader[4])) #np.ravel(training_loader) #, loader[3], loader[4], loader[5], loader[6], loader[7], loader[8])) #loader[0:trainnums[-1]]
	print('shape of training data = ', np.shape(training_data))
	train_loader = DataLoader(training_data, batch_size=np.shape(training_data)[0]) #:trainnums[-1] # loader2[0]
	#print('shape of testing_loader', np.shape(testing_loader))
	testing_data_full = []
	labels_data_full = []
	for i in np.arange(0, num_of_test_files, 1):
		test_data = testing_loader[i]
		#print(len(test_data))
		test_loader = DataLoader(test_data, batch_size=test_data.shape[0])
		testing_data_full.append(test_loader)
  
	for i in np.arange(0, num_of_test_files, 1):
		labels_data_name = labels_loader[i]
		#print(len(labels_data_name))
		labels = labels_data_name
		labels_data_full.append(labels)
	#labels_data_full = np.reshape(labels_data_full, (np.shape(labels_data_full), 1))
	print('shape of labels data = ', np.shape(labels_data_full))
	print('shape of testing data = ', np.shape(testing_data_full))


	return train_loader, testing_data_full, labels_data_full# , val_loader

def save_model(model, optimizer, scheduler, epoch, accuracy_list, accuracy_list_val):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list,
        'accuracy_list_val': accuracy_list_val}, file_path)

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
		accuracy_list_val = checkpoint['accuracy_list_val']
  
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []; accuracy_list_val = [];
	return model, optimizer, scheduler, epoch, accuracy_list, accuracy_list_val

def backprop(epoch, model, data, dataO, optimizer, scheduler, training, validation):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = dataO.shape[1]

	if 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		print(np.shape(data))
		data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d, _ in dataloader:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		elif validation:
			#torch.zero_grad = True
			#model.eval()
			for d, _ in dataloader:
				model.eval()
				local_bs = d.shape[0]
		###	model.eval()
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
			#	#loss.backward(retain_graph=True)
			#	optimizer.zero_grad()
			tqdm.write(f'Epoch {epoch},\tL1_val = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		#	scheduler.step()

		else:
			for d, _ in dataloader:
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, bs, feats)
				z = model(window, elem)
				if isinstance(z, tuple): z = z[1]
			loss = l(z, elem)[0]
			return loss.detach().numpy(), z.detach().numpy()[0]
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().numpy(), y_pred.detach().numpy()


if __name__ == '__main__':
    # The below parameters need set for every run!
	num_train_files = 100
	num_test_files = 50

	test_signal = True # if the data has signals == True, if no signals == False
 
	# load the data
	train_loader, testing_data_full, labels_data_full = load_dataset(args.dataset, False, num_train_files, num_test_files)
	num_val_files = int(num_train_files * 0.2)
	num_train_files = int(num_train_files * 0.8)
	sampling_rate = 2048
	test_sampling_rate = 512
	#print('labels_data_full shape ', labels_data_full[0].shape[1])#, #np.shape(testing_data_full)
	model, optimizer, scheduler, epoch, accuracy_list , accuracy_list_val = load_model(args.model, 1) #labels_data_full[0].shape[1]) #labels_data_full[0].shape[1])
 
	## Prepare data
	trainD = next(iter(train_loader))
	valD = trainD[int(sampling_rate*num_train_files) :int(sampling_rate * (num_train_files+num_val_files))]# next(iter(val_loader))
	print('Shape of Validaiton Data: ', np.shape(valD))
	trainD = trainD[0:int(sampling_rate * num_train_files)]
	print('Shape of Training Data: ', np.shape(trainD))
	trainO = trainD
	valO = valD
	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: 
		trainD = convert_to_windows(trainD, model)
		valD = convert_to_windows(valD, model)
 
	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = 40; e = epoch + 1; start = time() # 
		plottertrain_prior(f'{args.model}_{args.dataset}',trainO)
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler, training=True, validation=False)
			lossT_val, lr_val = backprop(e, model, valD, valO, optimizer, scheduler, training=False, validation=True)
   
			accuracy_list.append((lossT, lr))
			accuracy_list_val.append((lossT_val, lr_val))
   
		print('shape of accuracy lists = ',np.shape(accuracy_list_val), np.shape(accuracy_list))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		plottertrain(f'{args.model}_{args.dataset}',trainO)
		save_model(model, optimizer, scheduler, e, accuracy_list, accuracy_list_val)
		plot_accuracies(accuracy_list, accuracy_list_val, f'{args.model}_{args.dataset}')


	## Testing phase 

	for limit in np.arange(0, 10, 1): # 1, 100, 1
		signal_hit = 0 # records the number of files that have > a set number of hits out of the whole testing set
		false_hit = 0 # records the number of files with >0 hits above the higher limit for files without signals
		hits_above_lim1 = []
		hits_above_lim1_noSignal = []
 
		#test_name = 'Frequency3000_SNR15_noiseA1_lr0.001_epoch20_140225'  # alter
		test_name = args.test_name
		attempt_num = '_LossThresholds_'+str(limit)+'_'
		file_name = 'gw_sim_'+test_name 
		file_path = file_name+'/'
		os.system('mkdir hits/'+file_name)
		test_path = './hits/'+file_path+test_name+attempt_num
		for i in np.arange(0, num_test_files, 1):
			test_loader = testing_data_full[i]
			labels = labels_data_full[i]
			testD = next(iter(test_loader))
			testO = testD
			if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: 
				testD = convert_to_windows(testD, model)
			torch.zero_grad = True
			model.eval()
			print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
			loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False, validation=False)
			#lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False, validation=False)

			plotfiles = np.arange(0,100,10)
			for num in plotfiles:
				if i == num:
					filename = 'AnomalyPlot_'+str(i)+'.pdf'
					plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels, filename)
			if not args.test:
				if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0) 
				filename = 'AnomalyPlot_'+str(i)+'.pdf'
				#plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels, filename)
	
			backgroundloss =[]
			for ind in range(len(labels)):
				if labels[ind] == 0:
					backgroundloss.append(loss[ind])
			
			meanBG = np.mean(backgroundloss)
			stdBG = np.std(backgroundloss)
			lim1 = meanBG + (limit*stdBG) #np.max(np.abs(loss))*0.5
			
			hits_lim1 = 0 # number of hits per file above limit 2 
			
			hits_lim1_noSignal = 0
			total_length = len(testO)
			signal_length = total_length - (test_sampling_rate * 4)
			length_of_sections =  int(total_length/10) # test_sampling rate
			number_of_sections = total_length/length_of_sections

			false_hit_in_sec = 0
			for i in range(len(labels)):
				if labels[i] == 1 and loss[i] > lim1:
					hits_lim1 += 1
			if hits_lim1 > 20: # change this
				signal_hit += 1 
						#print('True')
			for l in np.arange(0,total_length, length_of_sections):
				#print(l, l+length_of_sections)
				loss_section = loss[l:l+length_of_sections]
				labels_section = labels[l:l+length_of_sections]
			#	if np.max(labels_section) == 1: print('Maximumum Label in Section = ', np.max(labels_section))
				for i in range(len(loss_section)):
					if loss[i] > lim1:
						hits_lim1_noSignal += 1
				l+=length_of_sections
				if hits_lim1_noSignal > 20:	 # change this 
					false_hit_in_sec += 1
			#print('False hit = ', false_hit_in_sec)
			#print('Signal hit = ', signal_hit_in_sec)
			hits_above_lim1.append(hits_lim1)
			hits_above_lim1_noSignal.append(hits_lim1_noSignal)
	
			#if signal_hit_in_sec > 1: # this number should be tailored to an acceptable value (the number of true signal hits will be frequency/signal length dependent)
		#		signal_hit += 1
			if false_hit_in_sec > 1:
				false_hit +=1
			
			#lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
			#lossTfinal, lossFinal = np.mean(lossT), np.mean(loss)
			#labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
			#result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
			#result.update(ndcg(loss, labels))
			#print(result)
		##	result.update(hit_att(loss, labels))
			
 
		print('Number of true signal anomly hits in total = ', signal_hit)
		print('Number of false signal anomly hits in total = ', false_hit)
		plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels, filename)

		TruePositive = signal_hit/num_test_files 
		FalseNegative = (num_test_files-signal_hit)/num_test_files
	
		FalsePositive = false_hit/num_test_files
		TrueNegative = (num_test_files-false_hit)/num_test_files
		total_hits_above_lim1 = np.sum(hits_above_lim1)
		total_hits_above_lim1_noSignal = np.sum(hits_above_lim1_noSignal)

		np.save(test_path+'TruePositive.npy', TruePositive)
		np.save(test_path+'FalseNegative.npy', FalseNegative)
		np.save(test_path+'FalsePositive.npy', FalsePositive)
		np.save(test_path+'TrueNegative.npy', TrueNegative)
		print('True Positive = ', TruePositive, ', False Positive = ', FalsePositive)
		
		break

	### Scores
	'''
	df = pd.DataFrame()
	lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result, pred = pot_eval(lt, l, ls); preds.append(pred)
	#	df = df.append(result, ignore_index=True)
		df = df._append(result, ignore_index=True)
		break
	# preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
	# pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')

	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	#lossTfinal, lossFinal = np.mean(lossT), np.mean(loss)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)

	result.update(hit_att(loss, labels))
	result.update(ndcg(loss, labels))

	pprint(result)
	# pprint(getresults2(df, result))
	# beep(4)
	'''
 
 		#	for i in np.arange(0, num_val_files, 1):
		#		print('Inside Loop')
			#	valD = next(iter(val_loader))
			#	valO = valD
			#	valD = np.reshape(valD, (1, 2048))
		###		val_loader = validation_data_full[int(i)]
			#	print('Validation Data', valD)
			#	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: 
			#		valD = convert_to_windows(valD, model)
			#	lossT_val, lr_val = backprop(e, model, valD, valO, optimizer, scheduler, training=False)
			#	losses_val.append(lossT_val)
			#	print(loassT_val)
			#	lrs_val.append(lr_val)
   
'''
		for i in np.arange(num_of_train_files+1, num_of_validation_files+num_of_train_files+1, 1):
			for file in ['train']:
				if dataset == 'GAUSS':
					file = f'{i}_GAUSS_Anomaly_Gaussnoise_'+file
				if dataset == 'SPIKE':
					file = f'{i}_SPIKE_Anomaly_Gaussnoise_'+file
				if dataset == '2STDNOISE':
					file = f'{i}_2STDNOISE_Anomaly_Gaussnoise_'+file    
				if dataset == 'GW':
					file = f'{i}_GW_Anomaly_Gaussnoise_'+file 
				if dataset == 'SG_COR_HIGHFA':
					file = f'{i}_SG_COR_HIGHFA_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF':
					file = f'{i}_SG_COR_LOWF_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF':
					file = f'{i}_SG_COR_MEDF_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_HIGHFA_shift':
					file = f'{i}_SG_COR_HIGHFA_shift_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_shift':
					file = f'{i}_SG_COR_LOWF_shift_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_MEDF_shift':
					file = f'{i}_SG_COR_MEDF_shift_Anomaly_Gaussnoise_'+file
				if dataset == 'SG_COR_LOWF_A125_10s_shifts':
					file = f'{i}_SG_COR_LOWF_A125_10s_shifts_Anomaly_Gaussnoise_'+file
				file = np.load(os.path.join(folder, f'{file}.npy'))
				#validation_loader = np.append(validation_loader, file)
				validation_loader.append(file)
'''
