import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile

#datasets = ['GUASS', 'SPIKE', '2STDNOISE', 'GW']

normalise_value = 1e-44 * 2 

#wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return (a / 2 + 0.5)


def load_data(dataset):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	
	if dataset == 'GAUSS':
		dataset_folder = 'data/GAUSS'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', np.zeros((20000,  1)))
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = np.array([test])
			test = test.T
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = np.array([labels])
			labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'SPIKE':
		dataset_folder = 'data/SPIKE'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = np.array([test])
			test = test.T
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)
	if dataset == '2STDNOISE':
		dataset_folder = 'data/2STDNOISE'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = np.array([test])
			test = test.T
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)
	if dataset == 'GW':
		dataset_folder = 'data/GW'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = np.array([test])
			test = test.T
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)
	if dataset == 'SG_COR_HIGHFA':
		dataset_folder = 'data/SG_COR_HIGHFA'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
		#	train= normalize(train)
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			
			print(np.shape(test))
		#	test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'SG_COR_LOWF':
		dataset_folder = 'data/SG_COR_LOWF'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
		#	train= normalize(train)
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			
			print(np.shape(test))
		#	test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)
		#for file in ['train', 'test', 'labels']:
		#	np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
   
	if dataset == 'SG_COR_MEDF':
		dataset_folder = 'data/SG_COR_MEDF'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			#train= normalize(train)
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			
			print(np.shape(test))
			#test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)
   
	if dataset == 'SG_COR_HIGHF_shift':
		dataset_folder = 'data/SG_COR_HIGHF_shift'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			#train= normalize(train)
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			
			print(np.shape(test))
			#test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'SG_COR_LOWF_shift':
		dataset_folder = 'data/SG_COR_LOWF_shift'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			#train= normalize(train)
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			
			print(np.shape(test))
		#	test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)
		#for file in ['train', 'test', 'labels']:
		#	np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
   
	if dataset == 'SG_COR_MEDF_shift':
		dataset_folder = 'data/SG_COR_MEDF_shift'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)


	if dataset == 'SG_COR_LOWF_A125_10s_shifts':
		dataset_folder = 'data/SG_COR_LOWF_A125_10s_shifts'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
		#	train= normalize(train)
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			
			print(np.shape(test))
			#test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)


	if dataset == 'SG_COR_LOWF_SNR15':
		dataset_folder = 'data/SG_COR_LOWF_SNR15'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+np.abs(np.min(train))

			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))

			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'SG_COR_LOWF_SNR10':
		dataset_folder = 'data/SG_COR_LOWF_SNR10'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+np.abs(np.min(train))

			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))

			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'SG_COR_LOWF_SNR5':
		dataset_folder = 'data/SG_COR_LOWF_SNR5'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+np.abs(np.min(train))

			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))

			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)
   
	if dataset == 'SG_COR_LOWF_SNR20':
		dataset_folder = 'data/SG_COR_LOWF_SNR20'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+np.abs(np.min(train))

			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))

			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'SG_COR_LOWF_SNR25':
		dataset_folder = 'data/SG_COR_LOWF_SNR25'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+np.abs(np.min(train))

			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))

			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)


	if dataset == 'SG_COR_MEDF_SNR15':
		dataset_folder = 'data/SG_COR_MEDF_SNR15'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
	#		train= normalize(train)
			train=train/normalise_value
			train = train+np.abs(np.min(train))

			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			test=test/normalise_value
			test = test+np.abs(np.min(test))

			print(np.shape(test))
		#	test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'SG_COR_MEDF_SNR10':
		dataset_folder = 'data/SG_COR_MEDF_SNR10'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
		#	train= normalize(train)
			train=train/normalise_value
			train = train+np.abs(np.min(train))

			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			test=test/normalise_value
			test = test+np.abs(np.min(test))

			print(np.shape(test))
		#	test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
		#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'SG_COR_MEDF_SNR5':
		dataset_folder = 'data/SG_COR_MEDF_SNR5'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
		#	train= normalize(train)
			train=train/normalise_value
			train = train+np.abs(np.min(train))

			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			test=test/normalise_value
			test = test+np.abs(np.min(test))

			print(np.shape(test))
		#	test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'SG_COR_HIGHF_SNR15':
		dataset_folder = 'data/SG_COR_HIGHF_SNR15'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
		#	train= normalize(train)
			#print(np.max(train))
			train=train/normalise_value
			train = train+np.abs(np.min(train))

			#print(np.max(train))

			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#print(np.max(test))
			#print(np.max(test))
			test=test/normalise_value
			test = test+np.abs(np.min(test))

			#print(np.max(test))

			print(np.shape(test))
			#test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'SG_COR_HIGHF_SNR10':
		dataset_folder = 'data/SG_COR_HIGHF_SNR10'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
		#	train= normalize(train)
			train=train/normalise_value
			train = train+np.abs(np.min(train))

			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			test=test/normalise_value
			test = test+np.abs(np.min(test))

			print(np.shape(test))
		#	test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'SG_COR_HIGHF_SNR5':
		dataset_folder = 'data/SG_COR_HIGHF_SNR5'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
		#	train= normalize(train)
			train=train/normalise_value
			train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			test=test/normalise_value
			test = test+np.abs(np.min(test))
			print(np.shape(test))
		#	test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)




### GRAVITATIONAL WAVES

	if dataset == 'gw_cor_f200_snr25':
		dataset_folder = 'data/gw_cor_f200_snr25'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'gw_cor_f200_snr20':
		dataset_folder = 'data/gw_cor_f200_snr20'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'gw_cor_f200_snr15':
		dataset_folder = 'data/gw_cor_f200_snr15'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'gw_cor_f200_snr10':
		dataset_folder = 'data/gw_cor_f200_snr10'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'gw_cor_f200_snr5':
		dataset_folder = 'data/gw_cor_f200_snr5'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'gw_cor_f200_snr25':
		dataset_folder = 'data/gw_cor_f200_snr25'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'gw_cor_f200_snr25':
		dataset_folder = 'data/gw_cor_f200_snr25'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

## GW REAL

	if dataset == 'GW_real':
		dataset_folder = 'data/GW_real'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)








	if dataset == 'GW_sim_3000_snr15':
		dataset_folder = 'data/GW_sim_3000_snr15'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		print(filelist2[0])
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			#print(f)
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			print(np.shape(test))
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			#print(f)
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'GW_sim_3000_snr10':
		dataset_folder = 'data/GW_sim_3000_snr10'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			print(f)
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'GW_sim_3000_snr8':
		dataset_folder = 'data/GW_sim_3000_snr8'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'GW_sim_3000_snr5':
		dataset_folder = 'data/GW_sim_3000_snr5'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			#train= normalize(train)
			train=train/normalise_value
			train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			test=test/normalise_value
			test = test+np.abs(np.min(test))
			print(np.shape(test))
			#test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)




	if dataset == 'GW_sim_3000_snr15_ng':
		dataset_folder = 'data/GW_sim_3000_snr15_ng'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		print(filelist2[0])
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			#print(f)
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			print(np.shape(test))
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			#print(f)
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'GW_sim_3000_snr10_ng':
		dataset_folder = 'data/GW_sim_3000_snr10_ng'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			print(f)
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'GW_sim_3000_snr8_ng':
		dataset_folder = 'data/GW_sim_3000_snr8_ng'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)
	
	if dataset == 'GW_sim_3000_snr5_ng':
		dataset_folder = 'data/GW_sim_3000_snr5_ng'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)



	if dataset == 'GW_sim_200_snr15':
		dataset_folder = 'data/GW_sim_200_snr15'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'GW_sim_200_snr10':
		dataset_folder = 'data/GW_sim_200_snr10'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)


	if dataset == 'GW_sim_200_snr5':
		dataset_folder = 'data/GW_sim_200_snr5'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'GW_sim_900_snr15':
		dataset_folder = 'data/GW_sim_900_snr15'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'GW_sim_900_snr10':
		dataset_folder = 'data/GW_sim_900_snr10'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'GW_sim_900_snr5':
		dataset_folder = 'data/GW_sim_900_snr5'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			#train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			#test=test/normalise_value
			#test = test+np.abs(np.min(test))
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'GW_noiseglitch3e-23':
		dataset_folder = 'data/GW_noiseglitch3e-23'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			#train= normalize(train)
			train=train/normalise_value
			train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			test=test/normalise_value
			test = test+np.abs(np.min(test))
			print(np.shape(test))
			#test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'GW_noiseglitch1.5e-23':
		dataset_folder = 'data/GW_noiseglitch1.5e-23'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			#train= normalize(train)
			train=train/normalise_value
			train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			test=test/normalise_value
			test = test+np.abs(np.min(test))
			print(np.shape(test))
			#test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)
   
	if dataset == 'GW_noiseglitch0.5e-23':
		dataset_folder = 'data/GW_noiseglitch0.5e-23'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			train=train/normalise_value
			#train = train+(np.abs(np.min(train)))
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = test#[0:100]
			test = np.array([test])
			test = test.T
			test=test/normalise_value
			test = test+np.abs(np.min(test))
			print(np.shape(test))
			#test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.loadtxt(f'{dataset_folder}/labels/{f}')
			labels = labels#[0:100]
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")
