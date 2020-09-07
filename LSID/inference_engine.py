import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

import time
import scipy.io
import torch
import torch.nn as nn

import datasets
import models.lsid as LSID
from trainer import Trainer, Validator
import utils
import tqdm

import gc
import os
import sys

import datasets
import utils
import tqdm
import scipy.io 

ABSPATH = "C:/Users/lab_phdi/Desktop/OpenVINO/LSID/"

cuda = torch.cuda.is_available()
kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}

#test dataloader
dataset_class = datasets.__dict__["Sony"]
dt = dataset_class("./dataset", './dataset/Sony_test_list.txt', split='test', gt_png=True, use_camera_wb=True, upper=-1)
test_loader = torch.utils.data.DataLoader(dt, batch_size=1, shuffle=False, **kwargs)
print(len(test_loader))
# dt = dataset_class("./Sony", "./Sony/Sony_test_list.txt", split='test')
# test_loader = torch.utils.data.DataLoader(dt, 1, shuffle=False, **kwargs)

class OriginalInference:
	def __init__(self):
		self.net = LSID.lsid(inchannel=4, block_size=2)
		self.criterion = nn.L1Loss()

		self.iteration = 0 
		self.epoch = 0
		self.val_loader = test_loader
		self.cmd = 'Test'
		self.result_dir = 'C:/Users/lab_phdi/Desktop/OpenVINO/LSID'

		self.exec_net = self.net.load_state_dict(torch.load("./checkpoint/Sony/checkpoint-99.pth.tar", map_location=torch.device('cpu'))['model_state_dict'])

	def run(self):
		batch_time = utils.AverageMeter()
		losses = utils.AverageMeter()
		psnr = utils.AverageMeter()
		ssim = utils.AverageMeter()

		self.net.eval()
		end = time.time()
		for batch_idx, (raws, imgs, targets, img_files, img_exposures, lbl_exposures, ratios) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='{} iteration={} epoch={}'.format('Valid' if self.cmd == 'train' else 'Test', self.iteration, self.epoch), ncols=80, leave=False):

			gc.collect()

			with torch.no_grad():
				raws = Variable(raws)
				targets = Variable(targets)
				output = self.net(raws)

			outputs = torch.clamp(output, 0, 1).cpu()
			targets = targets.cpu()

			for output, img, target, img_file, img_exposure, lbl_exposure, ratio in zip(outputs, imgs, targets, img_files, img_exposures, lbl_exposures, ratios):
				output = output.numpy().transpose(1, 2, 0) * 255
				target = target.numpy().transpose(1, 2, 0) * 255

				os.makedirs(self.result_dir, exist_ok=True)
				fname = os.path.join(self.result_dir, '{}_compare.jpg'.format(os.path.basename(img_file)[:-4]))
				temp = np.concatenate((target[:, :, :], output[:, :, :]), axis=1)
				scipy.misc.toimage(temp, high=255, low=0, cmin=0, cmax=255).save(fname)

				# psnr.update(utils.get_psnr(output, target), 1)
				_psnr = utils.get_psnr(output, target)
				# print("PSNR", img_file, _psnr)
				psnr.update(_psnr, 1)
				_ssim = utils.get_ssim(output, target)
				# print("SSIM", img_file, _ssim)
				ssim.update(_ssim, 1)


class InferenceEngineOpenVINO:
	def __init__(self):
		from openvino.inference_engine import IENetwork, IECore, IEPlugin
		ie = IECore() #IE instance
		xmlfile = ABSPATH + "Sony_resume_test.xml" 
		binfile = ABSPATH + "Sony_resume_test.bin"
		# self.net = IENetwork(model=xmlfile, weights=binfile) #Read OR model
		self.net = ie.read_network(model=xmlfile, weights=binfile)
		self.criterion = nn.L1Loss()

		self.iteration = 0 
		self.epoch = 0

		self.result_dir = 'C:/Users/lab_phdi/Desktop/OpenVINO/LSID/'

		# self.plugin = IEPlugin(device="CPU")
		#self.plugin = IEPlugin(device="GPU")  if you have intel GPU not nvidia  
		self.exec_net = ie.load_network(network = self.net, device_name="CPU")
		# self.exec_net = IEPlugin.load(network=self.net, config={"DEVICE_ID":"0"})
		self.input_blob = next(iter(self.net.inputs))
		self.output_blob = next(iter(self.net.outputs))

		print("========================================================")
		print("Load Network Topology : {}".format(xmlfile))
		print("Load Network Weight : {}".format(binfile))
		print("Input_Shape: {}".format(self.net.inputs[self.input_blob].shape))
		print("Output_Shape: {}".format(self.net.outputs[self.output_blob].shape))
		print("========================================================")

	#========================================================================#
	"""
	# note : 
			x = (1,28,28) float numpy array
			self.exec_net.infer(inputs={self.input_blob:x})[self.output_blob] => (1,10) numpy array 
	 		np.argmax(,axis=1) => (1,1) array
	"""
	def infer(self,x):
		return np.argmax(self.exec_net.infer(inputs={self.input_blob:x})[self.output_blob],axis=1)[0]
		
	# def run(self):
	# 	data = ReadMINST()
	# 	n_train = data.xtrain.size(0)
	# 	n_valid = data.xtest.size(0)

	# 	while 1:
	# 		#=============================================================
	# 		r1 = torch.randint(0,n_train,(1,)).item()
	# 		# 從 train 挑
	# 		x = data.xtrain[r1].numpy()
	# 		x = x.reshape(1,28,28)
	# 		y = data.ytrain[r1].item()
	# 		print("預測結果:{} , 正確結果:{} \t".format(self.infer(x),y),end="")
			
			
	# 		data.look(r1)
	# 		#============================================================
	# 		# 從 test 挑
	# 		r2 = torch.randint(0,n_valid,(1,)).item()
	# 		x = data.xtest[r2].numpy()
	# 		x = x.reshape(1,28,28)
	# 		y = data.ytest[r2].item()
	# 		print("預測結果:{} , 正確結果:{} \t".format(self.infer(x),y),end="")
			
			
	# 		data.look(r2,False)
	# 		_str = input("任何鍵重測 , CTRL+C 可終止程式 !!")

	def run(self): 
		batch_time = utils.AverageMeter()
		losses = utils.AverageMeter()
		psnr = utils.AverageMeter()
		ssim = utils.AverageMeter()
		
		for batch_idx, (raws, imgs, targets, img_files, img_exposures, lbl_exposures, ratios) in tqdm.tqdm(
			enumerate(test_loader), total=len(test_loader),desc='{} iteration={} epoch={}'.format('Test', 
				self.iteration, self.epoch), ncols=80, leave=False):
			
			gc.collect() 
			with torch.no_grad():
				raws = Variable(raws)
				targets = Variable(targets)
				
				output = Variable(torch.Tensor(self.exec_net.infer(inputs={self.input_blob:raws})[self.output_blob]))

				# targets = targets[:, :, :output.size(2), :output.size(3)]
				# loss = self.criterion(output, targets)

				# if np.isnan(float(loss.item())):
				# 	raise ValueError('loss is nan while validating')	

				# losses.update(loss.item(), targets.size(0))

			outputs = torch.clamp(output, 0, 1).cpu()
			targets = targets.cpu()

			for output, img, target, img_file, img_exposure, lbl_exposure, ratio in zip(outputs, imgs, targets,img_files, img_exposures,lbl_exposures, ratios):

				output = output.numpy().transpose(1, 2, 0) * 255
				target = target.numpy().transpose(1, 2, 0) * 255

				os.makedirs(self.result_dir, exist_ok=True)
				fname = os.path.join(self.result_dir, '{}_compare.jpg'.format(os.path.basename(img_file)[:-4]))
				temp = np.concatenate((target[:, :, :], output[:, :, :]), axis=1)
				scipy.misc.toimage(temp, high=255, low=0, cmin=0, cmax=255).save(fname)
				fname = os.path.join(self.result_dir, '{}_single.jpg'.format(os.path.basename(img_file)[:-4]))
				scipy.misc.toimage(output, high=255, low=0, cmin=0, cmax=255).save(fname)

				_psnr = utils.get_psnr(output, target)
				print("PSNR", img_file, _psnr)
				psnr.update(_psnr,1)
				_ssim = utils.get_ssim(output, target)
				print("SSIM", img_file, _ssim)
				ssim.update(_ssim, 1)

#-------------------------------------------------------------------------------------------------------------------
# https://www.youtube.com/watch?v=Nmf-aHeRFq4&list=PLDKCjIU5YH6jMzcTV5_cxX9aPHsborbXQ&index=38&fbclid=IwAR02DybVCQ9KiMbnbmgFTxUM3h6oc54Aa6ed5wBJQGTugwcnH8fWBSeoIyM
#-------------------------------------------------------------------------------------------------------------------

import cv2 as cv
class InferenceEngineOpenCV:
	def __init__(self,batch_size=1):
		xmlfile = ABSPATH + "Sony_resume_test.xml"
		binfile = ABSPATH + "Sony_resume_test.bin"
		self.net = cv.dnn.readNet(xmlfile,binfile)   
		self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
		
	def inferFromFileName(self,imgfile): 
		# read as grayscale
		frame = cv.resize(cv.imread(imgfile,cv.IMREAD_GRAYSCALE),(28,28))
		# blob NCHW
		blob = cv.dnn.blobFromImage(frame)
		print("blob_shape={}".format(blob.shape))
		self.net.setInput(blob)
		out = self.net.forward()
		print(out)





if __name__ == "__main__":
	if sys.argv[1] == "--inferOPENVINO":
		InferenceEngineOpenVINO().run()

	# if sys.argv[1] == "--inferOpenCV":
	# 	InferenceEngineOpenCV.inferFromFileName(sys.argv[2])

	# OriginalInference().run()
