import math
import os
import argparse
import torch
import json
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import imutils
import time
import csv
import mediapipe as mp
import pickle
from numpy.linalg import norm
import datetime
import imageio
from skimage.transform import resize
from tqdm import tqdm
from PIL import Image
import ffmpeg
import subprocess
LOGURU_FFMPEG_LOGLEVELS = {
	"trace": "trace",
	"debug": "debug",
	"info": "info",
	"success": "info",
	"warning": "warning",
	"error": "error",
	"critical": "fatal",
}

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

FACEMESH_lips_1 = [164, 393, 391, 322, 410, 432, 422, 424, 418, 421, 200, 201,
					   194, 204, 202, 212, 186, 92, 165, 167]
FACEMESH_lips_2 = [326, 426, 436, 434, 430, 431, 262, 428, 199, 208, 32, 211, 210,
					   214, 216, 206, 97]


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--path_video', help='Path to input video')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
					type=str, help='Trained state_dict file path to open')
parser.add_argument('--output_path', default='result_pipeline.mp4', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='FDDB', type=str, choices=['FDDB'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.9, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--output_json', default="./data_pose.json",help='Path to save JSON data file')

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
	ckpt_keys = set(pretrained_state_dict.keys())
	model_keys = set(model.state_dict().keys())
	used_pretrained_keys = model_keys & ckpt_keys
	unused_pretrained_keys = ckpt_keys - model_keys
	missing_keys = model_keys - ckpt_keys
	print('Missing keys:{}'.format(len(missing_keys)))
	print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
	print('Used keys:{}'.format(len(used_pretrained_keys)))
	assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
	return True


def remove_prefix(state_dict, prefix):
	''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
	print('remove prefix \'{}\''.format(prefix))
	f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
	return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
	print('Loading pretrained model from {}'.format(pretrained_path), load_to_cpu)
	if load_to_cpu:
		pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
	else:
		device = torch.cuda.current_device()
		pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
	if "state_dict" in pretrained_dict.keys():
		pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
	else:
		pretrained_dict = remove_prefix(pretrained_dict, 'module.')
	check_keys(model, pretrained_dict)
	model.load_state_dict(pretrained_dict, strict=False)
	return model


def predict_retinaface(model, cfg, img, scale, im_height, im_width, device):
	loc, conf, landms = model(img)  # forward pass
	priorbox = PriorBox(cfg, image_size=(im_height, im_width))
	priors = priorbox.forward()
	priors = priors.to(device)
	# print(device)
	prior_data = priors.data
	boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
	boxes = boxes * scale
	boxes = boxes.cpu().numpy()
	scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
	landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
	scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
							img.shape[3], img.shape[2], img.shape[3], img.shape[2],
							img.shape[3], img.shape[2]])
	scale1 = scale1.to(device)
	landms = landms * scale1
	landms = landms.cpu().numpy()

	# ignore low scores
	inds = np.where(scores > args.confidence_threshold)[0]
	boxes = boxes[inds]
	landms = landms[inds]
	scores = scores[inds]

	# keep top-K before NMS
	# order = scores.argsort()[::-1][:args.top_k]
	order = scores.argsort()[::-1]
	boxes = boxes[order]
	landms = landms[order]
	scores = scores[order]

	# do NMS
	dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
	keep = py_cpu_nms(dets, args.nms_threshold)

	dets = dets[keep, :]
	landms = landms[keep]

	dets = np.concatenate((dets, landms), axis=1)
	return dets


def detection_face(mobile_net,resnet_net, img, device):
	H,W,_ = img.shape
	# print(img.shape)
	padding_size_ratio = 0.5
	detected_face = False
	img = imutils.resize(img, width = 640)
	H_resize, W_resize, _ = img.shape
	img_draw = img.copy()
	img = np.float32(img)
	im_height, im_width, _ = img.shape
	# print(img.shape)
	#Resize, normalize and preprocess
	scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
	img -= (104, 117, 123)
	img = img.transpose(2, 0, 1)
	img = torch.from_numpy(img).unsqueeze(0)
	img = img.to(device)
	scale = scale.to(device)

	dets = predict_retinaface(mobile_net, cfg_mnet, img, scale, im_height, im_width, device)
	if dets.shape[0] == 0:
		dets = predict_retinaface(resnet_net, cfg_re50, img, scale, im_height, im_width, device)
	l_coordinate = []

	for k in range(dets.shape[0]):
		detected_face = True
		xmin = int(dets[k, 0])
		ymin = int(dets[k, 1])
		xmax = int(dets[k, 2])
		ymax = int(dets[k, 3])
		bbox = ((xmin, ymin , xmax, ymax))
		topleft = (int(bbox[0]), int(bbox[1]))
		bottomright = (int(bbox[2]), int(bbox[3]))
		padding_X = int((bottomright[0] - topleft[0]) * padding_size_ratio)
		padding_Y = int((bottomright[1] - topleft[1]) * padding_size_ratio)
		padding_topleft = (max(0, topleft[0] - padding_X), max(0, topleft[1]- padding_Y))
		padding_bottomright = (min(W, bottomright[0] + padding_X), min(H, bottomright[1] + padding_Y))
		coordinate = (padding_topleft, padding_bottomright)
		l_coordinate.append(coordinate)
	# print("before scale ", l_coordinate)
	truth_face_coordinate = []
	highest_ycenter_bottomright = 0
	index_truth_face = -1
	for index, coordinate in enumerate(l_coordinate):
		scale_ratio = W/W_resize
		scale_topleft = (int(coordinate[0][0] * scale_ratio), int(coordinate[0][1] * scale_ratio))
		scale_bottomright = (min(int(coordinate[1][0] * scale_ratio), W), min(int(coordinate[1][1] * scale_ratio),H))
		y_center = (scale_topleft[1] + scale_bottomright[1])/2
		if y_center > highest_ycenter_bottomright:
			highest_ycenter_bottomright = y_center
			# index_truth_face = index
			truth_face_coordinate = [(scale_topleft, scale_bottomright)]

	# print(truth_face_coordinate)
	return truth_face_coordinate, detected_face


def facial_landmark_detection(face_mesh, image_in):
	image = image_in.copy()
	image.flags.writeable = False
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	results = face_mesh.process(image)

	# Draw the face mesh annotations on the image.
	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	detected = False
	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			detected = True
			# mp_drawing.draw_landmarks(
			# 	image=image,
			# 	landmark_list=face_landmarks,
			# 	connections=mp_face_mesh.FACEMESH_TESSELATION,
			# 	landmark_drawing_spec=None,
			# 	connection_drawing_spec=mp_drawing_styles
			# 	.get_default_face_mesh_tesselation_style())
			# mp_drawing.draw_landmarks(
			# 	image=image,
			# 	landmark_list=face_landmarks,
			# 	connections=mp_face_mesh.FACEMESH_CONTOURS,
			# 	landmark_drawing_spec=None,
			# 	connection_drawing_spec=mp_drawing_styles
			# 	.get_default_face_mesh_contours_style())
			# mp_drawing.draw_landmarks(
			# 	image=image,
			# 	landmark_list=face_landmarks,
			# 	connections=mp_face_mesh.FACEMESH_IRISES,
			# 	landmark_drawing_spec=None,
			# 	connection_drawing_spec=mp_drawing_styles
			# 	.get_default_face_mesh_iris_connections_style())
	return image, detected

FACEMESH_pose_estimation = [34,264,168,33, 263]

def point_point(point_1,point_2):
	x1 = point_1[0]
	y1 = point_1[1]
	x2 = point_2[0]
	y2 = point_2[1]
	# distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
	distance = math.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)
	# distance = math.hypot(x2-x2, y1-y2)
	# if distance == 0:
	# 	distance = distance + 0.1
	return distance

def point_line(point,line):
	x1 = line[0]
	y1 = line[1]
	x2 = line[2]
	y2 = line[3]

	x3 = point[0]
	y3 = point[1]

	k1 = (y2 - y1)*1.0 /(x2 -x1)
	b1 = y1 *1.0 - x1 *k1 *1.0
	k2 = -1.0/k1
	b2 = y3 *1.0 -x3 * k2 *1.0
	x = (b2 - b1) * 1.0 /(k1 - k2)
	y = k1 * x *1.0 +b1 *1.0
	return [x,y]

def facePose(point1, point31, point51, point60, point72, list_source):
	bestidx = 0

	crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
	yaw_mean = point_point(point1, point31) / 2
	yaw_right = point_point(point1, crossover51)
	yaw = (yaw_mean - yaw_right) / yaw_mean
	if math.isnan(yaw):
		return  None, None, None
	yaw = int(yaw * 71.58 + 0.7037)

	# if yaw >= 45:
	# 	bestidx = 1
	# elif yaw <= -45:
	# 	bestidx = 2
	#pitch
	pitch_dis = point_point(point51, crossover51)
	if point51[1] < crossover51[1]:
		pitch_dis = -pitch_dis
	if math.isnan(pitch_dis):
		return bestidx# None, None, None
	pitch = int(1.497 * pitch_dis + 18.97)
	
	#roll
	roll_tan = abs(point60[1] - point72[1]) / abs(point60[0] - point72[0])
	roll = math.atan(roll_tan)
	roll = math.degrees(roll)
	if math.isnan(roll):
		return bestidx# None, None, None
	if point60[1] > point72[1]:
		roll = -roll
	roll = int(roll)
	# cosine_similarity = [np.dot(driving_ypr,value)/(norm(driving_ypr)*norm(value)) for value in list_source]
	# # print(cosine_similarity)
	# bestidx = cosine_similarity.index(max(cosine_similarity))

	return [yaw, pitch, roll]


def ffmpeg_encoder(outfile, fps, width, height):
	frames = ffmpeg.input(
		"pipe:0",
		format="rawvideo",
		pix_fmt="rgb24",
		vsync="1",
		s='{}x{}'.format(width, height),
		r=fps,
	)

	encoder_ = subprocess.Popen(
		ffmpeg.compile(
			ffmpeg.output(
				frames,
				outfile,
				pix_fmt="yuv420p",
				vcodec="libx264",
				acodec="copy",
				r=fps,
				crf=17,
				vsync="1",
			)
			.global_args("-hide_banner")
			.global_args("-nostats")
			.global_args(
				"-loglevel",
				LOGURU_FFMPEG_LOGLEVELS.get(
					os.environ.get("LOGURU_LEVEL", "INFO").lower()
				),
			),
			overwrite_output=True,
		),
		stdin=subprocess.PIPE,
		# stdout=subprocess.DEVNULL,
		# stderr=subprocess.DEVNULL,
	)
	return encoder_

# def loadmodelface():
# 	torch.set_grad_enabled(False)
# 	# face_mesh = mp_face_mesh.FaceMesh(
# 	# 						max_num_faces=1,
# 	# 						refine_landmarks=True,
# 	# 						min_detection_confidence=0.5,
# 	# 						min_tracking_confidence=0.5)
# 	if torch.cuda.is_available():
# 		device = torch.device("cuda")
# 	else:
# 		device = torch.device("cpu")
# 	print("DEVICE",device)
# 	mobile_net = RetinaFace(cfg=cfg_mnet, phase = 'test')
# 	mobile_net = load_model(mobile_net, "./weights/mobilenet0.25_Final.pth", args.cpu)
# 	mobile_net.eval()
# 	print('Finished loading model!')
# 	cudnn.benchmark = True

# 	mobile_net = mobile_net.to(device)

# 	resnet_net = RetinaFace(cfg=cfg_re50, phase = 'test')
# 	resnet_net = load_model(resnet_net, "./weights/Resnet50_Final.pth", args.cpu)
# 	resnet_net.eval()
# 	resnet_net = resnet_net.to(device)

# 	return mobile_net, resnet_net
 
font = cv2.FONT_HERSHEY_SIMPLEX

def get_concat_h(im1, im2):
	dst = Image.new('RGB', (im1.width + im2.width, im1.height))
	dst.paste(im1, (0, 0))
	dst.paste(im2, (im1.width, 0))
	# print(dst.size)
	return dst
def get_concat_v(im1, im2):
	dst = Image.new('RGB', (im1.width, im1.height + im2.height))
	dst.paste(im1, (0, 0))
	dst.paste(im2, (0, im1.height))
	return dst

def save_data(path_data,data_tmp):
	global data_frame_tmp
	with open(save_path,"a") as f:
		json.dump(data_tmp,f)
	data_frame_tmp = {}
	

if __name__ == '__main__':
	torch.set_grad_enabled(False)
	save_path = args.output_json
	
 
 
	face_mesh = mp_face_mesh.FaceMesh(
							max_num_faces=1,
							refine_landmarks=True,
							min_detection_confidence=0.5,
							min_tracking_confidence=0.5)
	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")
	print("DEVICE",device == torch.device("cpu"))
	mobile_net = RetinaFace(cfg=cfg_mnet, phase = 'test')
	mobile_net = load_model(mobile_net, "./weights/mobilenet0.25_Final.pth", device == torch.device("cpu"))
	mobile_net.eval()
	print('Finished loading model!')
	cudnn.benchmark = True
 
	# device = torch.device("cuda")
	mobile_net = mobile_net.to(device)

	resnet_net = RetinaFace(cfg=cfg_re50, phase = 'test')
	resnet_net = load_model(resnet_net, "./weights/Resnet50_Final.pth", device == torch.device("cpu"))
	resnet_net.eval()
	resnet_net = resnet_net.to(device)

	dim = (256, 256)
	#
	source_image_euler_angles = [(1, 0, 0), (45, 0, 0), (-45, 0, 0)]

	# save file
	if not os.path.exists(args.save_folder):
		os.makedirs(args.save_folder)
	cap = cv2.VideoCapture(args.path_video)
	# cap = cv2.VideoCapture(0)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	fps = cap.get(cv2.CAP_PROP_FPS)

	size = (frame_width, frame_height)

	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	count_frame = 0
	# count_detected = 0
	# count_landmark = 0
	FRAME_SEQ_LEN = int(fps/2)
	print('FRAME_SEQ_LEN ', FRAME_SEQ_LEN)

	pbar = tqdm(total=length)
	previous_facePose = []
	previous_angle = []
	data_frame_tmp = None
	frame_count = 0
	while cap.isOpened():

		success, image = cap.read()
		frame_count += 1
		if not success :#or frame_count == 21:
			print("Ignoring empty camera frame.")
			save_data(save_path,data_frame_tmp)
			break
		if frame_count // 10 == 0:
			save_data(save_path,data_frame_tmp)
		pbar.update(1)
		most_frequent_value = 0

		l_coordinate, detected_face = detection_face(mobile_net,resnet_net, image, device)
		# o_frame = image.copy()
		if not detected_face:
			print("Not detect face")
			data_frame_tmp.update({'Frame_' + format(frame_count, '06d'):{
									"angle":[],
									"landmarks":[]}})
		else:
			coordinate = l_coordinate[0]
			topleft, bottomright = coordinate
			# print(coordinate)
			crop_image = image[topleft[1]:bottomright[1], topleft[0]:bottomright[0],:]
			# crop_image.flags.writeable = False
			# cv2.imwrite("/home/anlab/Downloads/test_tmp.jpg",crop_image)
			results = face_mesh.process(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
			# print(results.multi_face_landmarks[0])
			# break
			if not results.multi_face_landmarks:
				print("Not detect multi_face_landmarks")
				data_frame_tmp.update({'Frame_' + format(frame_count, '06d'):{
									"angle":[],
									"landmarks":[]}})
				continue
			face_landmarks = results.multi_face_landmarks[0]
			bbox = []
			bbox.append(topleft[0])
			bbox.append(topleft[1])
			bbox.append(bottomright[0])
			bbox.append(bottomright[1])

			if not face_landmarks:
				print("Not detect landmarks")
				data_frame_tmp.update({'Frame_' + format(frame_count, '06d'):{
									"angle":[],
									"landmarks":[]}})
				continue
			# print('bbox ', bbox)
			bbox_w = bottomright[0] - topleft[0]
			bbox_h = bottomright[1] - topleft[1]
			posePoint = []
			data_pose = []
			# Get 468 landmarks
			for i in range(468):
				idx = i
				x = face_landmarks.landmark[idx].x
				y = face_landmarks.landmark[idx].y
				realx = x * bbox_w
				realy = y * bbox_h
				data_pose.append([realx, realy])

			for i in range(len(FACEMESH_pose_estimation)):
				idx = FACEMESH_pose_estimation[i]
				x = face_landmarks.landmark[idx].x
				y = face_landmarks.landmark[idx].y
				realx = x * bbox_w
				realy = y * bbox_h
				posePoint.append((realx, realy))
			curid = facePose(posePoint[0], posePoint[1], posePoint[2], posePoint[3], posePoint[4], source_image_euler_angles)
			data_frame_tmp.update({'Frame_' + format(frame_count, '06d'):{
									"angle":curid,
									"landmarks":data_pose}})
			previous_facePose = data_pose
			previous_angle = curid

	# if cv2.waitKey(5) & 0xFF == 27:
		# 	break
	cap.release()
	pbar.close()
	# encoder.stdin.flush()
	# encoder.stdin.close()
