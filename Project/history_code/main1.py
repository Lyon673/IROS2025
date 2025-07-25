import message_filters
import rospy
import random
import numpy as np
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from ambf_msgs.msg import RigidBodyState
from geomagic_control.msg import DeviceButtonEvent
from os.path import join
import atexit
import time
import sys
import os
import struct
import math
from collections import namedtuple
import ctypes
import roslib.message
from sensor_msgs.msg import PointField
from ambf_client import Client
from numpy.linalg import inv
import functools
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray
from collections import deque
import ipa
from std_msgs.msg import Float32  
import datetime
import torch
import threading
import time
from pynput import keyboard
import sys
import platform



# set resolution
resolution_x = 640
resolution_y = 480

stereo_l_img = []
stereo_r_img = []
segment_l_img = []
segment_r_img = []
psm_ghost_pose = []
psm_pose_list = []
psm_twist_list = []
gazepoint_list = []
gaze_list = []

distance_list = []
psm_velocity_list = []
ipaL_data_list = []
ipaR_data_list = []
distance_plane_list = []
pupilL_list = []
pupilR_list = []
scale_list = []

velocity_deque_L = deque(maxlen=8)
velocity_deque_R = deque(maxlen=8)

pupilL = deque(maxlen=128)
pupilR = deque(maxlen=128)
gazePoint = np.array([0.5,0.5])
scale = deque(maxlen=10)
velocity_deque = deque(maxlen=8)
psmPosePre = np.zeros(4)
psmPosePreL = np.zeros(4)
psmPosePreR = np.zeros(4)
flag = False
ipaL_data = 2
ipaR_data = 2
last_button_state = 0
last_button_stateL = 0
last_button_stateR = 0


clutch_times_list = [0]
total_distance_list = [0]
total_time_list = [0]
test_scale = True

_DATATYPES = {}
_DATATYPES[PointField.INT8]	= ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)


start_time = time.time()

init_params = {

	'd_min': 0, 'd_max': 50,
	'p_min': 0, 'p_max': 8,
	'v_min': 0, 'v_max': 8,

	# threshold
	'tau_d': 0.7,  
	'tau_p': 0.8,  
	'tau_v': 0.6,  

	# gains
	'A_d': 1.5,
	'A_p': 4.0,
	'A_v': 8.0,

	# weights
	'Y_base': 0.1,  # base
	'W_d': 1.0,	 
	'W_p': 1.0,	  
	'W_v': 1.0,	 
	'W_dp': 0.5,	 # distance * pupil
	'W_dv': 0.5,	 # distance * velocity
	'W_pv': 0.5,	 # pupil * velocity
	'W_dpv': 0.5,	# distance * pupil * velocity
	'C_offset': 0.05 
}

class DataCollector:
	def __init__(self):
		# Your original initialization code
		self.bridge = CvBridge()
		self.collecting = False
		self.collection_complete = False
		
		# Initialize ROS
		rospy.init_node('main', anonymous=True)
		self.scale_pub = rospy.Publisher('/scale', Float32MultiArray, queue_size=10)
		rospy.Subscriber("/touch_data", Float32MultiArray, self.maincb)

		# --- KEYBOARD LISTENER INTEGRATION START ---
		# In the class's __init__ function, start the keyboard listener
		self.start_keyboard_listener()
		# --- KEYBOARD LISTENER INTEGRATION END ---
		
	# --- KEYBOARD LISTENER INTEGRATION START ---
	# The following three functions are newly added to handle keyboard listening
	def reset_globals(self):
		"""Reset all global variables used for data collection"""
		global psm_ghost_pose, psm_pose_list, psm_twist_list, gazepoint_list, gaze_list
		global distance_list, psm_velocity_list, ipaL_data_list, ipaR_data_list
		global distance_plane_list, pupilL_list, pupilR_list, scale_list
		global velocity_deque_L, velocity_deque_R, pupilL, pupilR
		global gazePoint, scale, velocity_deque
		global psmPosePre, psmPosePreL, psmPosePreR, flag
		global ipaL_data, ipaR_data, last_button_state, last_button_stateL, last_button_stateR
		global clutch_times_list, total_distance_list, total_time_list, start_time
		
		# Reset all lists
		psm_ghost_pose = []
		psm_pose_list = []
		psm_twist_list = []
		gazepoint_list = []
		gaze_list = []
		distance_list = []
		psm_velocity_list = []
		ipaL_data_list = []
		ipaR_data_list = []
		distance_plane_list = []
		pupilL_list = []
		pupilR_list = []
		scale_list = []
		
		# Reset deques
		velocity_deque_L = deque(maxlen=8)
		velocity_deque_R = deque(maxlen=8)
		pupilL = deque(maxlen=128)
		pupilR = deque(maxlen=128)
		scale = deque(maxlen=10)
		velocity_deque = deque(maxlen=8)
		
		# Reset state variables
		psmPosePre = np.zeros(4)
		psmPosePreL = np.zeros(4)
		psmPosePreR = np.zeros(4)
		flag = False
		ipaL_data = 2
		ipaR_data = 2
		last_button_state = 0
		last_button_stateL = 0
		last_button_stateR = 0
		
		# Reset statistical variables
		clutch_times_list = [0]
		total_distance_list = [0]
		total_time_list = [0]
		
		# Reset other variables
		gazePoint = np.array([0.5, 0.5])
		start_time = time.time()
		
		print("全局变量已重置，准备开始新的数据收集...")
	def on_press(self, key):
		"""Callback function for key press"""
		# We set the 's' key (stop) to stop collection
		if hasattr(key, 'char') and key.char == 's':
			if self.collecting:
				print(f"\n检测到停止键 's'，正在停止收集...")
				self.stop_collection()
	
	def listener_thread_target(self):
		"""Target function for the listener thread"""
		# The 'with' statement ensures the listener is properly cleaned up on exit
		with keyboard.Listener(on_press=self.on_press) as listener:
			listener.join()

	def start_keyboard_listener(self):
		"""Create a background thread to run the keyboard listener"""
		listener_thread = threading.Thread(target=self.listener_thread_target, daemon=True)
		listener_thread.start()
	# --- KEYBOARD LISTENER INTEGRATION END ---

	def start_collection(self):
		"""Start data collection"""
		self.collecting = True
		self.collection_complete = False
		print("started collecting PSM 1&2 baselink, stereo image, and segmentation image ...")
		# --- MODIFICATION START ---
		# Modify the prompt message to inform the user of the new key
		print("数据收集中... (按 's' 键停止)")
		# --- MODIFICATION END ---
		
	def stop_collection(self):
		"""Stop data collection"""
		# Ensure messages are printed only when collection is in progress
		if self.collecting:
			print("收集任务已停止。")
			self.collecting = False
			self.collection_complete = True
			
	def wait_for_completion(self):
		"""
		Wait for collection to complete.
		--- MODIFICATION START ---
		This function is now very simple and no longer needs a try/except block.
		It just waits for the self.collecting flag to be changed by the keyboard listener thread.
		--- MODIFICATION END ---
		"""
		while self.collecting and not rospy.is_shutdown():
			rospy.sleep(0.1)

	def maincb(self,data):
		"""
		Main callback function.
		There are several data types in this function.
		1. Get the psm pose in the screen or the 3D simulated environment.
		2. Get the velocity of master.
		3. Get the gaze point and pupil data.

		data[1] represents the manipulatetion mode, 1 for unimanual and 2 for bimanual.
		When it is unimanual:
		The input parameter which named data means:
		data[0] : timestamp (float)
		data[1] : unimanual mode
		data[2~4] : mtm1 velocity
		data[5] : mtm1 angular velocity
		data[6] : clutch data
		data[7~9] : psm1 position in 3D simulated environment
		data[10~11] : psm1 position at 2D screen (not ratio)
		data[12~14] : gaze point in 3D simulated environment

		When it is bimanual:
		data[0] : timestamp (float)
		data[1] : bimanual mode
		data[2~4] : mtm1 velocity
		data[5] : mtm1 angular velocity
		data[6] : mtm1 clutch data
		data[7~9] : mtm2 velocity
		data[10] : mtm2 angular velocity
		data[11] : mtm2 clutch data
		data[12~14] : psm1 position in 3D simulated environment
		data[15~17] : psm2 position in 3D simulated environment
		data[18~19] : psm1 position at 2D screen (not ratio)
		data[20~21] : psm2 position at 2D screen (not ratio)
		data[22~24] : gaze point in 3D simulated environment

		while the gaze data is not included in the data, it is obtained from the subscriber, which is in the global variable such as gazePoint, pupilL.
		"""
		if self.collecting:
			sys.stdout.write('\r-- Time past: %02.1f' % float(time.time() - start_time))
			sys.stdout.flush()
		
		data = list(data.data)
		if data[1] == 1:
			psm_2d, psm_3d_position, gazepoint_2d, distance, mtm_velocity = cal_robot_data(data,velocity_deque_R)
			distance_psms = None
		elif data[1] == 2:
			# change the data to the unimanual data mode, just to use the same calculation func
			dataR = [data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[12],data[13],data[14],data[18],data[19],data[22],data[23],data[24]]
			dataL = [data[0],data[1],data[7],data[8],data[9],data[10],data[11],data[15],data[16],data[17],data[20],data[21],data[22],data[23],data[24]]
			psmL_2d, psmL_3d_position, gazepoint_2d, distanceL, mtmL_velocity = cal_robot_data(dataL,velocity_deque_L)
			psmR_2d, psmR_3d_position, gazepoint_2d, distanceR, mtmR_velocity = cal_robot_data(dataR,velocity_deque_R)
			
			psm_2d = [psmL_2d, psmR_2d]
			psm_3d_position = [psmL_3d_position, psmR_3d_position]
			distance = [distanceL, distanceR]
			mtm_velocity = [mtmL_velocity, mtmR_velocity]

			distance_psms = np.sqrt((psmL_3d_position[0]-psmR_3d_position[0])**2 + (psmL_3d_position[1]-psmR_3d_position[1])**2 + (psmL_3d_position[2]-psmR_3d_position[2])**2)

		else:
			print(f"<LYON> ERROR: Unknown mode {data[1]}")
			return

		# calculate ipa
		global ipaL_data
		global ipaR_data

		if(len(pupilL)>=128 and len(pupilR)>=128):
			_ipaL_data = ipa.ipa_cal(pupilL)
			_ipaR_data = ipa.ipa_cal(pupilR)
			if(_ipaL_data!=0):
				ipaL_data = _ipaL_data
			if(_ipaR_data!=0):
				ipaR_data = _ipaR_data
		else:

			ipaL_data = 0.23
			ipaR_data = 0.23
		

		# calculate and publish scale
		params = self.load_params()
		scale= scale_cal(data[1],distance, mtm_velocity, ipaL_data, ipaR_data, params,distance_psms)
		try:  
			self.scale_pub.publish(scale)
		except rospy.ROSException as e:
			rospy.logwarn(f"Failed to publish scale: {e}")

		scale_list.append(scale.data)

		cal_performance_data(data)
		# calculate total time
		total_time_list[0] = float(time.time() - start_time)

		# print the data
		# to fix: bimanual mode
		if data[1] == 1 and self.collecting:
			print("\n")
			print("=" * 50)
			print(f"{'PSM 3d Position':<25}: [{psm_3d_position[0]:.3f}, {psm_3d_position[1]:.3f}, {psm_3d_position[2]:.3f}]")
			print(f"{'PSM 2d Position':<25}: [{psm_2d[0]:.3f}, {psm_2d[1]:.3f}]")
			print(f"{'Gazing Point ':<25}: [{gazePoint[0]:.3f}, {gazePoint[1]:.3f}]")
			print(f"Gaze Point 2d: {gazepoint_2d[0]:.3f}, {gazepoint_2d[1]:.3f}")
			print(f"{'PSM Gazing Point Distance':<25}: {distance:.3f}")
			print(f"{'MTM Velocity (x, y, z)':<25}: [{mtm_velocity:.3f}]")
			print(f"{'IPA Left Data':<25}: [{ipaL_data:.3f}]")
			print(f"{'IPA Right Data':<25}: [{ipaR_data:.3f}]")
			print(f"{'Scale':<25}: {scale.data[0]:.8f}")
			print("=" * 50)
			print("\n")

		elif data[1] == 2 and self.collecting:
			print("\n")
			print("=" * 50)
			print(f"{'PSM Left 3d Position':<25}: [{psm_3d_position[0][0]:.3f}, {psm_3d_position[0][1]:.3f}, {psm_3d_position[0][2]:.3f}]")
			print(f"{'PSM Right 3d Position':<25}: [{psm_3d_position[1][0]:.3f}, {psm_3d_position[1][1]:.3f}, {psm_3d_position[1][2]:.3f}]")
			print(f"{'PSM Left 2d Position':<25}: [{psm_2d[0][0]:.3f}, {psm_2d[0][1]:.3f}]")
			print(f"{'PSM Right 2d Position':<25}: [{psm_2d[1][0]:.3f}, {psm_2d[1][1]:.3f}]")
			print(f"{'Gazing Point ':<25}: [{gazePoint[0]:.3f}, {gazePoint[1]:.3f}]")
			print(f"Gaze Point 2d: {gazepoint_2d[0]:.3f}, {gazepoint_2d[1]:.3f}")
			print(f"{'PSM Left Gazing Point Distance':<25}: {distance[0]:.3f}")
			print(f"{'PSM Right Gazing Point Distance':<25}: {distance[1]:.3f}")
			print(f"{'PSM Left Velocity (x, y, z)':<25}: [{mtm_velocity[0]:.3f}]")
			print(f"{'PSM Right Velocity (x, y, z)':<25}: [{mtm_velocity[1]:.3f}]")
			print(f"{'IPA Left Data':<25}: [{ipaL_data:.3f}]")
			print(f"{'IPA Right Data':<25}: [{ipaR_data:.3f}]")
			print(f"{'Scale Left':<25}: {scale.data[0]:.8f}")
			print(f"{'Scale Right':<25}: {scale.data[1]:.8f}")
			print("=" * 50)
			print("\n")

		elif self.collecting:
			print(f"<LYON> ERROR: Unknown mode {data[1]}")

		# store the data
		distance_list.append(distance)
		psm_velocity_list.append(mtm_velocity)

		if(ipaL_data != 0):
			ipaL_data_list.append(ipaL_data)
		if(ipaR_data != 0):
			ipaR_data_list.append(ipaR_data)


	def print_statistics(self):
		#print msg container info
		print("\n")
		print(len(psm_ghost_pose))
		print(len(psm_pose_list))
		print(len(psm_twist_list))
		print(len(gazepoint_list))
		print(len(gaze_list))
		print(f"{'Distance List Length':<25}: {len(distance_list)}")
		print(f"{'PSM Velocity List Length':<25}: {len(psm_velocity_list)}")
		print(f"{'IPA Left Data List Length':<25}: {len(ipaL_data_list)}")
		print(f"{'IPA Right Data List Length':<25}: {len(ipaR_data_list)}")
		print(f"{'Pupil Left Data List Length':<25}: {len(pupilL_list)}")
		print(f"{'Pupil Right Data List Length':<25}: {len(pupilR_list)}")
		print(f"{'Distance Plane List Length':<25}: {len(distance_plane_list)}")
		print(f"pupilL_list Length:{len(pupilL)}")
		print(f"pupilL_list Length:{len(pupilR)}")

		print(f"\n{'='*50}")
		print(f"Global factors:")
		print(f"  {'Total time':<12} : {total_time_list[0]:.6f}")
		# to fix: bimanual mode print
		print(f"  {'Total distance':<12} : {total_distance_list[0]:.6f}")
		print(f"  {'Clutch times':<12} : {clutch_times_list[0]:.6f}")
		print(f"{'='*50}")


	def load_params(self):
		"""
		Load the parameters.
		"""
		params = init_params.copy()

		current_file_path = os.path.abspath(__file__)
		current_dir = os.path.dirname(current_file_path)

		filename = os.path.join(current_dir, 'params', 'params.txt')
		# Load parameters from the file if it exists
		try:
			with open(filename, 'r') as f:
				for line in f:
					key, value = line.strip().split('=')
					params[key] = float(value)
		except FileNotFoundError:
			if self.collecting:
				print("params.txt not found, using default parameters.")

		return params

	def run(self):
		"""Main run loop (your function remains unchanged)"""
		while not rospy.is_shutdown():
			# First, reset the data to be used
			self.reset_globals()
			# Start data collection
			self.start_collection()
			self.wait_for_completion()
			self.print_statistics()
			
			# Save data
			save_data_cb()
			flush_input()
			# Ask whether to continue
			try:
				ok = input("是否进行下一次数据收集？(y or n): ")
				if ok.lower() != "y":
					break
			except KeyboardInterrupt:
				# Using Ctrl+C here to exit the entire program is reasonable
				print("\n程序退出")
				break

def flush_input():
	"""Clear the standard input buffer according to the operating system"""
	try:
		# For Linux and macOS
		if platform.system() in ["Linux", "Darwin"]:
			import termios
			termios.tcflush(sys.stdin, termios.TCIFLUSH)
		# For Windows
		elif platform.system() == "Windows":
			import msvcrt
			while msvcrt.kbhit():
				msvcrt.getch()
	except (ImportError, OSError) as e:
		# May fail in some special environments (e.g., non-interactive scripts)
		print(f"清空输入缓冲区时出错 (可以忽略): {e}")


def cal_robot_data(data,velocity_deque):
	# set input data
	psm_2d = np.array([data[10], data[11]])
	psm_3d_position = np.array([data[7], data[8], data[9]])
	mtm_velocity_xyz_angle_time = np.array([data[2],data[3],data[4],data[5],data[0]])


	# TODO: check the timestamp correction for calculating gracefulness in the file <gracefulness.py> 
	psm_3d_pos_time = np.append(psm_3d_position,data[0]*1e-6)
	psm_ghost_pose.append(psm_3d_pos_time)

	psm_twist_list.append(mtm_velocity_xyz_angle_time)	

	gazepoint_2d = [gazePoint[0]*resolution_x, gazePoint[1]*resolution_y]

	"""
	calculate the features
	- distance between psm and gazing point
	- psm velocity
	"""
	distance = get_distance_between_psm_gazingpoint(psm_2d, gazepoint_2d)

	# mtm_velocity = get_mtm_velocity(mtm_velocity_xyz_angle_time)
	mtm_velocity = get_mtm_velocity(mtm_velocity_xyz_angle_time, velocity_deque)
	return psm_2d, psm_3d_position, gazepoint_2d, distance, mtm_velocity


def normalize(value, min_val, max_val, corr):
	if corr == 1:
		normalized_value = np.clip((value - min_val) / (max_val - min_val),0,1)
	else: 
		normalized_value = np.clip((-value + max_val) / (max_val - min_val),0,1)

	return normalized_value


def scale_cal(mode,distance_GP, velocity_psm, IPAL, IPAR,params,distance_psms=None):	

	scaleArray = Float32MultiArray()
	if mode == 1:
		None
		""" segment func """
		# IPA_m = (IPAL + IPAR) / 2
		# N_d = normalize(distance_GP, params['d_min'], params['d_max'],1)
		# N_p = normalize(IPA_m, params['p_min'], params['p_max'],1)
		# N_v = normalize(velocity_psm, params['v_min'], params['v_max'],1)

		# # # (G_x)
		# G_d = params['A_d'] if N_d >= params['tau_d'] else 1.0
		# G_p = params['A_p'] if N_p >= params['tau_p'] else 1.0
		# G_v = params['A_v'] if N_v >= params['tau_v'] else 1.0

		# weighted_sum = (
		#	 params['W_d'] * N_d +
		#	 params['W_p'] * N_p +
		#	 params['W_v'] * N_v +
		#	 params['W_dp'] * N_d * N_p +
		#	 params['W_dv'] * N_d * N_v +
		#	 params['W_pv'] * N_p * N_v +
		#	 params['W_dpv'] * N_d * N_p * N_v +
		#	 params['C_offset']
		# )
		
		# output = params['Y_base'] * G_d * G_p * G_v * weighted_sum
		
		""" tan func"""
		IPA_m = (IPAL + IPAR) / 2
		N_d = normalize(distance_GP, params['d_min'], params['d_max'],1)
		N_p = normalize(IPA_m, params['p_min'], params['p_max'],1)
		N_v = normalize(velocity_psm, params['v_min'], params['v_max'],1)

		T_d = np.tan(0.49* np.pi * params['tau_d'] * N_d)  
		T_p = np.tan(0.49* np.pi * params['tau_p'] * N_p)
		T_v = np.tan(0.49* np.pi * params['tau_v'] * N_v)

		weighted_sum = (
			params['W_d'] * T_d +
			params['W_p'] * T_p +
			params['W_v'] * T_v +
			params['W_dp'] * T_d * T_p +
			params['W_dv'] * T_d * T_v +
			params['W_pv'] * T_p * T_v +
			params['W_dpv'] * T_d * T_p * T_v +
			params['C_offset']
		)

		output = params['Y_base'] * weighted_sum

		scaleArray.data = [np.clip(output,0.1,8)]

	elif mode == 2:

		# to fix: bimanual mode's scale
		#scale = k_s * (np.exp((min(1, (distance_GP)/(d_GP_max)) * min(1, velocity_psm/v_psm_max+0.01) * \
		#	max(0, (ipa_max-IPAL)/ipa_max) * max(0, (ipa_max-IPAR)/ipa_max) )**0.2 *2-1) /2.35*9-0.4089)		

		# the left one is the left touch, the right one is the right touch
		# IPA_m = (IPAL+IPAR)/2
		# N_d_L = normalize(distance_GP[0], params['d_min'], params['d_max'],1)
		# N_d_R = normalize(distance_GP[1], params['d_min'], params['d_max'],1)
		# N_p = normalize(IPA_m, params['p_min'], params['p_max'],1)
		# N_v_L = normalize(velocity_psm[0], params['v_min'], params['v_max'],1)
		# N_v_R = normalize(velocity_psm[1], params['v_min'], params['v_max'],1)
		# N_s = normalize(distance_psms, params['s_min'], params['s_max'],1) 
	
		# # # (G_x)
		# G_d_L = params['A_d'] if N_d_L >= params['tau_d'] else 1.0
		# G_d_R = params['A_d'] if N_d_R >= params['tau_d'] else 1.0
		# G_p = params['A_p'] if N_p >= params['tau_p'] else 1.0
		# G_v_L = params['A_v'] if N_v_L >= params['tau_v'] else 1.0
		# G_v_R = params['A_v'] if N_v_R >= params['tau_v'] else 1.0
		# G_s = params['A_s'] if N_s >= params['tau_s'] else 1.0

		# weighted_sum_L = (
		#	 params['W_d'] * N_d_L +
		#	 params['W_p'] * N_p +
		#	 params['W_v'] * N_v_L +
		#	 params['W_dps'] * N_d_L * N_p * N_s +
		#	 params['W_dvs'] * N_d_L * N_v_L * N_s +
		#	 params['W_pvs'] * N_p * N_v_L * N_s +
		#	 params['W_dpv'] * N_d_L * N_p * N_v_L +
		#	 params['W_dpvs'] * N_d_L * N_p * N_v_L * N_s +
		#	 params['C_offset']
		# )
		
		# output_L = params['Y_base'] * G_d_L * G_p * G_v_L * G_s* weighted_sum_L

		# # right 
		# weighted_sum_R = (
		#	 params['W_d'] * N_d_R +
		#	 params['W_p'] * N_p +
		#	 params['W_v'] * N_v_R +
		#	 params['W_dps'] * N_d_R * N_p * N_s  +
		#	 params['W_dvs'] * N_d_R * N_v_R * N_s  +
		#	 params['W_pvs'] * N_p * N_v_R * N_s  +
		#	 params['W_dpv'] * N_d_R * N_p * N_v_R +
		#	 params['W_dpvs'] * N_d_R * N_p * N_v_R * N_s +
		#	 params['C_offset']
		# )
		
		# output_R = params['Y_base'] * G_d_R * G_p * G_v_R * weighted_sum_R		

		""" tan func"""
		IPA_m = (IPAL+IPAR)/2
		N_d_L = normalize(distance_GP[0], params['d_min'], params['d_max'],1)
		N_d_R = normalize(distance_GP[1], params['d_min'], params['d_max'],1)
		N_p = normalize(IPA_m, params['p_min'], params['p_max'],1)
		N_v_L = normalize(velocity_psm[0], params['v_min'], params['v_max'],1)
		N_v_R = normalize(velocity_psm[1], params['v_min'], params['v_max'],1)
		N_s = normalize(distance_psms, params['s_min'], params['s_max'],1) 

		T_d_L = np.tan(0.49* np.pi * params['tau_d'] * N_d_L)
		T_d_R = np.tan(0.49* np.pi * params['tau_d'] * N_d_R)
		T_p = np.tan(0.49* np.pi * params['tau_p'] * N_p)
		T_v_L = np.tan(0.49* np.pi * params['tau_v'] * N_v_L)
		T_v_R = np.tan(0.49* np.pi * params['tau_v'] * N_v_R)
		T_s = np.tan(0.49* np.pi * params['tau_s'] * N_s)

		weighted_sum_L = (
			params['W_d'] * T_d_L +
			params['W_p'] * T_p +
			params['W_v'] * T_v_L +
			params['W_dps'] * T_d_L * T_p * T_s +
			params['W_dvs'] * T_d_L * T_v_L * T_s +
			params['W_pvs'] * T_p * T_v_L * T_s +
			params['W_dpv'] * T_d_L * T_p * T_v_L +
			params['W_dpvs'] * T_d_L * T_p * T_v_L * T_s +
			params['C_offset']
		)
		
		output_L = params['Y_base'] * weighted_sum_L

		weighted_sum_R = (
			params['W_d'] * T_d_R +
			params['W_p'] * T_p +
			params['W_v'] * T_v_R +
			params['W_dps'] * T_d_R * T_p * T_s  +
			params['W_dvs'] * T_d_R * T_v_R * T_s  +
			params['W_pvs'] * T_p * T_v_R * T_s  +
			params['W_dpv'] * T_d_R * T_p * T_v_R +
			params['W_dpvs'] * T_d_R * T_p * T_v_R * T_s +
			params['C_offset']
		)
		
		output_R = params['Y_base'] * weighted_sum_R

		scaleArray.data = [np.clip(output_L,0.1,8), np.clip(output_R,0.1,8)]

	else:
		return None	
	
	return scaleArray

def cal_performance_data(data):
	"""
	Calculate the performance data.
	- total distance
	- clutch times
	"""
	if data[1] == 1:
		# calculate total distance
		position = np.array([data[7], data[8], data[9]])
		global psmPosePre
		global flag
		if(flag):
			total_distance_list[0] += ((psmPosePre[0]-position[0])**2+(psmPosePre[1]-position[1])**2+(psmPosePre[2]-position[2])**2)**(1/2)
		flag = True
		psmPosePre = position

		# calculate clutch times
		cal_clutch_times([data[6]], data[1])

	elif data[1] == 2:
		# calculate total distance
		position1 = np.array([data[7], data[8], data[9]])
		position2 = np.array([data[12], data[13], data[14]])
		global psmPosePreL
		global psmPosePreR
		if(flag):
			total_distance_list[0] += ((psmPosePreL[0]-position1[0])**2+(psmPosePreL[1]-position1[1])**2+(psmPosePreL[2]-position1[2])**2)**(1/2)
			total_distance_list[0] += ((psmPosePreR[0]-position2[0])**2+(psmPosePreR[1]-position2[1])**2+(psmPosePreR[2]-position2[2])**2)**(1/2)
		flag = True
		psmPosePreL = position1
		psmPosePreR = position2

		# calculate the clutch times
		cal_clutch_times([data[6],data[11]], data[1])
	


"""
The functions for calculating features
"""
def get_distance_between_psm_gazingpoint(psm, gazepoint):
	# calculate the distance between psm and gaze point, may 2d or 3d
	distance = np.sqrt((psm[0]-gazepoint[0])**2 + (psm[1]-gazepoint[1])**2)
	return distance


def get_mtm_velocity(velocity, deque_instance):
	"""
	Calculates the smoothed velocity of the master using a provided deque instance.
	"""
	# calculate the velocity of master with window filter
	mtm_velocity = 0
	deque_instance.append(np.sqrt(velocity[0]**2+velocity[1]**2+velocity[2]**2))
	if not deque_instance:
		return 0.0
	for i in range(len(deque_instance)):
		mtm_velocity += deque_instance[i]
	mtm_velocity = mtm_velocity/len(deque_instance)
	return mtm_velocity

def cal_clutch_times(button_state,mode):
	"""
	It's known that the button state is a discrete value.
	When none of buttons is pressed, the clutch state is 0.
	When the grasp button is pressed, the clutch state is 1.
	When the clutch is pressed, the clutch state is 2.
	When both of the buttons are pressed, the clutch state is 3.
	
	This function uses the edge trigger to calculate the clutch times.
	"""

	if mode == 1:
		global last_button_state
		if last_button_state != 2 and button_state[0] == 2:
			clutch_times_list[0] += 1
		last_button_state = button_state[0]
	elif mode == 2:
		global last_button_stateL
		global last_button_stateR
		if last_button_stateL != 2 and button_state[0] == 2:
			clutch_times_list[0] += 1
		if last_button_stateR != 2 and button_state[1] == 2:
			clutch_times_list[1] += 1
		last_button_stateL = button_state[0]
		last_button_stateR = button_state[1]


"""
if left gaze point and right are both available, then set gaze point as their medium point.If only one, set as that.If none, no change.
"""
def gaze_data_cb(gaze):
	w = resolution_x
	h = resolution_y

	gazedata = list(gaze.data)
	gazedata[1] = np.clip(gazedata[1],0,1)
	gazedata[2] = np.clip(gazedata[2],0,1)
	gazedata[4] = np.clip(gazedata[4],0,1)
	gazedata[5] = np.clip(gazedata[5],0,1)


	if (gazedata[8]==pupilL[-1].get_timestamp() if pupilL else 0):
		return
	if(gazedata[0] == True):
		if(gazedata[3] == True):
			gazePoint[0] = np.round(((gazedata[1]+gazedata[4])/2)*w).astype(int)
			gazePoint[1] = np.round(((gazedata[2]+gazedata[5])/2)*h).astype(int)
		else:
			gazePoint[0] = np.round((gazedata[1])*w).astype(int)
			gazePoint[1] = np.round((gazedata[2])*h).astype(int)
	elif(gazedata[3] == True):
		gazePoint[0] = np.round((gazedata[4])*w).astype(int)
		gazePoint[1] = np.round((gazedata[5])*h).astype(int)
	if (gazedata[6] != 0):
		pupilL.append(ipa.Pupil(gazedata[6], gazedata[8]*1e-6))
		pupilL_list.append(ipa.Pupil(gazedata[6], gazedata[8]*1e-6))
	else:
		last_pupilL = pupilL[-1] if pupilL else None
		pupilL.append(last_pupilL)
		pupilL_list.append(last_pupilL)
	if (gazedata[7] != 0):
		pupilR.append(ipa.Pupil(gazedata[7], gazedata[8]*1e-6))
		pupilR_list.append(ipa.Pupil(gazedata[7], gazedata[8]*1e-6))
	else:
		last_pupilR = pupilR[-1] if pupilR else None
		pupilR.append(last_pupilR)
		pupilR_list.append(last_pupilR)
	gaze_list.append(gaze)
	return gaze.data


"""
Utility functions
"""

# change pose.position to numpy array
def position2numpy(pose):
	return np.array([pose.position.x,pose.position.y,pose.position.z])


# change pose.orientation to numpy array
def orientation2numpy(pose):
	return np.array([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])


def get_transformation_matrix(position, quaternion):
   
	
	r = R.from_quat(quaternion)  
	rotation_matrix = r.as_matrix()  

	
	transformation_matrix = np.eye(4)  
	transformation_matrix[:3, :3] = rotation_matrix  
	transformation_matrix[:3, 3] = position  

	return transformation_matrix


def save_data_cb():
	"""
	data storage callback, invoked when script terminates
	"""
	if not os.path.exists('./data'):
		os.mkdir('./data')
	if not os.path.exists('./data/ipa_data'):
		os.mkdir('./data/ipa_data')
	print("saving data...")
	np.save(join('./data', 'psm_ghost_pose.npy'), psm_ghost_pose)
	np.save(join('./data', 'psm_pose.npy'), psm_pose_list)
	np.save(join('./data', 'psm_twist.npy'), psm_twist_list)
	np.save(join('./data', 'gazepoint_data.npy'), gazepoint_list)
	np.save(join('./data', 'gaze_data.npy'), gaze_list)
	  
	np.save(join('./data', 'distance_data.npy'), distance_list)
	np.save(join('./data', 'psm_velocity_data.npy'), psm_velocity_list)
	np.save(join('./data', 'ipaL_data.npy'), ipaL_data_list)
	np.save(join('./data', 'ipaR_data.npy'), ipaR_data_list)
	np.save(join('./data', 'distance_plane_data.npy'), distance_plane_list)
	  
	np.save(join('./data', 'clutch_times.npy'), clutch_times_list)
	np.save(join('./data', 'total_distance.npy'), total_distance_list)
	np.save(join('./data', 'total_time.npy'), total_time_list)

	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	np.save(join('./data', 'pupilL_data.npy'), pupilL_list)
	np.save(join('./data', 'pupilR_data.npy'), pupilR_list)
	np.save(join('./data/ipa_data', f'pupilL_data_{timestamp}.npy'), pupilL_list)
	np.save(join('./data/ipa_data', f'pupilR_data_{timestamp}.npy'), pupilR_list)
	np.save(join('./data', 'scale_data.npy'), scale_list)
	print("done saving...")


if __name__ == '__main__':
	print("开始进行数据收集")
	letter = input("是否开始数据收集（y or n）: ")
	
	if letter == "y":
		collector = DataCollector()
		collector.run()
	else:
		print("quit")