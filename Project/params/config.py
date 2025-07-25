feature_bound = {
    'd_min': 10, 'd_max': 70,
	'p_min': 0.95, 'p_max': 1,
	'v_min': 0, 'v_max': 1.2,
    's_min': 0, 's_max': 50,
    'C_offset': 0.05
}

oneHanded_range = { 
    # threshold
    'tau_d': (0.7, 1),
    'tau_p': (0.7, 1),
    'tau_v': (0.7, 1),

    # weights
    'Y_base': (0.01, 0.12),  # base
    'W_d': (0.1, 1),	 
    'W_p': (0.1, 1),	   
    'W_v': (0.1, 1),	 
    'W_dp': (0.1, 2),	   # distance * pupil
    'W_dv': (0.1, 2),	  # distance * velocity
    'W_pv': (0.1, 2),	   # pupil * velocity
    #'W_dpv': (0.1, 1),	 # distance * pupil * velocity
}

twoHanded_range = { 
    # threshold
    'tau_d': (0.7, 1),
    'tau_p': (0.7, 1),
    'tau_v': (0.7, 1),
    'tau_s': (0.7, 1),  

    # gains
    # 'A_d': (1, 3),
    # 'A_p': (1, 10),
    # 'A_v': (1, 10),
    # 'A_s': (1, 10),

    # weights
    'Y_base': (0.01, 0.15),  # base
    'W_d': (0.1, 5),	 
    'W_p': (0.1, 5),	   
    'W_v': (0.1, 3),	
    'W_s': (0.1, 5),
    'W_dps': (0.1, 1.5),	  
    'W_dvs': (0.1, 1.5),	 
    'W_pvs': (0.1, 1.5),	 
    'W_dpv': (0.1, 1.5),
    'W_dpvs': (0.1, 1.5),   
}

scoreParams_bound = {
    'gracefulness_max': 2.5,
    'gracefulness_min': 1.5,
	'smoothness_max': 8.5,
    'smoothness_min': 5.5,
    'clutch_times_max': 10,
    'total_distance_max': 12,
    'total_time_max': 120
}

logname ='logs.log.json'

scorefilename = 'zhai_score.json'

exflag = 1

adaptive = {"W_d": 0.6557110261229989, "W_dp": 1.5489982389822312, "W_dv": 1.661834391297523, "W_p": 1.0, "W_pv": 1.340468616420758, "W_v": 0.4538660096723739, "Y_base": 0.03331922812540701, "tau_d": 0.7, "tau_p": 0.7, "tau_v": 1.0}

fixed = {"fixed_scale": 0.35, "AFflag": 1}



