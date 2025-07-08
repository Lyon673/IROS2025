feature_bound = {
    'd_min': 0, 'd_max': 70,
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

    # gains
    # 'A_d': (1, 3),
    # 'A_p': (1, 10),
    # 'A_v': (1, 10),

    # weights
    'Y_base': (0.01, 0.1),  # base
    'W_d': (0.1, 5),	 
    'W_p': (0.1, 5),	   
    'W_v': (0.1, 3),	 
    'W_dp': (0.1, 1.5),	   # distance * pupil
    'W_dv': (0.1, 1.5),	  # distance * velocity
    'W_pv': (0.1, 1.5),	   # pupil * velocity
    'W_dpv': (0.1, 1.5),	 # distance * pupil * velocity
}

twoHanded_range = { 
    # threshold
    'tau_d': (0.4, 1),
    'tau_p': (0.4, 1),
    'tau_v': (0.4, 1),
    'tau_s': (0.4, 1),  

    # gains
    # 'A_d': (1, 3),
    # 'A_p': (1, 10),
    # 'A_v': (1, 10),
    # 'A_s': (1, 10),

    # weights
    'Y_base': (0.01, 0.5),  # base
    'W_d': (0.1, 10),	 
    'W_p': (0.1, 10),	   
    'W_v': (0.1, 10),	 
    'W_dps': (0.1, 10),	  
    'W_dvs': (0.1, 10),	 
    'W_pvs': (0.1, 10),	 
    'W_dpv': (0.1, 10),
    'W_dpvs': (0.1, 10),   
}

scoreParams_bound = {
    'gracefulness_max': 3.5,
	'smoothness_max': 5.5,
    'clutch_times_max': 10,
    'total_distance_max': 12,
    'total_time_max': 120
}

logname = 'chai.json'


