"""
factors: distance_GP, velocity_psm, IPAL, IPAR
"""

from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from gracefulness import cal_GS
import numpy as np
from bayes_opt import UtilityFunction
import os
import sys


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

filename = os.path.join(current_dir, 'params', 'params.txt')

mode =1

def save_params_to_txt(dic):
    
    with open(filename, 'w') as f:
        for key, value in dic.items():
            f.write(f"{key}={value}\n")
    
    print(f"Parameters saved to {filename}")

def main():
    # Bounded region of parameter space
    # pbounds = {'k_s': (1, 8),  'v_psm_max': (50,300),  'd_plane_min':(0,0.003), 'd_plane_max':(0.01,0.08)}
    # pbounds = {'k_s': (1, 8), 'd_GP_max': (0.01,0.15), 'v_psm_max': (50,300), 'ipa_max':(0.2,0.5), 'd_plane_min':(0,0.003), 'd_plane_max':(0.01,0.08)}



    global mode
    if mode ==1:
    # Set parameter bounds
        pbounds = { 
            # threshold
            'tau_d': (0.4, 1),
            'tau_p': (0.4, 1),
            'tau_v': (0.4, 1),

            # gains
            # 'A_d': (1, 3),
            # 'A_p': (1, 10),
            # 'A_v': (1, 10),

            # weights
            'Y_base': (0.01, 0.5),  # base
            'W_d': (0.1, 10),	 
            'W_p': (0.1, 10),	   
            'W_v': (0.1, 10),	 
            'W_dp': (0.1, 10),	   # distance * pupil
            'W_dv': (0.1, 10),	  # distance * velocity
            'W_pv': (0.1, 10),	   # pupil * velocity
            'W_dpv': (0.1, 10),	 # distance * pupil * velocity
        }
    else:
        if mode != 2:
            print(f"<LYON> ERROR: mode {mode} is not supported, automatically enter mode 2")
            pbounds = { 
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

    # Initialize acquisition function and optimizer
    utility = UtilityFunction(kind="ei", xi=0.0)

    #bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.5)

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
        #bounds_transformer=bounds_transformer
    )

    # Set optimization parameters
    iter_times = 10

    # Define maximum values for scoring
    gratefulness_max = 3.5
    smoothness_max = 5.5
    clutch_times_max = 10
    total_distance_max = 2.5
    total_time_max = 120

    # Main optimization loop
    for i in range(iter_times):
        # Get next point to evaluate
        next_point = optimizer.suggest(utility)
        print(f"\n{'='*50}")
        print(f"Iteration {i+1}/{iter_times}")
        print(f"Next point to probe is:")
        for key, value in next_point.items():
            print(f"    {key:<12} = {value:.6f}")

        save_params_to_txt(next_point)
        # Get user input for subjective score
        sub_score = float(input("subject_score:"))

        # Load performance metrics
        gratefulness, smoothness = cal_GS()
        clutch_times = np.load('./data/clutch_times.npy', allow_pickle=True)[0]
        total_distance = np.load('./data/total_distance.npy', allow_pickle=True)[0]
        total_time = np.load('./data/total_time.npy', allow_pickle=True)[0]

        # Calculate composite score
        score = 0.2 * sub_score + (5 * np.clip((gratefulness_max-gratefulness)/(gratefulness_max-2),0,1) + 5 * np.clip((smoothness_max-smoothness)/(smoothness_max-4),0,1) \
                + 20 * np.clip((clutch_times_max-clutch_times+1)/clutch_times_max,0,1) + 10 * np.clip((total_distance_max-total_distance)/total_distance_max,0,1)\
                + 10 * np.clip((total_time_max-total_time)/total_time_max,0,1))*1.6
        
        # Register the result
        optimizer.register(params=next_point, target=score)

        # Print metrics and scores
        print(f"Gratefulness: {gratefulness:<12}")
        print(f"Smoothness {smoothness:<12}")
        print(f"Clutch_Times: {clutch_times:<12}")
        print(f"Total_Distance: {total_distance:<12}")
        print(f"Total_Time: {total_time:<12}")

        print(f"Gratefulness_Score: {1.6 * 5 * np.clip((gratefulness_max-gratefulness)/(gratefulness_max-2),0,1):<12}")
        print(f"Smoothness_Score: {1.6 * 5 * np.clip((smoothness_max-smoothness)/(smoothness_max-4),0,1):<12}")
        print(f"Clutch_Times_Score: {1.6 *20 * np.clip((clutch_times_max-clutch_times+1)/clutch_times_max,0,1):<12}")
        print(f"Total_Distance_Score: {1.6 *10 * np.clip((total_distance_max-total_distance)/total_distance_max,0,1):<12}")
        print(f"Total_Time_Score: {1.6 *10 * np.clip((total_time_max-total_time)/total_time_max,0,1):<12}")
        print(f"Score: {score:<12}")
        print(f"{'='*50}")

    # Print final optimization results
    print(f"\n{'*'*50}")
    print("Optimization completed!")
    print("Best result found:")
    for key, value in optimizer.max['params'].items():
        print(f"  {key:<12} : {value:.6f}")
    print(f"  Best score   : {optimizer.max['target']:.6f}")
    print(f"{'*'*50}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) > 0 :
        mode = float(args[0])
    main()


