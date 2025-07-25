import os
import params.config as config
import json
import numpy as np

adaptive_params = config.adaptive_params
fixed_params = config.fixed_params
runLoop = config.runLoop


def save_scores():
    
    # Call the original performance calculation function
    gracefulness, smoothness = cal_GS()
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    clutch_times = np.load(os.path.join(current_dir, 'data', 'clutch_times.npy'), allow_pickle=True)[0]
    total_distance = np.load(os.path.join(current_dir, 'data', 'total_distance.npy'), allow_pickle=True)[0]
    total_time = np.load(os.path.join(current_dir, 'data', 'total_time.npy'), allow_pickle=True)[0]
    
    # Calculate individual scores
    gracefulness_score =  5 * np.clip((gracefulness_max - gracefulness) / (gracefulness_max - gracefulness_min), 0, 1)
    smoothness_score = 5 * np.clip((smoothness_max - smoothness) / (smoothness_max - smoothness_min), 0, 1)
    clutch_times_score = 15 * np.clip((clutch_times_max - clutch_times + 1) / clutch_times_max, 0, 1)
    total_distance_score = 15 * np.clip((total_distance_max - total_distance) / total_distance_max, 0, 1)
    total_time_score = 10 * np.clip((total_time_max - total_time) / total_time_max, 0, 1)
    
    # Calculate the total score
    total_score = 0.5 * sub_score + gracefulness_score + smoothness_score + clutch_times_score + total_distance_score + total_time_score

    output_json_path = os.path.join(current_dir, 'BayesianLog','scores.json')

    new_entry = {
        "subscore": float(sub_score),
        "gracefulness": float(gracefulness),
        "smoothness": float(smoothness),
        "clutch_times": float(clutch_times),
        "total_distance": float(total_distance),
        "total_time": float(total_time),
        "gracefulness_score": float(gracefulness_score),
        "smoothness_score": float(smoothness_score),
        "clutch_times_score": float(clutch_times_score),
        "total_distance_score": float(total_distance_score),
        "total_time_score": float(total_time_score),
        "total_score": float(total_score),
    }
    
    existing_data = []
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            
            existing_data = []
        except Exception as e:
            print(f"JSON error: {e}")
            existing_data = []

    existing_data.append(new_entry)

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        print(f"scores are saved in : {output_json_path}")
    except Exception as e:
        print(f"JSON error: {e}")

for i in range(runLoop):
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_file = os.path.join(current_dir, 'params', 'params.txt')

    with open(params_file, 'w') as f:
        for key, value in dic.items():
            f.write(f"{key}={value}\n")

        print(f"Parameters saved to {params_file}")