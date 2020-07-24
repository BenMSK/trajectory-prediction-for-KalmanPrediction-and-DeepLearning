import numpy as np
# this script is for generating track file for Lyft dataset
# dataset = os.listdir(raw_data_dir)

# for ind, raw_file_name in enumerate(dataset):
#     file_path = os.path.join(raw_data_dir, raw_file_name)# Each .txt or .csv file
#     read = np.load(file_path)#Lyft

#     agents = np.unique(read[:,id_ind])
#     track_data = None# Shape: agent x (frame, x, y)
#     for agent in agents:
#         a_track = read[read[:,id_ind]==agent][:,[id_ind, x_ind, y_ind]]
#         track_data[agent] = a_track
#     track_data_list[dataset_name] = track_data
frame_idx = 20
interval = 6
obs_length = 6
a = np.arange(100)
# print(frame_idx+1 - obs_length*(interval)+(interval-1))
obs_length = int(frame_idx/interval + 1) if frame_idx - (obs_length-1)*(interval) < 0 else obs_length
        # print("minus!: ", obs_length)
start_idx = np.maximum(0, frame_idx - (obs_length-1)*(interval))
end_idx = frame_idx + 1
print(a[start_idx:end_idx:interval])

start = frame_idx-len(a) - (obs_length-1)
end = frame_idx-len(a)+1
print(a[start:end:-interval])
