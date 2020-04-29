import numpy as np

def get_box(pose):
    #input: pose([15,2])
    return(np.min(pose[:,0]), np.max(pose[:,0]), np.min(pose[:,1]), np.max(pose[:,1]))

def get_height(pose):
    #input: pose([15,2])
    mean_ankle = (pose[14]+pose[11])/2
    nose = pose[0]
    return np.linalg.norm(mean_ankle-nose)

def get_base_mean(pose):
    #input: pose([15,2])
    x1, x2, y1, y2 = get_box(pose)
    return np.array([(x1+x2)/2, y2])

def global_norm(driving_npy, target_npy):
    #input: pose([15,2,frame1]), pose([15,2,frame2])
    target_mean = np.mean(target_npy, axis=2)
    driving_mean = np.mean(driving_npy, axis=2)
    k2 = get_height(target_mean)/get_height(driving_mean)
    target_mean_base = get_base_mean(target_mean)
    driving_mean_base = get_base_mean(driving_mean)
    driving_npy_permuted = np.transpose(driving_npy, axes=[2, 0, 1])
    k = [1, k2]
    normalized_permuted = (driving_npy_permuted-driving_mean_base)*k+target_mean_base
    normalized = np.transpose(normalized_permuted, axes=[1,2,0])
    return normalized # pose([15,2,frame1])