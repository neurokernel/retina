from matplotlib import pyplot as plt
import cv2 
import numpy as np

def video_capture(video_file_name, scale):
    # read a mp4 file, and turn RGB to gray, and convert it into an ndarray
    cap = cv2.VideoCapture(video_file_name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    # frame number now
    #print(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print(f'the origin video has {total_frames} frames')
    #print(f' height and width is {frame_height} and {frame_width}')
    #print(f'fps is {fps}')

    video_array = np.empty((total_frames, frame_height, frame_width), dtype=np.double)

    fc = int(0)
    ret = True

    while (fc < total_frames and ret):
        ret, image = cap.read()
        # make RGB to gray
        #ash_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ash_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.double)
        ash_image *= scale/np.max(ash_image)
        #ash_image.astype('uint8')
        video_array [fc, :, :] = ash_image.astype(np.double)
        fc += 1
    cap.release()
    return video_array
    
def video_adapter(video_array, T, steps):
    # whatever the time length of origin video is, we always convert it to adapt to the retina model, 
    # which is defined in configuration file
    # T and steps are dt & steps in the configuration files. 
    # The input to the retina model has total_frames_number = steps and the frame_interval = T
    # which means we need to duplicate some frames or extract some
    
    
     
    fps_retina = 1/T # the actually fps in retina_input
    total_frames, frame_height, frame_width = np.shape(video_array)
    
    retina_input_video = np.empty((steps, frame_height, frame_width), dtype=np.double)
    
    if steps > total_frames: # we need duplicate some frames
        normal_repeat_time = int(steps/total_frames) # randomly choose frames to repeat one more time
        #print(normal_repeat_time)
        reminder = steps - normal_repeat_time * total_frames
        chosen_frames = np.random.permutation(total_frames)[:reminder] # those doing extra dupication

        count = 0
        for i in range(total_frames):
            if not (i in chosen_frames):
                retina_input_video[count:count+ normal_repeat_time] = np.tile(video_array[i],
                                                                              (normal_repeat_time, 1,1))
                count += normal_repeat_time
            else:
                repeat_time = normal_repeat_time+1
                retina_input_video[count:count+ repeat_time] = np.tile(video_array[i], (repeat_time, 1,1))
                count += repeat_time
    else: # we need extract some frames randomly, but usually we don't need this 
        chosen_frames = np.random.permutation(total_frames)[:steps] 
        # simply randomly choose which frames to be remained
        count = 0
        for i in range(total_frames):
            if i in chosen_frames:
                retina_input_video[count] = video_array[i]
                count += 1
    
    return retina_input_video, [fps_retina, total_frames, frame_height, frame_width]

def frames_adapter(video_array, T, steps):
    # whatever the time length of origin video is, we always convert it to adapt to the retina model, 
    # which is defined in configuration file
    # T and steps are dt & steps in the configuration files. 
    # The input to the retina model has total_frames_number = steps and the frame_interval = T
    # which means we need to duplicate some frames or extract some
    
    
     
    fps_retina = 1/T # the actually fps in retina_input
    total_frames, frame_height, frame_width = np.shape(video_array)
    
    retina_input_frames = np.empty(steps)
    
    if steps > total_frames: # we need duplicate some frames
        normal_repeat_time = int(steps/total_frames) # randomly choose frames to repeat one more time
        #print(normal_repeat_time)
        reminder = steps - normal_repeat_time * total_frames
        chosen_frames = np.random.permutation(total_frames)[:reminder] # those doing extra dupication

        count = 0
        for i in range(total_frames):
            if not (i in chosen_frames):
                retina_input_frames[count:count+ normal_repeat_time] = i
                count += normal_repeat_time
            else:
                repeat_time = normal_repeat_time+1
                retina_input_frames[count:count+ repeat_time] = i
                count += repeat_time
    else: # we need extract some frames randomly, but usually we don't need this 
        chosen_frames = np.random.permutation(total_frames)[:steps] 
        # simply randomly choose which frames to be remained
        count = 0
        for i in range(total_frames):
            if i in chosen_frames:
                retina_input_frames[count] = i
                count += 1
    
    return retina_input_frames

def main():
    # for test to see if it works
    video_array = video_capture('video_demo.mp4')
    T = 1e-3
    steps = 2000
    retina_input_video, retina_input_video_info = video_adapter(video_array, T, steps)
    fps_retina, total_frames, frame_height, frame_width = retina_input_video_info
    fps_retina = 1/(1e-3)
    
    writer = cv2.VideoWriter('video_demo_re.mp4',  fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                       fps=fps_retina, frameSize=(frame_width, frame_height), isColor=0)

    for frame in range(steps):
        writer.write(retina_input_video[frame])

    writer.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()

