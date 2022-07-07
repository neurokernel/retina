from matplotlib import pyplot as plt
import cv2 
import numpy as np

def video_capture(video_file_name):
    # read a mp4 file, and turn RGB to gray, and convert it into an ndarray
    cap = cv2.VideoCapture(video_file_name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    # frame number now
    #print(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print(f'total frames number is {total_frames}')
    print(f' height and width is {frame_height} and {frame_width}')
    print(f'fps is {fps}')

    video_array = np.empty((total_frames, frame_height, frame_width), dtype='uint8')

    fc = int(0)
    ret = True

    while (fc < total_frames and ret):
        ret, image = cap.read()
        # make RGB to gray
        ash_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        video_array [fc, :, :] = ash_image
        fc += 1

    cap.release()
    return video_array
    
def video_adapter(video_array, T, steps):
    # T and steps are those in the configuration files. 
    # The input to the retina model has total_frames_number = steps and the frame_interval = T
    # which means somehow we need to duplicate some frames or extract some
    # now it's only dulication
    
    fps_retina = 1/T # the actually fps in retina_input
    # the actual frames are steps = 5000 in ths example
    # so actually we have to convert the video into 5s
    # human and flies are really different in this way
    # 1094 to 5000

    # if total_frame > steps
    # randomly choose frames to repeat one more time
    normal_repeat_time = int(steps/total_frames)
    print(normal_repeat_time)
    reminder = steps - normal_repeat_time * total_frames
    chosen_frames = np.random.permutation(total_frames)[:reminder] # without dupication
    print(type(chosen_frames))
    print(np.shape(chosen_frames))
    print(np.shape(np.unique(chosen_frames)))
    # then do the duplication
    retina_input_video = np.empty((steps, frame_height, frame_width), dtype='uint8')
    '''
    print(np.shape(retina_input_video[3:6]))
    print(np.shape(video_array[1]))
    print(np.shape(np.tile(video_array[1], (3, 1,1))))
    '''

    count = 0
    for i in range(total_frames):
        if not (i in chosen_frames):
            retina_input_video[count:count+ normal_repeat_time] = np.tile(video_array[i], (normal_repeat_time, 1,1))
            count += normal_repeat_time
        else:
            repeat_time = normal_repeat_time+1
            retina_input_video[count:count+ repeat_time] = np.tile(video_array[i], (repeat_time, 1,1))
            count += repeat_time
    
    return retina_input_video

    