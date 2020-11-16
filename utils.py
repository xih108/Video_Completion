import os
import cv2
import numpy as np
from PIL import Image
import imageio

def video_to_frames(filename, resize_h=64, resize_w=64):
    """                                                                                                           
    Extract every frame from a video, and keep one in every 3 frames, and resize each frame.
    For original video: expect 90 frames per 3 second video, each frame 256-by-256.
    If the video has less than 90 frames, fill all remaining frames up to num_frames
    """
    # an example for filename: 'videos/skiing_train/flickr-0-0-0-1-1-5-9-7-2400011597_4.mp4'
    vidcap = cv2.VideoCapture(filename)
    frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #expect 30 frames to be extracted; if original video has less than
    #90 frames, fill the trailing ones with the last sampled frame
    reduced_frameCount = 30

    data = np.empty((reduced_frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    
    fc = 0
    success = True
    while (fc < min(frameCount, 90) and success):
        if (fc%3==0):
            success, data[fc//3, ...] = vidcap.read()
        else:
            vidcap.read()
        fc += 1
    if frameCount < 90-3:
        for i in range(frameCount//3+1, 30):
            data[i, ...] = data[frameCount//3, ...]
        
    #print('Finished reading video')
        
    # keep only one frame out of every 3 frames, and resize every frame
    data_resized = np.empty((reduced_frameCount, resize_h, resize_w, 3), np.float)
    for i in range(reduced_frameCount):
        img = Image.fromarray(data[i,...], 'RGB')
        img = img.resize((resize_h, resize_w), Image.ANTIALIAS)
        data_resized[i, ...] = np.asarray(img, dtype=np.float)
        # below line just for testing purpose, make sure to comment out after testing!
        ###img.save('extracted_images/skiing_test_1_frame%d.jpg' % i)
        
    data_first_last = np.empty((2, resize_h, resize_w, 3), np.float)
    data_first_last[0] = data_resized[0]
    data_first_last[1] = data_resized[-1]

        
    return data_resized, data_first_last


def generated_video_to_frames(filename, resize_h=64, resize_w=64):
    """                                                                                                           
    Extract every frame from a generated video.
    For original video: expect 30 frames per 3 second video, each frame 64-by-64.
    If the video has less than 90 frames, fill all remaining frames up to num_frames
    """
    vidcap = cv2.VideoCapture(filename)
    frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #expect 30 frames to be extracted
    reduced_frameCount = 30

    data = np.empty((reduced_frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    
    fc = 0
    success = True
    while success and fc < 30:
        success, data[fc] = vidcap.read()
        fc += 1
    return data


def generate_resized_and_interpolated_video(file_in, input_frames=None, frames = False):
    # make two 3s long videos at 10 fps by reducing original video size, and by
    # linear interpolation between 1st and last frame
    if not frames:
        data, data_first_last = video_to_frames(file_in, resize_h=64, resize_w=64)
    else:
        data = file_in
        data_first_last = input_frames
    length = data.shape[0]
    h = data.shape[1]
    w = data.shape[2]
    
    data_interpolated = np.zeros([data.shape[0],h,w,3])

    for i in range(length):
        data_interpolated[i] = data_first_last[0] + i/length * (data_first_last[1] - data_first_last[0])
        
        
    data_interpolated = data_interpolated.astype(np.uint8)
    
    #data_interpolated = data_interpolated.astype(np.uint8)
#     print("Interpolated video written out into: {}".format(file_out_interpolated))
#     print("Reduced video written out into: {}".format(file_out_reduced))
    return data_interpolated

def L2Difference (video_gt, video_comp):
    distance = np.sqrt(np.sum(np.square(video_comp - video_gt)))
    return distance