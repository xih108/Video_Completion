"""
Code to extract every frame from a video.
Expect 90 frames per 3 second video.
"""

import cv2


def main():
    vidcap = cv2.VideoCapture('videos/skiing_train/flickr-0-0-0-1-1-5-9-7-2400011597_4.mp4')
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        cv2.imwrite("extracted_images/skiing_test_1_frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    return 0

main()
