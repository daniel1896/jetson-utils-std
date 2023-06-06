"""
A program to test the cuda implementation of the standard deviation function.
"""

import jetson_utils
import numpy as np
import cv2
import time


class CudaImageList:
    """
    Uses queue to store images on the same memory location using an index that is incremented to indicate the current frame.
    Use cudaMemCpy to copy the frame to the current index location.
    """
    def __init__(self, size, image_shape, format):
        self.size = size
        self.image_shape = image_shape
        self.frame_count = 5
        self.current_index = 0
        self.frame_list = []
        # fill the list with empty cudaImages
        for i in range(self.frame_count):
            self.frame_list.append(jetson_utils.cudaAllocMapped(width=image_shape[0], height=image_shape[1], format=format))

    def append(self, gray_frame):
        jetson_utils.cudaMemcpy(self.frame_list[self.current_index], gray_frame)
        self.current_index = (self.current_index + 1) % self.frame_count

    def get(self):
        return self.frame_list
    
    def get_sorted(self):
        return self.frame_list[self.current_index:] + self.frame_list[:self.current_index]
    
    def get_latest(self):
        return self.frame_list[self.current_index]
    
    def clear(self):
        self.frame_list.clear()


# Create a video source
# rtsp://CamUser:CamUser@<IP address>/axis-media/media.amp
input_src = "rtsp://CamUser:CamUser@10.0.220.114/axis-media/media.amp"  # RTSP Stream
input_src = "/dev/video0"   # USB Webcam
format = 'gray32f'
stream = jetson_utils.videoSource(input_src, {'width': 1280, 'height': 720, 'framerate': 30})

img_width = stream.GetWidth()
img_height = stream.GetHeight()

# resize variables
img_width = 1920
img_height = 1080

# Create an empty cuda image
image_list = CudaImageList(5, (img_width, img_height), format)
res_image = jetson_utils.cudaAllocMapped(width=img_width, height=img_height, format='rgb8')
gray_image = jetson_utils.cudaAllocMapped(width=img_width, height=img_height, format=format)
std_image = jetson_utils.cudaAllocMapped(width=img_width, height=img_height, format=format)


# Capture a frame
while True:
    frame = stream.Capture()
    # resize the frame
    jetson_utils.cudaResize(frame, res_image)
    jetson_utils.cudaConvertColor(res_image, gray_image)

    # Append the frame to the image list
    image_list.append(gray_image)

    for i,img in enumerate(image_list.get()):
        cv2.imshow('frame'+str(i), jetson_utils.cudaToNumpy(img)/255)

    # Time the standard deviation function
    jetson_utils.cudaDeviceSynchronize()
    t_start = time.time()

    # Calculate the standard deviation
    if len(image_list.get()) >= 5:
        jetson_utils.cudaStdDev(*image_list.get_sorted(), std_image, img_width, img_height)
    jetson_utils.cudaDeviceSynchronize()

    # Print the time taken
    print("Time taken: ", time.time() - t_start)
    
    # Show the frame
    cv2.imshow('frame', jetson_utils.cudaToNumpy(std_image))

    # close the window if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #time.sleep(0.1)

# Release the capture
stream.Close()
cv2.destroyAllWindows()
