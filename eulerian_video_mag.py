import cv2
import numpy as np
import scipy.signal as signal
import sys


###########################################################################################################
## Main function for Eulerian Motion Magnification ########################################################
###########################################################################################################

def magnify_motion(video_name, low, high, levels=3, amplification = 10):
    t,f = load_video(video_name)
    lap_video_list = laplacian_video(t,levels=levels)
    filter_list = []

    for i in range(levels):
        #Find the desired range of frequencies
        filter = butter_bandpass_filter(lap_video_list[i],low,high,f)

        #Amplify these frequencies
        filter *= amplification
        filter_list.append(filter)


    #Collapse the pyramid whose frames' specific frequencies are amplified
    recon = pyr_reconstruct(filter_list)

    final = t + recon
    save_video(final)

###########################################################################################################
## Video input and output handling ########################################################################
###########################################################################################################

#load video from file
def load_video(video_filename):
    capture = cv2.VideoCapture(video_filename)
    number_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    video = np.zeros((number_of_frames, h, w, 3),dtype='float')
    x=0
    while capture.isOpened():
        ret,frame = capture.read()
        if ret is True:
            video[x] = frame
            x+=1
        else:
            break
    return video,fps

def save_video(video):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [h, w]=video[0].shape[0:2]
    writer = cv2.VideoWriter("out.avi", fourcc, 30, (w, h), 1)
    for i in range(0, video.shape[0]):
        writer.write(cv2.convertScaleAbs(video[i]))
    writer.release()

###########################################################################################################
## Spatial Decomposition: Construct laplacian pyramids for video frames ###################################
###########################################################################################################

#Build Laplacian Pyramid
def pyr_build(src,levels=3):
    #Building the gaussian pyramid
    img = src.copy()
    gaussian = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        gaussian.append(img)
    laplacian = []
    #Since the Laplacian is just the difference between successive gaussian blurs,
    #we can build the laplacian based off of the gaussian.
    for i in range(levels,0,-1):
        layer = cv2.subtract(gaussian[i-1], cv2.pyrUp(gaussian[i]))
        laplacian.append(layer)
    return laplacian

#Build laplacian pyramid for video
def laplacian_video(video,levels=3):
    mylist = []
    for i in range(0,video.shape[0]):
        frame = video[i]
        pyr = pyr_build(frame,levels=levels)
        if i == 0:
            for j in range(levels):
                mylist.append(np.zeros((video.shape[0],pyr[j].shape[0],pyr[j].shape[1],3)))
        for k in range(levels):
            mylist[k][i] = pyr[k]
    return mylist

###########################################################################################################
## Butterworth bandpass filter for signal processing ######################################################
###########################################################################################################

#The following code in this section is adapted from:
#https://www.programcreek.com/python/example/59508/scipy.signal.butter

def butter_bandpass_filter(data, lowcut, highcut, sampling_rate, order=5):
    # calculate the Nyquist frequency
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    return y

###########################################################################################################
## Reconstract video by collapsing the amplified laplacian pyramid ########################################
###########################################################################################################

def pyr_reconstruct(filter_list,levels=3):
    final = np.zeros(filter_list[-1].shape)
    for i in range(filter_list[0].shape[0]):
        up = filter_list[0][i]
        for n in range(levels-1):
            up = cv2.pyrUp(up)+filter_list[n + 1][i]

        #Clip intensity values to 0,255 range then converting to uint8
        up = np.clip(up,0,255).astype(np.uint8)
        final[i] = up
    return final

###########################################################################################################
## Terminal argument parsing ##############################################################################
###########################################################################################################

def main():
    if len(sys.argv) != 4:
        print('usage: python eulerian_video_mag.py video_name low high')
        sys.exit(1)

    video_name = sys.argv[1]
    low = float(sys.argv[2])
    high = float(sys.argv[3])
    magnify_motion(video_name, low, high)


if __name__=="__main__":
    main()
