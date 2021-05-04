import cv2 
import numpy as np 
import os.path as osp
import re

def saveVideoFrames(save_path, frames, w, h, fps, codec):
    '''
        function to save video frames as video file
        args:
            save_path: the path to save the file
            frames: list of frames (frame is images, np.array)
            w: width of the video
            h: height of the video
            fps: fps of the video
            codec: codec of the video
        return:
            None
    '''
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(save_path, fourcc, fps*1.0, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

def SaveKinematics(save_path, kinematics):
    '''
        function to save kinematics:
        args:
            save_path: the path to save the kinematics
            kinematics: list of kinematics, (one item is a dict with all kinematic informations)
        return:
            None
    '''
    f = open(save_path, 'w')
    out = ''
    for i in range(len(kinematics)):
        kinematic = kinematics[i]
        out += kinematic['origin']
    # print(kinematic['origin'])
    f.write(out)
    f.close()

def write_annotate_file(annotate_file, annotate_list):
  '''
        function to save annotation file
        args:
            annotate_file: the path to save the annotate_file
            annotate_list: list of annotations, each item is a dict contains all annotation info for each gesture
  '''
  f = open(annotate_file, 'w')
  output = ''
  for s in annotate_list:
    output += str(s['start'])+' '+str(s['end'])+' '+s['label']+' '
    if 'keyframes' not in s:
      s['keyframes'] = [s['start'], s['end']]
    for keyframe in s['keyframes']:
      output += str(keyframe)+','
    output = output[:-1]+' '
    if 'annotations' not in s:
      s['annotations'] = [ord('o')]
    for annotate in s['annotations']:
      output += chr(annotate)+','
    output = output[:-1]+'\n'
  f.write(output)

def getVideoFrames(path):
    '''
        get video frames from the file
        args:
            path: the path of the video
        return:
            frames: list of frames (frame is images, np.array)
            w: width of the video
            h: height of the video
            fps: fps of the video
            codec: codec of the video
    '''
    cap = cv2.VideoCapture(path) 
    w,h,fps,codec = int(cap.get(3)),int(cap.get(4)),int(cap.get(5)),int(cap.get(6))
    codec = chr(codec&0xFF) + chr((codec>>8)&0xFF) + chr((codec>>16)&0xFF) + chr((codec>>24)&0xFF)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, w, h, fps, codec

def readKinematics(path):
  '''
        get kinematics from file
        args:
            path: path of the kinematics file
        return:
            kinematics: list of kinematics, (one item is a dict with all kinematic informations)
  '''
  f = open(path, 'r')
  kinematics_of_all_frames = f.readlines()
  # print('frame number', len(kinematics_of_all_frames))
  kinematics = []

  for k_per_frame in kinematics_of_all_frames:
    kinematic = {}
    kinematic['origin'] = k_per_frame
    k_per_frame = re.sub(' +', ',', k_per_frame)
    data = k_per_frame[:-1].split(',')[1:]
    #print(len(data))
    float_data = []
    for d in data:
      #print(type(d), d, float(d))
      float_data.append(float(d))
    data = float_data
    offset = 0
    kinematic['ml_tool_xyz'] = [data[0+offset], data[1+offset], data[2+offset]]
    kinematic['ml_tool_R'] = [[data[3+offset], data[4+offset], data[5+offset]],
                                  [data[6+offset], data[7+offset], data[8+offset]],
                                  [data[9+offset], data[10+offset], data[11+offset]]]
    kinematic['ml_tool_tv'] = ([data[12+offset], data[13+offset], data[14+offset]])
    kinematic['ml_tool_rv'] = ([data[15+offset], data[16+offset], data[17+offset]])
    kinematic['ml_grip_ang'] = ([data[18+offset]])
    offset = 19
    kinematic['mr_tool_xyz'] = ([data[0+offset], data[1+offset], data[2+offset]])
    kinematic['mr_tool_R'] = ([[data[3+offset], data[4+offset], data[5+offset]],
                                  [data[6+offset], data[7+offset], data[8+offset]],
                                  [data[9+offset], data[10+offset], data[11+offset]]])
    kinematic['mr_tool_tv'] = ([data[12+offset], data[13+offset], data[14+offset]])
    kinematic['mr_tool_rv'] = ([data[15+offset], data[16+offset], data[17+offset]])
    kinematic['mr_grip_ang'] = ([data[18+offset]])
    offset = 38
    kinematic['sl_tool_xyz'] = ([data[0+offset], data[1+offset], data[2+offset]])
    kinematic['sl_tool_R'] = ([[data[3+offset], data[4+offset], data[5+offset]],
                                  [data[6+offset], data[7+offset], data[8+offset]],
                                  [data[9+offset], data[10+offset], data[11+offset]]])
    kinematic['sl_tool_tv'] = ([data[12+offset], data[13+offset], data[14+offset]])
    kinematic['sl_tool_rv'] = ([data[15+offset], data[16+offset], data[17+offset]])
    kinematic['sl_grip_ang'] = ([data[18+offset]])
    offset = 57
    kinematic['sr_tool_xyz'] = ([data[0+offset], data[1+offset], data[2+offset]])
    kinematic['sr_tool_R'] = ([[data[3+offset], data[4+offset], data[5+offset]],
                                  [data[6+offset], data[7+offset], data[8+offset]],
                                  [data[9+offset], data[10+offset], data[11+offset]]])
    kinematic['sr_tool_tv'] = ([data[12+offset], data[13+offset], data[14+offset]])
    kinematic['sr_tool_rv'] = ([data[15+offset], data[16+offset], data[17+offset]])
    kinematic['sr_grip_ang'] = ([data[18+offset]])
    kinematics.append(kinematic)
  f.close()
  return kinematics

def read_annotate_file(annotate_file):
  '''
        function to read annotate file:
        args:
            annotate_file: path to the annotate_file
        return:
            annotate_list: list of annotations, each item is a dict contains all annotation info for each gesture
  '''
  assert osp.exists(annotate_file)
  annotate_list = []
  f = open(annotate_file, 'r')
  slices = f.readlines()
  for line in slices:
    s = line.split(' ')
    s_dict = {'start':int(s[0]),'end':int(s[1]),'label':s[2]}
    if len(s) >= 5:
        keyframes = s[3].split(',')
        s_dict['keyframes'] = [int(keyframe) for keyframe in keyframes]
        s_dict['annotations'] = [ord(c) for c in s[4][:-1].split(',')]
    else:
        s_dict['keyframes'] = [int(s[0]), int(s[1])]
        s_dict['annotations'] = [ord('o')]
    annotate_list.append(s_dict)
  f.close()
  return annotate_list



def testGetVideoFrames():
    print("################################################################################")
    print("test getVideoFrames")
    path = '/Users/dinghao/Desktop/CIS2-PROJECT/data/Knot_Tying/video/Knot_Tying_B001_capture1.avi'
    frames, w, h, fps, codec = getVideoFrames(path)
    #Compare Results and ground truth
    print("GroundTruth:", 640,480,30,'BX50')
    print("Results", w,h,fps,codec)
    for frame in frames:
        cv2.imshow('testGetVideoFrames', frame)
        cv2.waitKey(10)
    print("################################################################################")

def test_read_annotate_file():
    print("################################################################################")
    print("test read_annotate_file")
    print("file1:")
    path1 = '/Users/dinghao/Desktop/CIS2-PROJECT/data/Knot_Tying/transcriptions/Knot_Tying_B001.txt'
    annotate_list = read_annotate_file(path1)
    print("the ground truth first line:", "45 85 G12 ")
    print("the first line result:", annotate_list[0])
    print("the ground truth last line:", "1627 1735 G11 ")
    print("the last line:", annotate_list[-1])
    print("\nfile2:")
    path2 = '/Users/dinghao/Desktop/CIS2-PROJECT/data/Knot_Tying/rectified_transcriptions/Knot_Tying_B001.txt'
    annotate_list = read_annotate_file(path2)
    print("the ground truth first line:", "45 85 G12 45,85 s")
    print("the first line result:", annotate_list[0])
    print("the ground truth last line:", "1627 1735 G11 1627,1735 s")
    print("the last line:", annotate_list[-1])
    print("################################################################################")

def test_write_annotate_file():
    print("################################################################################")
    print("test write_annotate_file")
    path = '/Users/dinghao/Desktop/CIS2-PROJECT/data/Knot_Tying/transcriptions/Knot_Tying_B001.txt'
    annotate_list = read_annotate_file(path)
    save_path = '/Users/dinghao/Desktop/CIS2-PROJECT/data/Knot_Tying/test_file/Knot_Tying_B001.txt'
    write_annotate_file(save_path, annotate_list)
    new_annotate_list = read_annotate_file(save_path)
    print("readed line 0:", annotate_list[0])
    print("writted line 0:", new_annotate_list[0])
    print("readed line -1:", annotate_list[-1])
    print("writted line -1:", new_annotate_list[-1])
    print("################################################################################")

def test_readKinematics():
    print("################################################################################")
    print("test readKinematics")
    path = '/Users/dinghao/Desktop/CIS2-PROJECT/data/Knot_Tying/kinematics/AllGestures/Knot_Tying_B001.txt'
    kinematics = readKinematics(path)
    print("the ground truth first line:", "     0.11118000     -0.03302000      0.41599100     -0.42126600      0.60901200     -0.67207500     -0.69965700     -0.68968900     -0.18649700     -0.57658500      0.39089600      0.71748000      0.01490500      0.00326700      0.00217600      0.00094400     -0.01469700      0.02952000      0.00067300     -0.12736600     -0.01991700      0.41087700      0.39098400      0.71030400      0.58531300     -0.64734500      0.66432300     -0.37356900     -0.65464700     -0.23374400      0.71879400     -0.04899900     -0.03408100      0.09553500     -0.26803100      0.57987800     -0.19505400     -0.75238300      0.03130500      0.03212300     -0.00635100     -0.27196000     -0.77711500      0.56756400      0.71895100     -0.55611700     -0.41694200      0.63964500      0.29465900      0.70995000     -0.02453300     -0.01743100      0.04755400     -0.09255700      0.96064200     -0.43986100     -0.88222000      0.06526200      0.01951400     -0.06257000     -0.53124800      0.54530700     -0.64839400     -0.56094500     -0.79993900     -0.21316000     -0.63491400      0.25047300      0.73085300      0.00765200      0.00650700     -0.00390900      0.04638100     -0.03959800      0.08034500     -0.21441900")
    print("the first line result:", kinematics[0])
    print("the ground truth last line:", "     0.11718600      0.03620000      0.48334700     -0.25987100      0.86023900     -0.43875100     -0.96333600     -0.26229800      0.05618400     -0.06643600      0.43619400      0.89741200      0.01017400     -0.00410700     -0.00186000      0.00098400     -0.01471500      0.02955300     -1.00590500     -0.09710300      0.03984200      0.45207100      0.45142700      0.54504800      0.70648900     -0.83400800      0.53910900      0.11715500     -0.31756600     -0.64278500      0.69702400      0.01085000      0.00254900      0.00055400      0.00204500     -0.01520500      0.02924500     -1.05407900      0.04640200      0.06198100      0.01431600     -0.34987300     -0.62612800      0.69681500      0.90307500     -0.42320500      0.07316100      0.24908800      0.65487400      0.71350800      0.00541000      0.00134000      0.00029900      0.00201500     -0.01522700      0.02920100     -0.84482700      0.06827700      0.05413400     -0.02889100     -0.41740300      0.81326200     -0.40543500     -0.89772000     -0.43824600      0.04514100     -0.14096900      0.38280900      0.91300800      0.00510000     -0.00200200     -0.00091900      0.00093600     -0.01471100      0.02970000     -0.87248100")
    print("the last line:", kinematics[-1])
    print("################################################################################")

def test_saveVideoFrames():
    print("################################################################################")
    print("test saveVideoFrames")
    path = '/Users/dinghao/Desktop/CIS2-PROJECT/data/Knot_Tying/video/Knot_Tying_B001_capture1.avi'
    frames, w, h, fps, codec = getVideoFrames(path)
    save_path = '/Users/dinghao/Desktop/CIS2-PROJECT/data/Knot_Tying/test_file/Knot_Tying_B001_capture1.avi'
    saveVideoFrames(save_path, frames, w, h, fps, codec)
    frames, w, h, fps, codec = getVideoFrames(save_path)
    #Compare Results and ground truth
    print("GroundTruth:", 640,480,30,'BX50')
    print("Results", w,h,fps,codec)
    for frame in frames:
        cv2.imshow('testSaveVideoFrames', frame)
        cv2.waitKey(10)
    print("################################################################################")

def test_SaveKinematics():
    print("################################################################################")
    print("test SaveKinematics")
    path = '/Users/dinghao/Desktop/CIS2-PROJECT/data/Knot_Tying/kinematics/AllGestures/Knot_Tying_B001.txt'
    kinematics = readKinematics(path)
    save_path = '/Users/dinghao/Desktop/CIS2-PROJECT/data/Knot_Tying/test_file/Knot_Tying_B001.txt'
    SaveKinematics(save_path, kinematics)
    new_kinematics = readKinematics(save_path)
    print("first ground truth:\n", kinematics[0])
    print("first saved:\n", new_kinematics[0])
    print("last ground truth:\n", kinematics[-1])
    print("last saved:\n", new_kinematics[-1])
    print("################################################################################")




if __name__ == '__main__':
    pass
    #test_saveVideoFrames()
    #testGetVideoFrames()
    #test_read_annotate_file()
    #test_write_annotate_file()
    #test_readKinematics()
    #test_SaveKinematics()
