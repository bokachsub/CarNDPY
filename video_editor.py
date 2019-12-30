import numpy as np
import math, cv2, os

class VideoEditor():
    def __init__(self, input_file, output_file):        
        self.cap = cv2.VideoCapture(input_file)
        self.out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MP4V'), 25.0,(1280,720))
        self.start_frame=None
        self.end_frame=None
        self.next_frame_number = 0 # 1  OR 0 ??
        self.max_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) # does not count correctly
        print(f'Number of frames in video file is #{self.max_frame}')
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.next_frame_number) # setting frame to 0
        self.input_file_closed = False
        
    # reads the whole file frame by frame
    def read_frames(self):
        if (self.cap.isOpened()== True): 
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.next_frame_number)
            ret, frame = self.cap.read()
            if ret==False:                
                return False, frame, 0
            else: 
                self.next_frame_number += 1
                return True, frame, self.next_frame_number-1
        else:
            return False, None, 0
    
    # user can specify custom frames range instead of the whole file
    def read_frames_range(self, start_frame = 2,end_frame = 10):
        if( (self.start_frame==None) or (self.end_frame==None) ):
            self.start_frame = start_frame
            self.end_frame = end_frame
            self.next_frame_number = start_frame
        if((self.cap.isOpened()== True) and (start_frame <= self.next_frame_number<=end_frame)): 
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.next_frame_number)
            ret, frame = self.cap.read()
            if ret==False:                
                return False, frame, 0
            else: 
                self.next_frame_number += 1
                return True, frame, self.next_frame_number-1
        else:            
            self.start_frame = None
            self.end_frame = None
            return False, None, 0

    # records processed frames to output video file
    def write_frame(self,new_frame):
        self.out.write(new_frame)
    
    def write_single_image_to_folder(self, new_frame, folder = '/misc_frames/',  file_suffix = 'misc_frame_'):
        directory = os.getcwd() + folder
        try:
            os.stat(directory)
        except:
            print(f'Creating new directory {directory}...')
            os.mkdir(directory)
        file_path = os.getcwd() + folder + file_suffix + str(self.next_frame_number-1) + ".jpg"
        cv2.imwrite(file_path, new_frame)
    
    # adds custom text on a frame
    def add_text_on_frame(self, frame, text, text_location = (50,50), text_color = (255,0,0), thickness = 2):
        frame_with_text = cv2.putText(frame, text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, thickness, cv2.LINE_AA)
        return frame_with_text

    # returns TRUE if maximum frame(aka end of file) reached
    def max_frame_input_file_reached(self):
        return self.next_frame_number==self.max_frame
    
    # returns TRUE if class property is set to TRUE
    def input_file_closed(self):
        return self.input_file_closed
    
    # returns maximum frame number of input file, failing to work correctly
    def get_input_file_frame_number(self):
        return self.next_frame_number
    
    # closes input file explicitly
    def close_input_file(self):
        #if (self.cap.isOpened()== True): 
        self.cap.release()
        self.input_file_closed = True

    # closes output file explicitly
    def close_output_file(self):
        #if (self.out.isOpened()== True): 
        self.out.release()
    




        