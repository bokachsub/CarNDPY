import matplotlib.pyplot as plt
import math, cv2, glob
import numpy as np

class ImageVisualizer:
    # region vars

    # endregion
    def __init__(self, shape):
        self.shape = shape

    # to get a name of parameter variable to print a name in subplot
    def param_to_str(self, obj, namespace):
        return [name for name in namespace if namespace[name] is obj]

    # tool to draw multiple images on plot
    def plot_images(self, images, descriptions = None, invert_colors=False):
        w = self.shape[0]
        h = self.shape[1]
        fig = plt.figure(figsize=(8, 8))
        total_images = len(images)
        rows = math.ceil(total_images/3)
        columns = math.ceil(total_images/rows)
        for i in range(1, total_images + 1):
            fig.add_subplot(rows, columns, i)
            if invert_colors and len(images[i-1].shape) > 2:            
                plt.imshow(cv2.cvtColor(images[i-1], cv2.COLOR_BGR2RGB)) # need to invert colors
            else:
                plt.imshow(images[i-1]) # do not need to invert colors when showing binary images
            #plt.gca().set_title(self.param_to_str(images[i-1], globals()))
            if not descriptions is None:
                plt.gca().set_title(descriptions[i-1])
        mng = plt.get_current_fig_manager()  # to maximize window
        mng.window.state('zoomed')
        plt.show()
    
    def get_image_info(self, image, alias ):
        print(f'Image {alias} shape is: {image.shape}')
    
    # tests solid green area on blank image
    def draw_solid_area_on_blank_image(self, array_of_points):
        blank_image = np.zeros(self.shape, dtype=np.uint8)        
        cv2.fillPoly(blank_image, pts = [array_of_points],color = (255,255,0))
        return blank_image
    
    def overlay_image_with_solid_area(self, main_image, image2):        
        return cv2.addWeighted(main_image, 0.8, image2, 0.2, 0)

