import os
import matplotlib.image as mpimg

class Loader():
    def __init__(self, input_imgs_folder = "", output_imgs_folder = "", input_video_folder = "", output_video_folder = ""):
        self.input_imgs_folder = input_imgs_folder
        self.output_imgs_folder = output_imgs_folder
        self.input_video_folder = input_video_folder
        self.output_video_folder = output_video_folder
        self.list_files_imgs = []
        self.list_files_videos = []

    def get_input_videos(self, folder = None):
        if folder is None:
            folder = self.input_video_folder
        self.list_files_vids = os.listdir(folder)
        for n in range(len(self.list_files_vids)):
            self.list_files_vids[n]=folder+self.list_files_vids[n]
        return self.list_files_vids 

    def get_output_videos(self, folder = None):
        if folder is None:
            folder = self.output_video_folder
        for n in range(len(self.list_files_vids)):
            self.list_files_vids[n] = self.list_files_vids[n].replace(self.input_video_folder,folder)
        return self.list_files_vids 

    def read_imgs(self, folder = None):
        if folder is None:
            folder = self.input_imgs_folder
        images = []
        self.list_files_imgs = os.listdir(folder)
        for file in self.list_files_imgs:
            images.append(mpimg.imread(folder+file))
        return images

    def read_imgs_isolated(self, folder):
        images = []
        list_files = os.listdir(folder)
        for file in list_files:
            images.append(mpimg.imread(folder+file))
        return images

    def save_imgs(self, images, list_files = None, folder = None):
        if list_files is None:
            list_files = self.list_files_imgs
        self.list_files_imgs = list_files
        if folder is None:
            folder = self.output_imgs_folder
        self.output_imgs_folder = folder
        for n in range(len(self.list_files_imgs)):
            mpimg.imsave(self.output_imgs_folder+self.list_files_imgs[n], images[n])
