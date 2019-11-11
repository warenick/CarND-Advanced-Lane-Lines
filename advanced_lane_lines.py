
from Loader import Loader
from Line import Line
from Calibrator import Calibrator
from Pipeline import Pipeline
from pipeline_lib import *
import imageio
# imageio.plugins.ffmpeg.download
# imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# load images
images_folder = "test_images/"
images_folder_out = "output_images/"
calibration_folder = "camera_cal/"
video_folder = "test_video/"
video_folder_out = "output_video/"

loader = Loader(input_imgs_folder=images_folder,
                output_imgs_folder=images_folder_out,
                input_video_folder=video_folder,
                output_video_folder=video_folder_out)
images = loader.read_imgs()
calibrator = Calibrator(folder=calibration_folder)
calibrator.load_calibration_images()
calibrator.calculate()
# init lines
left_line = Line()
right_line = Line()
pipeline = Pipeline(calibrator)
output_imgs = []
for img in images:
    show_img(img)
    undist = calibrator.undistort(img)
    # found gradient
    gradient = pipeline.compute_gradient(undist)
    # wrap imgs
    wrapped = pipeline.wrap(gradient)
    # find left and right pixel poses
    # find left and right polyline
    left_line.count_check()
    right_line.count_check()
    lx, ly, rx, ry = pipeline.find_lines(wrapped)
    if len(rx) > 0 and len(ry) > 0:
        # set was detected
        right_line.was_detected = True
        # add founded points to instanse
        right_line.allx = rx
        right_line.ally = ry
        # add poly 
        l_fit, right_line.polx, right_line.ploty = pipeline.find_poly(wrapped.shape,right_line.allx, right_line.ally)
        right_line.add_fit(l_fit)
    # if find points of left line
    if len(lx) > 0 and len(ly) > 0:
        # set was detected
        left_line.was_detected = True
        # add founded points to instanse
        left_line.allx = lx
        left_line.ally = ly
        # add poly 
        r_fit, left_line.polx, left_line.ploty = pipeline.find_poly(wrapped.shape,left_line.allx, left_line.ally)
        left_line.add_fit(r_fit)

    left_curvd = pipeline.measure_curv(wrapped.shape, lx, ly)
    right_curvd = pipeline.measure_curv(wrapped.shape, rx, ry)
    avg_curvrd = round(np.mean([left_curvd,right_curvd]), 0)
    curv_text = f"current radius of  curvature is {avg_curvrd} meters"
    car_pose = pipeline.measure_pose(wrapped,left_line.current_fit, right_line.current_fit, left_line.ploty)
    if car_pose > 0:
        pose_text = f"car is {round(abs(car_pose), 2)} meters left from center"
    else:
        pose_text = f"car is {round(abs(car_pose), 2)} meters right from center"
    font = cv2.FONT_HERSHEY_PLAIN
    
    cv2.putText(undist, curv_text, (10,100), font, 2, (30,240,30), 3)
    cv2.putText(undist, pose_text, (10,150), font, 2, (30,240,30), 3)
        
    #Draving lines
    # find best fits  for drawing
    left_line.find_best_fit()
    right_line.find_best_fit()
    # invert matrix of tranfer
    color_wrap = pipeline.draw_by_fit(wrapped, left_line.best_fit, right_line.best_fit)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarp = pipeline.unwrap(color_wrap)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, unwarp, 0.3, 0) 
    show_img(result)
    output_imgs.append(result.copy())
loader.save_imgs(output_imgs)


def process_image(image):
    global pipeline
    result = pipeline.proccess(image)
    return result
videos_in = loader.get_input_videos()
videos_out = loader.get_output_videos()

for n in range(len(videos_in)):
    clip1 = VideoFileClip(videos_in[n])
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(videos_out[n], audio=False)
print("finish proccess")