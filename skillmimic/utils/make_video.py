from moviepy.editor import ImageSequenceClip
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_path",type=str)
parser.add_argument("--fps",type=int,default=30)
args = parser.parse_args()

image_folder = args.image_path

image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".png")]

image_files.sort()

clip = ImageSequenceClip(image_files, fps=args.fps)

data_name = image_folder[len('skillmimic/data/images/'):] #projectname
os.makedirs("skillmimic/data/videos/", exist_ok=True)
out_path = 'skillmimic/data/videos/' + data_name + '.mp4' #projectname
clip.write_videofile(out_path) 
