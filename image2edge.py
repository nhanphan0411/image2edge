import cv2
import imutils
import numpy as np
import pathlib
import os
from tqdm.notebook import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['one', 'many'], help='process one image or multiple images')
parser.add_argument('--width', type=int, default=256, help='output image width')
parser.add_argument('--height', type=int, default=256, help='output image width')
parser.add_argument('--input', type=str, help='one: path to a single image, many:path to the image folder')
parser.add_argument('--output', type=str, help='path to output folder')
parser.add_argument('--blackbg', type=bool, help='whether output image has black background or white background')

args = parser.parse_args()

class ImageProcessing(object):
    def __init__(self, image_width, image_height):  
        self.image_size = (image_width, image_height)
  
    def read(self, path):
        """Read one path to one BGR image."""
        img = cv2.imread(path)
        return cv2.resize(img, self.image_size)
    
    def get_one_edge(self, image_path, black_background=False, save_path=None):
        """Read and convert one image to Canny edge."""
        img = self.read(image_path)
        name = pathlib.Path(image_path).stem

        edge = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = imutils.auto_canny(edge)
        if black_background: 
            edge = cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)
        else:
            edge = cv2.cvtColor(255-edge,cv2.COLOR_GRAY2BGR)
        
        output = np.hstack((edge,img))

        if save_path:
            save_path = os.path.join(save_path, name+'.jpg')
            cv2.imwrite(save_path, output)
 
        return output
    
    def get_many_edge(self, folder_path, black_background=False, save_path=None):
        """Load, read, convert a folder of images into Canny edge."""
 
        # Load and preprocess images
        folder_lib = pathlib.Path(folder_path)
        all_image_paths = [str(item) for item in folder_lib.glob('*') if item.is_file()]
        
        if save_path:
            folder = pathlib.Path(save_path)
            out_folder = os.path.join(folder, 'output')
            try: 
                os.makedirs(out_folder)
                print(f"Directory {out_folder} created")
            except Exception as e:
                print(f'Directory {out_folder} already exists')
            
            outputs=[]
            for p in tqdm(all_image_paths):
                try: 
                    name = pathlib.Path(p).stem
                    output_path = os.path.join(out_folder, name+'.jpg')
                    output = self.get_one_edge(image_path=p,
                                               black_background=black_background, 
                                               save_path=output_path)
                    outputs.append(output)
                except Exception as e:
                    print(f"{e} at {p}")
        else: 
            try: 
                outputs = [self.get_one_edge(p, black_background=black_background) for p in all_image_paths]
            except Exception as e:
                print(f"{e} at {p}")
 
        return outputs

if __name__ == '__main__':
    processor = ImageProcessing(args['width'], args['height'])
    if args['mode'] == 'one':
        processor.get_one_edge(image_path=args['input'],
                               black_background=args['blackbg'],
                               save_path=args['output'])
    else:
        processor.get_many_edge(folder_path=args['input'],
                                black_background=args['blackbg'],
                                save_path=args['output'])
