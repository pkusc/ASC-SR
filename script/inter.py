import cv2
import os
import sys

if __name__ == '__main__':    
    origin_dir = sys.argv[1]
    modify_dir = sys.argv[2]

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    out_dir = sys.argv[3]
    alpha = 0.05

    items = os.listdir(dir1)
    for item in items:
        if not item.endswith('.png'):
            continue
        in_file1 = os.path.join(dir1, item)
        in_file2 = os.path.join(dir2, item)

        out_file = os.path.join(out_dir, item) 
        in_img1 = cv2.imread(in_file1).astype('float32')
        in_img2 = cv2.imread(in_file2).astype('float32')
        out_img = (1 - alpha) * in_img1 + alpha * in_img2

        cv2.imwrite(out_file, out_img)

    
