import os
import sys

if __name__ == '__main__':
    origin_dir = sys.argv[1]
    items = os.listdir(origin_dir)
    for item in items:
        if not item.endswith('.png'):
            continue
        in_file = os.path.join(origin_dir,item)
        out_name = item[2:-12]
        print(item) 
        out_file = os.path.join(origin_dir, out_name + '.png')
        print(out_file)
        os.rename(in_file, out_file)
