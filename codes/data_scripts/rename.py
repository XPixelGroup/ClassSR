import os
import glob


def main():
    folder = '/data0/xtkong/data/DIV2K_valid_sub/LR'
    DIV2K(folder)
    print('Finished.')


def DIV2K(path):
    index=0
    img_path_l = os.listdir(path)
    for img_path in img_path_l:
        print(index)
        index+=1
        if 'x4' in img_path:
            new_path = img_path.replace('x4', '')
            os.rename(os.path.join(path,img_path), os.path.join(path,new_path))


if __name__ == "__main__":
    main()