import cv2
import os

def main():    
    data_path = "/space1/zhaoqing/dataset/MADPG_data_3/gt/"
    fps = 10          # 视频帧率
    size = (512, 256) # 需要转为视频的图片的尺寸
    video = cv2.VideoWriter("gt.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    
    file_list = os.listdir(data_path)
    file_list.sort()

    n = 0
    for i in file_list:
        if n >= 128:
            break
        n += 1
        image_path = data_path + i
        print(image_path)
        img = cv2.imread(image_path)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()