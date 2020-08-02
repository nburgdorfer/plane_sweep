import numpy as np
import sys
import cv2

def canny_edge_detect(img):
    edges = cv2.Canny(img,80,100)

    return edges

def main():
    
    if (len(sys.argv) != 2):
        print("Error: usage python3 {} <filename>".format(sys.argv[0]))
        sys.exit()

    filename = sys.argv[1]
    img = cv2.imread(filename)
    edges = canny_edge_detect(img)
    cv2.imwrite("edges.png",edges)

if __name__ == "__main__":
    main()
