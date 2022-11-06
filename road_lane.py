import cv2 
import numpy as np

def middle_lane_point(lines):
    y_const = 350
    x_const = 360
    x_right = 720
    x_left = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1 = (x1+x2)/2
        check = x1 - x_const
        if check>0 and x1<x_right:
            x_right = x1
        elif check<0 and x1>x_left:
            x_left = x1
    x= int((x_right+x_left)/2)
    return (x, y_const)
# def middle_lane_point(lines):

def lane_tracking(edges):
    lines_list =[]
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=30, # Min number of votes for valid line
                minLineLength=10, # Min allowed length of line
                maxLineGap=3# Max allowed gap between line for joining them
                )
    # print(lines)
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        lines_list.append([(x1,y1),(x2,y2)])
    (x,y) = middle_lane_point(lines)
    
    return lines, (x,y)

if __name__ == "__main__":
    cap = cv2.VideoCapture("./test_video.mp4")
    while True:
        _,src_img = cap.read() 
        src_img = cv2.resize(src_img,(720,480))
        gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        ksize = (5, 5)
        blur_img = cv2.blur(gray_img, ksize, cv2.BORDER_DEFAULT) 
        edges = cv2.Canny(blur_img,30,200,None, 3)
        image = src_img
        try:
            lines ,(x,y) = lane_tracking(edges)
            for line in lines:
                x1,y1,x2,y2=line[0]
                # Draw the lines joing the points
                # On the original image
                cv2.line(src_img,(x1,y1),(x2,y2),(0,255,0),2)
            image = cv2.circle(src_img, (x,y), radius=1, color=(0, 0, 255), thickness=4)
        except:
            print("error")
        cv2.imshow("black white image",edges)
        cv2.imshow("Image with lines",image)
        if cv2.waitKey(100) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()