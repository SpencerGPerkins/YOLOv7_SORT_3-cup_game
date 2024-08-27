import numpy as np
import cv2
import json

def find_center(coords):
    """Calculate the centers for quadrilateral
    Params
    --------
    coords : list of list, corner corrdinates of section
            [[left_side], [right_side]]

    Returns
    ---------
    (center_x, center_y) : tuple, the center coordinates of the section
    """
    x1, y1, x3, y3 = coords[0]
    x2, y2, x4, y4 = coords[1]
    center_x = (x1+x2+x3+x4) / 4
    center_y = (y1+y2+y3+y4) / 4

    return (center_x, center_y)

def interpolation(points):
    """Linear interpolation for defining sections
    Params
    -------
    points : darray, corner points for the ROI

    returns
    -------
    sections : dict, contains the corner coords for each section
    """
    # ROI coordinates
    topL_x, topL_y  = points[1]
    topR_x, topR_y = points[0]
    bottomL_x, bottomL_y  = points[2]
    bottomR_x, bottomR_y = points[3]    

    # Slope and width of top line
    top_slope = (topR_y - topL_y) / (topR_x - topL_x)
    top_width = topR_x - topL_x
    # Top Point for section 1 far right corner
    x1 = topL_x + (1/3) * top_width
    y1 = topL_y + (x1 - topL_x) * top_slope
    # Top Point for section 2 far right corner 
    x2 = topL_x + (2/3) * top_width
    y2 = topL_y + (x2 - topL_x) * top_slope
    #Top Point for section 3 far right corner
    x3, y3 = (topR_x, topR_y)

    # Slope and width of bottom line
    bottom_slope = (bottomR_y - bottomL_y) / (bottomR_x - bottomL_x)
    bottom_width = bottomR_x - bottomL_x
    # Bottom Point fo section 1 far right corner
    qx1 = bottomL_x + (1/3) * bottom_width
    qy1 = bottomL_y + (qx1 - bottomL_x) * bottom_slope
    # Bottom Point for section 2 far right corner
    qx2 = bottomL_x + (2/3) * bottom_width
    qy2 = bottomL_y + (qx2 - bottomL_x) * bottom_slope
    # Bottom point for section 3 far right corner
    qx3, qy3 = (bottomR_x, bottomR_y)

    sections = {
        "1": [[topL_x, topL_y, bottomL_x, bottomL_y], [x1,y1, qx1, qy1]],
        "2": [[x1, y1, qx1, qy1], [x2, y2, qx2, qy2]],
        "3": [[x2, y2, qx2, qy2], [topR_x, topR_y, bottomR_x, bottomR_y]]
        }
    
    return sections

class SectionCenters():
    
    def __init__(self):

        centers = {
            "1":[],
            "2":[],
            "3": []
        }
        self.centers = centers

    def get_section_centers(self, points):
        """Calculates ROI coordinates, finds sections coordinates, calculates section centers
        Params
        -------
        frame : ndarray, the input image/frame

        returns
        --------
        section_centers : list, the center coordinates for the three sections
        """
        ROIs = interpolation(points)
        keys = ROIs.keys()
        section_centers = []
        for key in keys:
            center = find_center(ROIs[key])
            section_centers.append(center)
        self.centers["3"] = section_centers[0] # ADJUST 1 and 3 based on camera perspective and ROS code
        self.centers["2"] = section_centers[1]
        self.centers["1"] = section_centers[2]
        with open("section_centers.json", "w") as f:
            json.dump(self.centers, f)

        return self.centers