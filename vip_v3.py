import numpy as np
import json

def centroid(xyxy):
    """Centroids from xyxy"""
    x1, y1, x2, y2 = xyxy
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return center_x, center_y


def euclidean_dist(x1y1, x2y2):
    """Euclidean distance"""
    distance = np.linalg.norm(np.array(x1y1) - np.array(x2y2)) 

    return distance


def bbox_overlap(bbox1, bbox2):
    """Intersection over Union (IoU) of two bounding boxes"""
    x1, y1, x2, y2 = bbox1
    x1_p, y1_p, x2_p, y2_p = bbox2
    
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = bbox1_area + bbox2_area - inter_area
    
    return inter_area / union_area if union_area != 0 else 0

#------------------------------------VIP history---------------------------------------------------------------------------#

class VIPHist:
    """Finding, storing, and updating information on VIP"""

    def __init__(self, vip_center_thresh=50, vip_iou_thresh=0.6):

        with open("section_centers.json", 'r') as f:
            section_centers = json.load(f)
        self.vip = []
        self.potential_vip = []
        self.prev_ball = []
        self.void_ids = []
        self.center_thresh = vip_center_thresh
        self.iou_thresh = vip_iou_thresh
        self.section_centers = section_centers
    
    def get_vip(self, ball_info, cups_list):

        ball_center = centroid(ball_info[:4]) # Compute ball's bbox centroid
        og_num_cups = len(cups_list)
        for c, cup in enumerate(cups_list):
            cup_center = centroid(cup[:4]) # Compute cup's center
            centroid_distance = euclidean_dist(ball_center, cup_center) # Distance
            if centroid_distance < self.center_thresh: # Check for VIP
                print(f"\n\n\nPossible VIP found...\n\n\n")
                self.vip = [cup] # Assign VIP

                print(f"\n\n\nVIP cup ID {cup[8]} : Updated Void IDS {self.void_ids}\n\n\n")
                print(f"Original number of cups {og_num_cups} : Num cups after VIP found {len(cups_list)}\n\n\n")

                break

    def get_potential(self, det, vip_cent, potential_thresh=50):
        det_center = centroid(det[:4])
        euc_dist = euclidean_dist(vip_cent, det_center)
        if euc_dist < potential_thresh:
            self.potential_vip.append(det)  
        print(f"\n\nPotential VIPS {self.potential_vip}\n\n")
    

    def update_history(self, dets, filename="vip_info.txt"):

        """The workhorse function in this module
        Parameters
        ----------
        dets : ndarray, the detections [top_left_x, top_left_y, bottom_right_x, bottom_right_y, class, velocity(x), velocity(y), scale_change_rate, track_ID]

        Returns (for now at least, though it isn't necessary to return anythin)
        ---------
        vip : ndarray, contains the information about the VIP
        void_ids : list, other ids that are present
        """

        # Set all present IDs as void (can't be used once they disappear, we will handle the VIP ID later if found)
        cups = []
        for detection in dets:
            if detection[4] == 0:
                self.prev_ball = [detection] # Store ball if found
            elif detection[4] == 1: # Store cups
                cups.append(detection)

        # Base case, no VIP found, if we have a ball or ball history check distances
        if len(self.vip) == 0:
            if len(self.prev_ball) == 1:
                ball = self.prev_ball[0]
                self.get_vip(ball, cups) # Check for ball occlusion, assign VIP if present
            else:
                print("\n Ball has not yet been detected.\n") # This should not appear unless the ball has never been seen in the run

        # Case where we have a VIP in the previous frame
        elif len(self.vip) == 1:
            num_potential = 0
            prev_vip_id = self.vip[0][8]
            if 0 in dets[:,4]: # Make sure ball isn't redetected
                self.vip = []
                print(f"\n\n\nBALL REDETECTED, WIPING VIP HIST\n\n\n")
                return self.vip
            
            if prev_vip_id not in dets[:,8]:
                print(f"\n\n\nPrevious VIP ID LOST... PreviousID {prev_vip_id} : IDs {dets[:,8]}")
                print(f"\nPrevious VIP ID {prev_vip_id} : Updated void IDs {self.void_ids}\n\n\n")
                vip_center = centroid(self.vip[0][:4])
                for detection in dets:
                    if detection[8] in self.void_ids: # Skip detection if its ID is in void IDs
                        print(f"\n\n\nThis detection's ID {detection[8]} found in Void IDs {self.void_ids}\n\n\n")
                        continue
                    else:
                        print(f"\n\n\nThis detection's ID {detection[8]} not found in Void IDs {self.void_ids}\n\n\n")
                        num_potential += 1
                        self.get_potential(detection, vip_center)

                if len(self.potential_vip) == 1:
                    self.vip = self.potential_vip
                    self.potential_vip = []

            elif prev_vip_id in dets[:,8]:
                print(f"\n\n\nPrevious VIP ID {prev_vip_id} found in detections {dets[:,8]}, updating history...\n\n\n")
                for detection in dets:
                    if detection[8] == prev_vip_id: # Check for detection of previous frame's VIP
                        self.vip = [detection]
                        self.potential_vip = []
                        print(f"\n\n\nVIP {self.vip}\n\n\n")

                        return self.vip  
        else:
            raise ValueError(f"len(vip) is {len(self.vip)}, but should be either 0 or 1")

        return self.vip

