import numpy as np
import cv2
import os

class EASTDetector:
    def __init__(self, small_overlap = 0.1, large_overlap = 0.9, horizontal_padding=0, vertical_padding=50):
        assert small_overlap < large_overlap, "Small Overlap must be smaller than Large Overlap"
        assert large_overlap < 1 and small_overlap > 0, "Large and Small overlap must be between 0 and 1"

        self.small_overlap = small_overlap
        self.large_overlap = large_overlap
        self.vertical_padding = vertical_padding
        self.horizontal_padding = horizontal_padding

    def non_max_suppression_fast(self, boxes, overlapThresh, merge):
        """
        Modified Non-maximal Suppression to condense the number of bounding boxes.
        Modified from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

        Parameters:
        - Boxes: 2D numpy array. Each entry represents a bounding box (minX, minY, maxX, maxY)
        - Overlap Threshold: The percentage of area overlap required to consider two boxes the same
        - Merge: Suppress boxes into a larger, merged box (modified non-max-suppression) or into the left most box (traditional non-max-suppression)
        """
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes are integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes	
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        # Add one because coordinates are zero-indexed
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i] - 100, x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i] + 100, x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            
            overlapIdxs = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))

            # delete all overlapping indices from the list
            idxs = np.delete(idxs, overlapIdxs)
            
            if merge and overlapIdxs.size != 0:        
                nx1 = np.min(x1[overlapIdxs])
                ny1 = np.min(y1[overlapIdxs])
                nx2 = np.max(x2[overlapIdxs])
                ny2 = np.max(y2[overlapIdxs])

                boxes[i, :] = np.array([nx1, ny1, nx2, ny2])
            
        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")

    def detect(self, image):
        """
        Retrives bounding boxes for detected text in an image
        The bounding boxes returned by the detector are slightly enlarged to ensure they capture all of the text
        Modified from https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
        """
        (originalH, originalW) = image.shape[:2]
 
        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = ((originalW // 32) * 32, (originalH // 32) * 32)
        rW = originalW / float(newW)
        rH = originalH / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]
        
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        net.setInput(blob)

        (scores, geometry) = net.forward(layerNames)
        
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < 0.5:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                
                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
        
        boxes = self.non_max_suppression_fast(np.array(rects), self.small_overlap, True)
        boxes = self.non_max_suppression_fast(boxes, self.large_overlap, False)

        finalBoxes = []
        # loop over the bounding boxes
        v_pad = self.vertical_padding // 2
        h_pad = self.horizontal_padding // 2

        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective ratios
            startX = max(int(startX * rW) - h_pad, 0)
            startY = max(int(startY * rH) - v_pad, 0)
            endX = min(int(endX * rW) + h_pad, originalW)
            endY = min(int(endY * rH) + v_pad, originalH)
            finalBoxes.append((startX, startY, endX, endY))

        return finalBoxes
    
    def get_slices(self, image):
        boxes = self.detect(image)
        book_slices = []
        for startX, startY, endX, endY in boxes:
            book_slice = image[startY:endY, startX:endX, :]
            book_slices.append(book_slice)
            
        return book_slices