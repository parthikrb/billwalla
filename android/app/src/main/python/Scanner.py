from imutils.perspective import four_point_transform
import cv2
from pathlib import Path
import os

def Scanner(img_file):
    height = 800
    width = 600
    img = cv2.imread(img_file)
    img = cv2.resize(img, (width, height))
    orig_img = img.copy()

    # preprocess the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(img, 75, 200)

    # find and sort the contours
    contours, _ = cv2.findContours(
        edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # go through each contour
    for contour in contours:
        # approximate each contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        # check if we have found our document
        if len(approx) == 4:
            doc_cnts = approx
            break

    # apply warp perspective to get the top-down view
    warped = four_point_transform(orig_img, doc_cnts.reshape(4, 2))
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    final_img = cv2.resize(warped, (600, 800))
    return final_img

# ...
#

# valid_formats = [".jpg", ".jpeg", ".png"]
# get_text = lambda f: os.path.splitext(f)[1].lower()

# img_files = ['input/' + f for f in os.listdir('input') if get_text(f) in valid_formats]
# # create a new folder that will contain our images
# Path("output").mkdir(exist_ok=True)

# # go through each image file
# for img_file in img_files:
#     # read, resize, and make a copy of the image
#     img = cv2.imread(img_file)
#     img = cv2.resize(img, (width, height))
#     orig_img = img.copy()

#     # preprocess the image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(img, 75, 200)

#     # find and sort the contours
#     contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     # go through each contour
#     for contour in contours:
#         # approximate each contour
#         peri = cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
#         # check if we have found our document
#         if len(approx) == 4:
#             doc_cnts = approx
#             break

#     # apply warp perspective to get the top-down view
#     warped = four_point_transform(orig_img, doc_cnts.reshape(4, 2))
#     warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#     final_img = cv2.resize(warped, (600, 800))

#     # write the image in the ouput directory
#     cv2.imwrite("output" + "/" + os.path.basename(img_file), final_img)
