
# Image alignment from: https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/
# Form OCR from: https://pyimagesearch.com/2020/09/07/ocr-a-document-form-or-invoice-with-tesseract-opencv-and-python/

# import the necessary packages
import numpy as np
import imutils
import cv2

from collections import namedtuple
import pytesseract
import argparse
import easyocr

reader = easyocr.Reader(['en'])

# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

def align_images(image, template, maxFeatures=500, keepPercent=0.2,
	debug=False):
	# convert both the input image and template to grayscale
	imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	# use ORB to detect keypoints and extract (binary) local
	# invariant features
	orb = cv2.ORB_create(maxFeatures)
	(kpsA, descsA) = orb.detectAndCompute(imageGray, None)
	(kpsB, descsB) = orb.detectAndCompute(templateGray, None)
	# match the features
	method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
	matcher = cv2.DescriptorMatcher_create(method)
	matches = matcher.match(descsA, descsB, None)

	# sort the matches by their distance (the smaller the distance,
	# the "more similar" the features are)
	matches = sorted(matches, key=lambda x:x.distance)
	# keep only the top matches
	keep = int(len(matches) * keepPercent)
	matches = matches[:keep]
	# check to see if we should visualize the matched keypoints
	if debug:
		matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
			matches, None)
		matchedVis = imutils.resize(matchedVis, width=1000)
		cv2.imshow("Matched Keypoints", matchedVis)
		cv2.waitKey(0)

	# allocate memory for the keypoints (x, y)-coordinates from the
	# top matches -- we'll use these coordinates to compute our
	# homography matrix
	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")
	# loop over the top matches
	for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images
		# map to each other
		ptsA[i] = kpsA[m.queryIdx].pt
		ptsB[i] = kpsB[m.trainIdx].pt

	# compute the homography matrix between the two sets of matched
	# points
	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
	# use the homography matrix to align the images
	(h, w) = template.shape[:2]
	aligned = cv2.warpPerspective(image, H, (w, h))
	# return the aligned image
	return aligned

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def ocr_form(image_path, template_path):
    # create a named tuple which we can use to create locations of the
    # input document which we wish to OCR
    OCRLocation = namedtuple("OCRLocation", ["id", "bbox",
        "filter_keywords"])
    # define the locations of each area of the document we wish to OCR
    OCR_LOCATIONS = [
        OCRLocation("car_num", (264, 291, 183, 172),
            []),
        OCRLocation("cm_A", (222, 892, 244, 154),
            []),
        OCRLocation("cm_B", (533, 888, 244, 135),
            []),
        OCRLocation("cm_C", (845, 896, 244, 140),
            []),
        OCRLocation("cm_D", (1170, 893, 244, 140),
            []),
        OCRLocation("cm_F", (1812, 893, 244, 140),
            []),
        OCRLocation("cm_G", (2136, 893, 244, 140),
            []),

        # OCRLocation("car_num", (135, 287, 306, 172),
        #     []),
        # OCRLocation("cm_A", (122, 892, 344, 154),
        #     []),

        # OCRLocation("car_num", (264, 291, 183, 172),
        #     []),
        # OCRLocation("step1_first_name", (265, 237, 751, 106),
        #     ["middle", "initial", "first", "name"]),
        # OCRLocation("step1_last_name", (1020, 237, 835, 106),
        #     ["last", "name"]),
        # OCRLocation("step1_address", (265, 336, 1588, 106),
        #     ["address"]),
        # OCRLocation("step1_city_state_zip", (265, 436, 1588, 106),
        #     ["city", "zip", "town", "state"]),
        # OCRLocation("step5_employee_signature", (319, 2516, 1487, 156),
        #     ["employee", "signature", "form", "valid", "unless",
        #         "you", "sign"]),
        # OCRLocation("step5_date", (1804, 2516, 504, 156), ["date"]),
        # OCRLocation("employee_name_address", (265, 2706, 1224, 180),
        #     ["employer", "name", "address"]),
        # OCRLocation("employee_ein", (1831, 2706, 448, 180),
        #     ["employer", "identification", "number", "ein"]),
    ]

    # load the input image and template from disk
    print("[INFO] loading images...")
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)
    # align the images
    print("[INFO] aligning images...")
    aligned = align_images(image, template)

    # initialize a results list to store the document OCR parsing results
    print("[INFO] OCR'ing document...")
    parsingResults = []
    # loop over the locations of the document we are going to OCR
    for loc in OCR_LOCATIONS:
        # extract the OCR ROI from the aligned image
        (x, y, w, h) = loc.bbox
        roi = aligned[y:y + h, x:x + w]
        # OCR the ROI using Tesseract
        # rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        thr = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)[1]
        res = cv2.GaussianBlur(thr, (5,5), 0)
        res = 255 - res
        cv2.imwrite('foo3.png', res) # debug
        #text = pytesseract.image_to_string(rgb, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        #text = pytesseract.image_to_string(res, config='--psm 7 -c tessedit_char_whitelist=0123456789')

        #pixel_values = processor(images=res, return_tensors="pt").pixel_values
        #generated_ids = model.generate(pixel_values)
        #text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        text = reader.readtext(res, allowlist='0123456789', detail=0)

        parsingResults.append((loc, text))

    return parsingResults


if __name__ == '__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image that we'll align to template")
    ap.add_argument("-t", "--template", required=True,
        help="path to input template image")
    args = vars(ap.parse_args())

    ocr_form(args.image, args.template)
