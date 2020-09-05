
import operator
import os

import numpy as np
from keras.models import model_from_json

import cv2


class Extractor:
    """Class to extract sudoku numbers from image
    """

    def __init__(self, image) -> None:
        self.image = cv2.imread(image, 0)
        self.image_processed = None
        self.cropped = None
        self.digits = []

        self.opencv_version = cv2.__version__.split('.')[0]

        self.loaded_model = None
        self.load_model()

    def load_model(self):
        json_file = open(os.path.join(
            'SudokuSolve', 'models', 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights(
            os.path.join('SudokuSolve', 'models', 'model.h5'))
        print("Loaded saved model from disk.")

    def find_corners_of_largest_polygon(self):
        """Finds the 4 extreme corners of the largest contour in the image."""
        self.image_processed = Extractor.pre_process_image(self.image)

        if self.opencv_version == '3':
            _, contours, _ = cv2.findContours(
                self.image_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv2.findContours(
                self.image_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        polygon = contours[0]

        bottom_right, _ = max(
            enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1]
                                     for pt in polygon]), key=operator.itemgetter(1))
        bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1]
                                        for pt in polygon]), key=operator.itemgetter(1))
        top_right, _ = max(enumerate([pt[0][0] - pt[0][1]
                                      for pt in polygon]), key=operator.itemgetter(1))

        return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

    def crop_and_warp(self, crop_rect):
        """Crops and warps a rectangular section from an image into a square of similar size."""

        # Rectangle described by top left, top right, bottom right and bottom left points
        top_left, top_right, bottom_right, bottom_left = crop_rect[
            0], crop_rect[1], crop_rect[2], crop_rect[3]

        # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
        src = np.array([top_left, top_right, bottom_right,
                        bottom_left], dtype='float32')

        # Get the longest side in the rectangle
        side = max([
            Extractor.distance_between(bottom_right, top_right),
            Extractor.distance_between(top_left, bottom_left),
            Extractor.distance_between(bottom_right, bottom_left),
            Extractor.distance_between(top_left, top_right)
        ])

        # Describe a square with side of the calculated length, this is the new perspective we want to warp to
        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1],
                        [0, side - 1]], dtype='float32')

        # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
        m = cv2.getPerspectiveTransform(src, dst)

        # Performs the transformation on the original image
        self.cropped = cv2.warpPerspective(
            self.image, m, (int(side), int(side)))

    def infer_grid(self):
        """Infers 81 cell grid from a square image."""
        squares = []
        side = self.cropped.shape[:1]
        side = side[0] / 9

        # Note that we swap j and i here so the rectangles are stored in the list reading left-right instead of top-down.
        for j in range(9):
            for i in range(9):
                p1 = (i * side, j * side)  # Top left corner of a bounding box
                # Bottom right corner of bounding box
                p2 = ((i + 1) * side, (j + 1) * side)
                squares.append((p1, p2))
        return squares

    def get_digits(self, squares, size=28):
        """Extracts digits from their cells and builds an array"""
        img = self.pre_process_image(self.cropped.copy(), skip_dilate=True)

        for square in squares:
            self.digits.append(Extractor.extract_digit(img, square, size))

    def show_digits(self, colour=255):
        """Shows list of 81 extracted digits in a grid format"""
        rows = []
        with_border = [cv2.copyMakeBorder(
            img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in self.digits]
        for i in range(9):
            row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
            rows.append(row)

        return np.concatenate(rows)

    def identify_number(self, image):
        image_resize = cv2.resize(image, (28, 28))
        # For input to model.predict_classes
        image_resize_2 = image_resize.reshape(1, 1, 28, 28)
        loaded_model_pred = self.loaded_model.predict_classes(
            image_resize_2, verbose=0)
        return loaded_model_pred[0]

    def extract_number(self, sudoku):
        sudoku = cv2.resize(sudoku, (450, 450))
        # split sudoku
        grid = np.zeros([9, 9])
        for i in range(9):
            for j in range(9):
                image = sudoku[i*50:(i+1)*50, j*50:(j+1)*50]
                if image.sum() > 25000:
                    grid[i][j] = self.identify_number(image)
                else:
                    grid[i][j] = 0
        return grid.astype(int)

    @staticmethod
    def pre_process_image(image, skip_dilate=False):
        """Uses a blurring function, adaptive thresholding 
           and dilation to expose the main features of an image.

        Args:
            skip_dilate (bool, optional): Defaults to False.
        """
        proc = cv2.GaussianBlur(image.copy(), (9, 9), 0)

        proc = cv2.adaptiveThreshold(
            proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        image_processed = cv2.bitwise_not(proc, proc)

        if not skip_dilate:
            kernel = np.array(
                [[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
            image_processed = cv2.dilate(image_processed, kernel)

        return image_processed

    @staticmethod
    def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
        """
        Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
        connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
        """
        img = inp_img.copy()  # Copy the image, leaving the original untouched
        height, width = img.shape[:2]

        max_area = 0
        seed_point = (None, None)

        if scan_tl is None:
            scan_tl = [0, 0]

        if scan_br is None:
            scan_br = [width, height]

        # Loop through the image
        for x in range(scan_tl[0], scan_br[0]):
            for y in range(scan_tl[1], scan_br[1]):
                # Only operate on light or white squares
                # Note that .item() appears to take input as y, x
                if img.item(y, x) == 255 and x < width and y < height:
                    area = cv2.floodFill(img, None, (x, y), 64)
                    if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                        max_area = area[0]
                        seed_point = (x, y)

        # Colour everything grey (compensates for features outside of our middle scanning range
        for x in range(width):
            for y in range(height):
                if img.item(y, x) == 255 and x < width and y < height:
                    cv2.floodFill(img, None, (x, y), 64)

        # Mask that is 2 pixels bigger than the image
        mask = np.zeros((height + 2, width + 2), np.uint8)

        # Highlight the main feature
        if all([p is not None for p in seed_point]):
            cv2.floodFill(img, mask, seed_point, 255)

        top, bottom, left, right = height, 0, width, 0

        for x in range(width):
            for y in range(height):
                if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                    cv2.floodFill(img, mask, (x, y), 0)

                # Find the bounding parameters
                if img.item(y, x) == 255:
                    top = y if y < top else top
                    bottom = y if y > bottom else bottom
                    left = x if x < left else left
                    right = x if x > right else right

        bbox = [[left, top], [right, bottom]]
        return img, np.array(bbox, dtype='float32'), seed_point

    @staticmethod
    def cut_from_rect(img, rect):
        """Cuts a rectangle from an image using the top left and bottom right points."""
        return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

    @staticmethod
    def scale_and_centre(img, size, margin=0, background=0):
        """Scales and centres an image onto a new background square."""
        h, w = img.shape[:2]

        def centre_pad(length):
            """Handles centering for a given length that may be odd or even."""
            if length % 2 == 0:
                side1 = int((size - length) / 2)
                side2 = side1
            else:
                side1 = int((size - length) / 2)
                side2 = side1 + 1
            return side1, side2

        def scale(r, x):
            return int(r * x)

        if h > w:
            t_pad = int(margin / 2)
            b_pad = t_pad
            ratio = (size - margin) / h
            w, h = scale(ratio, w), scale(ratio, h)
            l_pad, r_pad = centre_pad(w)
        else:
            l_pad = int(margin / 2)
            r_pad = l_pad
            ratio = (size - margin) / w
            w, h = scale(ratio, w), scale(ratio, h)
            t_pad, b_pad = centre_pad(h)

        img = cv2.resize(img, (w, h))
        img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad,
                                 r_pad, cv2.BORDER_CONSTANT, None, background)
        return cv2.resize(img, (size, size))

    @staticmethod
    def extract_digit(img, rect, size):
        """Extracts a digit (if one exists) from a Sudoku square."""

        # Get the digit box from the whole square
        digit = Extractor.cut_from_rect(img, rect)

        # Use fill feature finding to get the largest feature in middle of the box
        # Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
        h, w = digit.shape[:2]
        margin = int(np.mean([h, w]) / 2.5)
        _, bbox, _ = Extractor.find_largest_feature(
            digit, [margin, margin], [w - margin, h - margin])
        digit = Extractor.cut_from_rect(digit, bbox)

        # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
        w = bbox[1][0] - bbox[0][0]
        h = bbox[1][1] - bbox[0][1]

        # Ignore any small bounding boxes
        if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
            return Extractor.scale_and_centre(digit, size, 4)
        else:
            return np.zeros((size, size), np.uint8)

    @staticmethod
    def distance_between(p1, p2):
        """Returns the scalar distance between two points"""
        difference_x = p2[0] - p1[0]
        difference_y = p2[1] - p1[1]
        return np.sqrt((difference_x ** 2) + (difference_y ** 2))

    def get_sudoku(self):
        corners = self.find_corners_of_largest_polygon()
        self.crop_and_warp(corners)

        squares = self.infer_grid()
        self.get_digits(squares)

        final_image = self.show_digits()

        return self.extract_number(final_image)


def show_image(img):
    """Shows an image until any key is pressed"""
    print(type(img))
    print(img.shape)
    cv2.imshow('image', img)  # Display the image
    # Wait for any key to be pressed (with the image window active)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Close all windows
