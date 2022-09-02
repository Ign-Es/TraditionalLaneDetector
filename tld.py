import cv2
import numpy as np

class ColorRange:
    """
    The Color Range class holds one or multiple color ranges. It can easily be generated with
    the :py:meth:`fromDict` class method and extends the `OpenCV's inRange <https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981>`_
    method to work with multiple color ranges.
    All colours must be given in ``HSV`` space.
    Args:
        low (:obj:`numpy array`): An ``Nx3`` array with the low ends of ``N`` colour ranges.
        high (:obj:`numpy array`): An ``Nx3`` array with the high ends of ``N`` colour ranges.
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high

    @classmethod
    def fromDict(cls, dictionary):
        """
        Generates a :py:class:`ColorRange` object from a dictionary. Expects the colors to be given in ``HSV`` space.
        If multi-entry ranges are provided (e.g. if you are interested in yellow and white), then each should have
        keys ``high_X`` and ``low_X``, see example bellow:
        Examples:
            Single-entry color range::
                { 'low': [0,0,150], 'high': [180,60,255] }
            Multi-entry color range::
                { 'low_1': [0,0,150], 'high_1': [180,60,255], 'low_2': [165,140,100], 'high_2': [180,255,255] }
        Args:
            dictionary (:obj:`dict`): The yaml dictionary describing the color ranges.
        Returns:
            :obj:`ColorRange`: the generated ColorRange object
        """

        # if only two entries: single-entry, if more: multi-entry
        if len(dictionary) == 2:
            assert "low" in dictionary, "Key 'low' must be in dictionary"
            assert "high" in dictionary, "Key 'high' must be in dictionary"
            low = np.array(dictionary["low"]).reshape((1, 3))
            high = np.array(dictionary["high"]).reshape((1, 3))

        elif len(dictionary) % 2 == 0:

            # make the keys tuples with `low` or `high` and the id of the entry
            dictionary = {tuple(k.split("_")): v for k, v in list(dictionary.items())}
            entry_indices = set([k[1] for k, _ in list(dictionary.items())])

            assert len(entry_indices) == len(dictionary) / 2, (
                "The multi-entry definition doesn't " "follow the requirements"
            )

            # build an array for the low and an array for the high range bounds
            low = np.zeros((len(entry_indices), 3))
            high = np.zeros((len(entry_indices), 3))
            for idx, entry in enumerate(entry_indices):
                low[idx] = dictionary[("low", entry)]
                high[idx] = dictionary[("high", entry)]

        else:
            raise ValueError(
                "The input dictionary has two have an even number of "
                "entries: a low and high value for each color range."
            )

        return cls(low=low, high=high)

    def inRange(self, image):
        """
        Applies the `OpenCV inRange <https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981>`_
        method to every color range entry. Returns the bitwise OR of the results.
        In other words, returns a binary map with 1 for the pixels of the input image that fall in at least one of
        the color ranges.
        Args:
            image (:obj:`numpy array`): an ``HSV`` image
        Returns:
            :obj:`numpy array`: a two-dimensional binary map
        """

        selection = cv2.inRange(image, self.low[0], self.high[0])
        for i in range(1, len(self.low)):
            current = cv2.inRange(image, self.low[i], self.high[i])
            selection = cv2.bitwise_or(current, selection)

        return selection

    @property
    def representative(self):
        """
        Provides an representative color for this color range. This is the average color of the first range (if more
        than one ranges are set).
        Returns:
            :obj:`list`: a list with 3 entries representing an HSV color
        """

        return list(0.5 * (self.high[0] + self.low[0]).astype(int))

class Detections:
    """
    This is a data class that can be used to store the results of the line detection procedure performed
    by :py:class:`LineDetector`.
    """

    def __init__(self, lines, normals, centers, map):
        self.lines = lines  #: An ``Nx4`` array with every row representing a line ``[x1, y1, x2, y2]``
        self.normals = normals  #: An ``Nx2`` array with every row representing the normal of a line ``[nx,
        # ny]``

        self.centers = centers  #: An ``Nx2`` array with every row representing the center of a line ``[cx,
        # cy]``

        self.map = map  #: A binary map of the area from which the line segments were extracted


class LineDetector:
    """
    The Line Detector can be used to extract line segments from a particular color range in an image. It combines
    edge detection, color filtering, and line segment extraction.
    This class was created for the goal of extracting the white, yellow, and red lines in the Duckiebot's camera stream
    as part of the lane localization pipeline. It is setup in a way that allows efficient detection of line segments in
    different color ranges.
    In order to process an image, first the :py:meth:`setImage` method must be called. In makes an internal copy of the
    image, converts it to `HSV color space <https://en.wikipedia.org/wiki/HSL_and_HSV>`_, which is much better for
    color segmentation, and applies `Canny edge detection <https://en.wikipedia.org/wiki/Canny_edge_detector>`_.
    Then, to do the actual line segment extraction, a call to :py:meth:`detectLines` with a :py:class:`ColorRange`
    object must be made. Multiple such calls with different colour ranges can be made and these will reuse the
    precomputed HSV image and Canny edges.
    Args:
        canny_thresholds (:obj:`list` of :obj:`int`): a list with two entries that specify the thresholds for the hysteresis procedure, details `here <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny#canny>`__, default is ``[80, 200]``
        canny_aperture_size (:obj:`int`): aperture size for a Sobel operator, details `here <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny#canny>`__, default is 3
        dilation_kernel_size (:obj:`int`): kernel size for the dilation operation which fills in the gaps in the color filter result, default is 3
        hough_threshold (:obj:`int`): Accumulator threshold parameter. Only those lines are returned that get enough votes, details `here <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp#houghlinesp>`__, default is 2
        hough_min_line_length (:obj:`int`): Minimum line length. Line segments shorter than that are rejected, details `here <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp#houghlinesp>`__, default is 3
        hough_max_line_gap (:obj:`int`): Maximum allowed gap between points on the same line to link them, details `here <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp#houghlinesp>`__, default is 1
    """

    def __init__(
            self,
            canny_thresholds=[80, 200],
            canny_aperture_size=3,
            dilation_kernel_size=3,
            hough_threshold=2,
            hough_min_line_length=3,
            hough_max_line_gap=1,
    ):

        self.canny_thresholds = canny_thresholds
        self.canny_aperture_size = canny_aperture_size
        self.dilation_kernel_size = dilation_kernel_size
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap

        # initialize the variables that will hold the processed images
        self.bgr = np.empty(0)  #: Holds the ``BGR`` representation of an image
        self.hsv = np.empty(0)  #: Holds the ``HSV`` representation of an image
        self.canny_edges = np.empty(0)  #: Holds the Canny edges of an image

    def setImage(self, image):
        """
        Sets the :py:attr:`bgr` attribute to the provided image. Also stores
        an `HSV <https://en.wikipedia.org/wiki/HSL_and_HSV>`_ representation of the image and the
        extracted `Canny edges <https://en.wikipedia.org/wiki/Canny_edge_detector>`_. This is separated from
        :py:meth:`detectLines` so that the HSV representation and the edge extraction can be reused for multiple
        colors.
        Args:
            image (:obj:`numpy array`): input image
        """

        self.bgr = np.copy(image)
        self.hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.canny_edges = self.findEdges()

    def getImage(self):
        """
        Provides the image currently stored in the :py:attr:`bgr` attribute.
        Returns:
            :obj:`numpy array`: the stored image
        """
        return self.bgr

    def findEdges(self):
        """
        Applies `Canny edge detection <https://en.wikipedia.org/wiki/Canny_edge_detector>`_ to a ``BGR`` image.
        Returns:
            :obj:`numpy array`: a binary image with the edges
        """
        edges = cv2.Canny(
            self.bgr,
            self.canny_thresholds[0],
            self.canny_thresholds[1],
            apertureSize=self.canny_aperture_size,
        )
        return edges

    def houghLine(self, edges):
        """
        Finds line segments in a binary image using the probabilistic Hough transform. Based on the OpenCV function
        `HoughLinesP <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp
        #houghlinesp>`_.
        Args:
            edges (:obj:`numpy array`): binary image with edges
        Returns:
             :obj:`numpy array`: An ``Nx4`` array where each row represents a line ``[x1, y1, x2, y2]``. If no lines
             were detected, returns an empty list.
        """
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )
        if lines is not None:
            lines = lines.reshape((-1, 4))  # it has an extra dimension
        else:
            lines = []

        return lines

    def colorFilter(self, color_range):
        """
        Obtains the regions of the image that fall in the provided color range and the subset of the detected Canny
        edges which are in these regions. Applies a `dilation <https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm>`_
        operation to smooth and grow the regions map.
        Args:
            color_range (:py:class:`ColorRange`): A :py:class:`ColorRange` object specifying the desired colors.
        Returns:
            :obj:`numpy array`: binary image with the regions of the image that fall in the color range
            :obj:`numpy array`: binary image with the edges in the image that fall in the color range
        """
        # threshold colors in HSV space
        map = color_range.inRange(self.hsv)

        # binary dilation: fills in gaps and makes the detected regions grow
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.dilation_kernel_size, self.dilation_kernel_size)
        )
        map = cv2.dilate(map, kernel)

        # extract only the edges which come from the region with the selected color
        edge_color = cv2.bitwise_and(map, self.canny_edges)

        return map, edge_color

    def findNormal(self, map, lines):
        """
        Calculates the centers of the line segments and their normals.
        Args:
            map (:obj:`numpy array`):  binary image with the regions of the image that fall in a given color range
            lines (:obj:`numpy array`): An ``Nx4`` array where each row represents a line. If no lines were detected,
            returns an empty list.
        Returns:
            :obj:`tuple`: a tuple containing:
                 * :obj:`numpy array`: An ``Nx2`` array where each row represents the center point of a line. If no lines were detected returns an empty list.
                 * :obj:`numpy array`: An ``Nx2`` array where each row represents the normal of a line. If no lines were detected returns an empty list.
        """
        normals = []
        centers = []
        if len(lines) > 0:
            length = np.sum((lines[:, 0:2] - lines[:, 2:4]) ** 2, axis=1, keepdims=True) ** 0.5
            dx = 1.0 * (lines[:, 3:4] - lines[:, 1:2]) / length
            dy = 1.0 * (lines[:, 0:1] - lines[:, 2:3]) / length

            centers = np.hstack([(lines[:, 0:1] + lines[:, 2:3]) / 2, (lines[:, 1:2] + lines[:, 3:4]) / 2])
            x3 = (centers[:, 0:1] - 3.0 * dx).astype("int")
            y3 = (centers[:, 1:2] - 3.0 * dy).astype("int")
            x4 = (centers[:, 0:1] + 3.0 * dx).astype("int")
            y4 = (centers[:, 1:2] + 3.0 * dy).astype("int")

            np.clip(x3, 0, map.shape[1] - 1, out=x3)
            np.clip(y3, 0, map.shape[0] - 1, out=y3)
            np.clip(x4, 0, map.shape[1] - 1, out=x4)
            np.clip(y4, 0, map.shape[0] - 1, out=y4)

            flag_signs = (np.logical_and(map[y3, x3] > 0, map[y4, x4] == 0)).astype("int") * 2 - 1
            normals = np.hstack([dx, dy]) * flag_signs

        return centers, normals

    def detectLines(self, color_range):
        """
        Detects the line segments in the currently set image that occur in and the edges of the regions of the image
        that are within the provided colour ranges.
        Args:
            color_range (:py:class:`ColorRange`): A :py:class:`ColorRange` object specifying the desired colors.
        Returns:
            :py:class:`Detections`: A :py:class:`Detections` object with the map of regions containing the desired colors, and the detected lines, together with their center points and normals,
        """
        map, edge_color = self.colorFilter(color_range)
        lines = self.houghLine(edge_color)
        centers, normals = self.findNormal(map, lines)
        return Detections(lines=lines, normals=normals, map=map, centers=centers)

def plotSegments(image, detections):
    """
    Draws a set of line segment detections on an image.
    Args:
        image (:obj:`numpy array`): an image
        detections (`dict`): a dictionary that has keys :py:class:`ColorRange` and values :py:class:`Detection`
    Returns:
        :obj:`numpy array`: the image with the line segments drawn on top of it.
    """

    im = np.copy(image)

    for color_range, det in list(detections.items()):

        # convert HSV color to BGR
        c = color_range.representative
        c = np.uint8([[[c[0], c[1], c[2]]]])
        color = cv2.cvtColor(c, cv2.COLOR_HSV2BGR).squeeze().astype(int)
        # plot all detected line segments and their normals
        for i in range(len(det.normals)):
            center = det.centers[i]
            normal = det.normals[i]
            im = cv2.line(
                im,
                tuple(center.astype(int)),
                tuple((center + 10 * normal).astype(int)),
                color=(0, 0, 0),
                thickness=2,
            )
            # im = cv2.circle(im, (center[0], center[1]), radius=3, color=color, thickness=-1)
        for line in det.lines:
            im = cv2.line(im, (line[0], line[1]), (line[2], line[3]), color=(0, 0, 0), thickness=5)
            im = cv2.line(
                im, (line[0], line[1]), (line[2], line[3]), color=tuple([int(x) for x in color]), thickness=2
            )
    return im


def plotMaps(image, detections):
    """
    Draws a set of color filter maps (the part of the images falling in a given color range) on an image.
    Args:
        image (:obj:`numpy array`): an image
        detections (`dict`): a dictionary that has keys :py:class:`ColorRange` and values :py:class:`Detection`
    Returns:
        :obj:`numpy array`: the image with the line segments drawn on top of it.
    """

    im = np.copy(image)
    im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    color_map = np.zeros_like(im)

    for color_range, det in list(detections.items()):

        # convert HSV color to BGR
        c = color_range.representative
        c = np.uint8([[[c[0], c[1], c[2]]]])
        color = cv2.cvtColor(c, cv2.COLOR_HSV2BGR).squeeze().astype(int)
        color_map[np.where(det.map)] = color

    im = cv2.addWeighted(im, 0.3, color_map, 1 - 0.3, 0.0)

    return im