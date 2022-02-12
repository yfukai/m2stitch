import itertools
from typing import Tuple

import numpy as np
import numpy.typing as npt

from ._typing_utils import Float
from ._typing_utils import FloatArray
from ._typing_utils import Int
from ._typing_utils import IntArray
from ._typing_utils import NumArray


def pcm(image1: NumArray, image2: NumArray) -> FloatArray:
    """Compute peak correlation matrix for two images.

    Parameters
    ---------
    image1 : np.ndarray
        the first image (the dimension must be 2)

    image2 : np.ndarray
        the second image (the dimension must be 2)

    Returns
    -------
    PCM : np.ndarray
        the peak correlation matrix
    """
    assert image1.ndim == 2
    assert image2.ndim == 2
    assert np.array_equal(image1.shape, image2.shape)
    F1 = np.fft.fft2(image1)
    F2 = np.fft.fft2(image2)
    FC = F1 * np.conjugate(F2)
    return np.fft.ifft2(FC / np.abs(FC)).real.astype(np.float32)


def multi_peak_max(
    PCM: FloatArray,
) -> Tuple[IntArray, IntArray, FloatArray]:
    """Find the first to n th largest peaks in PCM.

    Parameters
    ---------
    PCM : np.ndarray
        the peak correlation matrix

    Returns
    -------
    rows : np.ndarray
        the row indices for the peaks
    cols : np.ndarray
        the column indices for the peaks
    vals : np.ndarray
        the values of the peaks
    """
    ys, xs = np.unravel_index(np.argsort(PCM.ravel()), PCM.shape)
    vals: FloatArray = PCM[ys[::-1], xs[::-1]]
    return ys[::-1], xs[::-1], vals


def ncc(image1: NumArray, image2: NumArray) -> Float:
    """Compute the normalized cross correlation for two images.

    Parameters
    ---------
    image1 : np.ndarray
        the first image (the dimension must be 2)

    image2 : np.ndarray
        the second image (the dimension must be 2)

    Returns
    -------
    ncc : Float
        the normalized cross correlation
    """
    assert image1.ndim == 2
    assert image2.ndim == 2
    assert np.array_equal(image1.shape, image2.shape)
    image1 = image1.flatten()
    image2 = image2.flatten()
    n = np.dot(image1 - np.mean(image1), image2 - np.mean(image2))
    d = np.linalg.norm(image1) * np.linalg.norm(image2)
    return n / d


def extract_overlap_subregion(image: NumArray, _y: Int, _x: Int) -> NumArray:
    """Extract the overlapping subregion of the image.

    Parameters
    ---------
    image : np.ndarray
        the image (the dimension must be 2)
    x : Int
        the x position
    y : Int
        the y position
    Returns
    -------
    subimage : np.ndarray
        the extracted subimage
    """
    sizeY = image.shape[0]
    sizeX = image.shape[1]
    assert (np.abs(_y) < sizeY) and (np.abs(_x) < sizeX)
    # clip x to (0, size_Y)
    xstart = int(max(0, min(_y, sizeY, key=int), key=int))
    # clip x+sizeY to (0, size_Y)
    xend = int(max(0, min(_y + sizeY, sizeY, key=int), key=int))
    ystart = int(max(0, min(_x, sizeX, key=int), key=int))
    yend = int(max(0, min(_x + sizeX, sizeX, key=int), key=int))
    return image[xstart:xend, ystart:yend]


def interpret_translation(
    image1: NumArray,
    image2: npt.NDArray,
    yins: IntArray,
    xins: IntArray,
    y_min: Int,
    y_max: Int,
    x_min: Int,
    x_max: Int,
    n: int = 2,
) -> Tuple[float, int, int]:
    """Interpret the translation to find the translation with heighest ncc.
    The candidates are ... (xin, sizeX-xin) * (+1,-1)

    Parameters
    ---------
    image1 : np.ndarray
        the first image (the dimension must be 2)
    image2 : np.ndarray
        the second image (the dimension must be 2)
    yins : IntArray
        the y positions estimated by PCM
    xins : IntArray
        the x positions estimated by PCM
    y_min : Int
        the minimum arrowed y (second last dim.) position of the peak
    y_max : Int
        the maximum arrowed y (second last dim.) position of the peak
    x_min : Int
        the minimum arrowed x (last dim.) position of the peak
    x_max : Int
        the maximum arrowed x (last dim.) position of the peak
    n : Int
        the number of the valid peaks to test

    Returns
    -------
    _ncc : Float
        the highest ncc
    x : Int
        the selected x position
    y : Int
        the selected y position
    """
    assert image1.ndim == 2
    assert image2.ndim == 2
    assert np.array_equal(image1.shape, image2.shape)
    _ncc = -np.infty
    _y = 0
    _x = 0
    sizeY = image1.shape[0]
    sizeX = image1.shape[1]
    peak_counts = 0

    for yin, xin in zip(yins, xins):
        peak_counted = False
        assert 0 <= yin and yin < sizeY
        assert 0 <= xin and xin < sizeX
        ymags = [yin, sizeY - yin] if yin > 0 else [yin]
        xmags = [xin, sizeX - xin] if xin > 0 else [xin]
        for ymag, xmag, ysign, xsign in itertools.product(
            ymags, xmags, [-1, +1], [-1, +1]
        ):
            yval = ymag * ysign
            xval = xmag * xsign
            if (y_min <= yval) & (yval <= y_max) & (x_min <= xval) & (xval <= x_max):
                peak_counted = True
                subI1 = extract_overlap_subregion(
                    image1, (ymag * ysign), (xmag * xsign)
                )
                subI2 = extract_overlap_subregion(
                    image2, -(ymag * ysign), -(xmag * xsign)
                )
                ncc_val = ncc(subI1, subI2)
                if ncc_val > _ncc:
                    _ncc = float(ncc_val)
                    _y = int(ymag * ysign)
                    _x = int(xmag * ysign)
        if peak_counted:
            peak_counts += 1
            if peak_counts >= n:
                break
    return _ncc, _y, _x
