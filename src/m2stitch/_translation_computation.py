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


def multi_peak_max(PCM: FloatArray) -> Tuple[IntArray, IntArray, FloatArray]:
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
    row, col = np.unravel_index(np.argsort(PCM.ravel()), PCM.shape)
    vals: FloatArray = PCM[row[::-1], col[::-1]]
    return row[::-1], col[::-1], vals


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


def extract_overlap_subregion(image: NumArray, y: Int, x: Int) -> NumArray:
    """Extract the overlapping subregion of the image.

    Parameters
    ---------
    image : np.ndarray
        the image (the dimension must be 2)
    y : Int
        the y (second last dim.) position
    x : Int
        the x (last dim.) position
    Returns
    -------
    subimage : np.ndarray
        the extracted subimage
    """
    sizeY = image.shape[0]
    sizeX = image.shape[1]
    assert (np.abs(y) < sizeY) and (np.abs(x) < sizeX)
    # clip x to (0, size_Y)
    xstart = int(max(0, min(y, sizeY, key=int), key=int))
    # clip x+sizeY to (0, size_Y)
    xend = int(max(0, min(y + sizeY, sizeY, key=int), key=int))
    ystart = int(max(0, min(x, sizeX, key=int), key=int))
    yend = int(max(0, min(x + sizeX, sizeX, key=int), key=int))
    return image[xstart:xend, ystart:yend]


def interpret_translation(
    image1: NumArray,
    image2: npt.NDArray,
    yins: IntArray,
    xins: IntArray,
    ymin: Int,
    ymax: Int,
    xmin: Int,
    xmax: Int,
    n: Int = 2,
) -> Tuple[float, int, int]:
    """Interpret the translation to find the translation with heighest ncc.

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
    ymin : Int
        the minimum value of y (second last dim.)
    ymax : Int
        the maximum value of y (second last dim.)
    xmin : Int
        the minimum value of x (last dim.)
    xmax : Int
        the maximum value of x (last dim.)
    n : Int
        the number of peaks to check, default is 2.

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
    sizeY = image1.shape[0]
    sizeX = image1.shape[1]
    assert np.all(0 <= yins) and np.all(yins < sizeY)
    assert np.all(0 <= xins) and np.all(xins < sizeX)

    _ncc = -np.infty
    y = 0
    x = 0

    ymagss = [yins, sizeY - yins]
    ymagss[1][ymagss[0] == 0] = 0
    xmagss = [xins, sizeX - xins]
    xmagss[1][xmagss[0] == 0] = 0

    # concatenate all the candidates
    _poss = []
    for ymags, xmags, ysign, xsign in itertools.product(
        ymagss, xmagss, [-1, +1], [-1, +1]
    ):
        yvals = ymags * ysign
        xvals = xmags * xsign
        _poss.append([yvals, xvals])
    poss = np.array(_poss)
    valid_ind = (
        (ymin <= poss[:, 0, :])
        & (poss[:, 0, :] <= ymax)
        & (xmin <= poss[:, 1, :])
        & (poss[:, 1, :] <= xmax)
    )
    assert np.any(valid_ind)
    valid_ind = np.any(valid_ind, axis=0)
    for pos in np.moveaxis(poss[:, :, valid_ind], -1, 0)[: int(n)]:
        for yval, xval in pos:
            if (ymin <= yval) and (yval <= ymax) and (xmin <= xval) and (xval <= xmax):
                subI1 = extract_overlap_subregion(image1, yval, xval)
                subI2 = extract_overlap_subregion(image2, -yval, -xval)
                ncc_val = ncc(subI1, subI2)
                if ncc_val > _ncc:
                    _ncc = float(ncc_val)
                    y = int(yval)
                    x = int(xval)
    return _ncc, y, x
