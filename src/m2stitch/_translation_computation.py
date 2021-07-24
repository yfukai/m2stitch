import itertools
from typing import Tuple

import numpy as np
import numpy.typing as npt

from ._typing_utils import Float
from ._typing_utils import FloatArray
from ._typing_utils import Int
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
    PCM: FloatArray, n: int = 2
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """Find the first to n th largest peaks in PCM.

    Parameters
    ---------
    PCM : np.ndarray
        the peak correlation matrix
    n : Int
        the number of the peaks


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
    vals = PCM[row[-n:][::-1], col[-n:][::-1]]
    return row[-n:][::-1], col[-n:][::-1], vals


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


def extract_overlap_subregion(image: NumArray, x: Int, y: Int) -> NumArray:
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
    W = image.shape[0]
    H = image.shape[1]
    assert (np.abs(x) < W) and (np.abs(y) < H)
    xstart = int(max(0, min(x, W, key=int), key=int))
    xend = int(max(0, min(x + W, W, key=int), key=int))
    ystart = int(max(0, min(y, H, key=int), key=int))
    yend = int(max(0, min(y + H, H, key=int), key=int))
    return image[xstart:xend, ystart:yend]


def interpret_translation(
    image1: NumArray, image2: npt.NDArray, xin: Int, yin: Int
) -> Tuple[float, int, int]:
    """Interpret the translation to find the translation with heighest ncc.

    Parameters
    ---------
    image1 : np.ndarray
        the first image (the dimension must be 2)
    image2 : np.ndarray
        the second image (the dimension must be 2)
    xin : Int
        the x position estimated by PCM
    yin : Int
        the y position estimated by PCM

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
    x = 0
    y = 0
    W = image1.shape[0]
    H = image1.shape[1]
    assert 0 <= xin and xin < W
    assert 0 <= yin and yin < H
    xmags = [xin, W - xin] if xin > 0 else [xin]
    ymags = [yin, H - yin] if yin > 0 else [yin]
    for xmag, ymag, xsign, ysign in itertools.product(xmags, ymags, [-1, +1], [-1, +1]):
        subI1 = extract_overlap_subregion(image1, (xmag * xsign), (ymag * ysign))
        subI2 = extract_overlap_subregion(image2, -(xmag * xsign), -(ymag * ysign))
        ncc_val = ncc(subI1, subI2)
        if ncc_val > _ncc:
            _ncc = float(ncc_val)
            x = int(xmag * xsign)
            y = int(ymag * ysign)
    return _ncc, x, y
