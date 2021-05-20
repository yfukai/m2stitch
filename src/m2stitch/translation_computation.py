import itertools

import numpy as np


def pcm(image1, image2):
    """
    compute peak correlation matrix for two images

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
    return np.fft.ifft2(FC / np.abs(FC)).real


def multi_peak_max(PCM, n=2):
    """
    find the first to n th largest peaks in PCM

    Parameters
    ---------
    PCM : np.ndarray
        the peak correlation matrix
    n : int
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
    return row[-n:][::-1], col[-n:][::-1], PCM[row[-n:][::-1], col[-n:][::-1]]


def ncc(image1, image2):
    """
    compute the normalized cross correlation for two images

    Parameters
    ---------
    image1 : np.ndarray
        the first image (the dimension must be 2)

    image2 : np.ndarray
        the second image (the dimension must be 2)

    Returns
    -------
    ncc : float
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


def extract_overlap_subregion(image, x, y):
    W = image.shape[0]
    H = image.shape[1]
    assert (np.abs(x) < W) and (np.abs(y) < H)
    xstart = max(0, min(x, W))
    xend = max(0, min(x + W, W))
    ystart = max(0, min(y, H))
    yend = max(0, min(y + H, H))
    return image[xstart:xend, ystart:yend]


def interpret_translation(image1, image2, xin, yin):
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
            _ncc = ncc_val
            x = xmag * xsign
            y = ymag * ysign
    return _ncc, x, y
