import numpy as np
from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
from sfft.utils.SFFTSolutionReader import Realize_MatchingKernel

try:
    import pyfftw.interfaces.numpy_fft as fft
except ImportError:
    from numpy import fft

__author__ = "ZOGY Scorr (Zackay, Ofek & Gal-Yam 2016) on sfft"
__version__ = "v0.4"

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _gaussian_kernel_2d(fwhm_px, half_size):
    sigma = fwhm_px * FWHM_TO_SIGMA
    y, x = np.mgrid[-half_size: half_size + 1, -half_size: half_size + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return g / g.sum()


def _embed_centered_then_shift(small, target_shape):
    big = np.zeros(target_shape, dtype=np.float32)
    nh, nw = small.shape
    cy, cx = target_shape[0] // 2, target_shape[1] // 2
    sy, sx = cy - nh // 2, cx - nw // 2
    if not (0 <= sy and sy + nh <= target_shape[0] and 0 <= sx and sx + nw <= target_shape[1]):
        big[sy: sy + nh, sx: sx + nw] = small
        return fft.fftshift(big)
    # NOTE == fftshift of the centred embedding (which moves every element by n // 2 along each axis,
    #      modulo n), placed directly instead of rolling the whole image twice
    rows = (np.arange(sy, sy + nh) + target_shape[0] // 2) % target_shape[0]
    cols = (np.arange(sx, sx + nw) + target_shape[1] // 2) % target_shape[1]
    big[np.ix_(rows, cols)] = small
    return big


def _embed_centered_then_shift_device(xp, small, target_shape):
    # _embed_centered_then_shift built on the device (xp = cupy): the same placement of the same float32 values
    big = xp.zeros(target_shape, dtype=np.float32)
    nh, nw = small.shape
    cy, cx = target_shape[0] // 2, target_shape[1] // 2
    sy, sx = cy - nh // 2, cx - nw // 2
    if not (0 <= sy and sy + nh <= target_shape[0] and 0 <= sx and sx + nw <= target_shape[1]):
        big[sy: sy + nh, sx: sx + nw] = xp.asarray(small)
        return xp.fft.fftshift(big)
    rows = xp.asarray((np.arange(sy, sy + nh) + target_shape[0] // 2) % target_shape[0])
    cols = xp.asarray((np.arange(sx, sx + nw) + target_shape[1] // 2) % target_shape[1])
    big[rows[:, None], cols[None, :]] = xp.asarray(np.asarray(small, dtype=np.float32))
    return big


class Scorr_Calculator:

    """Decorrelated ZOGY score image (Scorr) on an sfft difference image.

    The diff noise is whitened with a ZOGY decorrelation kernel, matched-filtered
    with the diff PSF, then divided by its background sigma so blank-sky pixels
    read ~N(0, 1). Threshold |Scorr| > k to find transients.
    """

    @staticmethod
    def build_diff_psf(K_sfft, fwhm_convd_side_px, image_shape, gaussian_half=None):
        """Diff PSF P_D = K_sfft (conv) Gaussian(fwhm_convd_side_px), unit sum."""
        if gaussian_half is None:
            gaussian_half = max(5, int(np.ceil(5.0 * fwhm_convd_side_px * FWHM_TO_SIGMA)))
        p_conv = _gaussian_kernel_2d(fwhm_convd_side_px, gaussian_half)
        K_big = _embed_centered_then_shift(K_sfft, image_shape)
        P_conv_big = _embed_centered_then_shift(p_conv, image_shape)
        try:
            import cupy as xp
            K_big, P_conv_big = xp.asarray(K_big), xp.asarray(P_conv_big)
            P_D = xp.real(xp.fft.ifft2(xp.fft.fft2(K_big) * xp.fft.fft2(P_conv_big)))
            s = P_D.sum()
            if s > 0:
                P_D = P_D / s
            return xp.asnumpy(P_D)
        except Exception:
            P_D = np.real(fft.ifft2(fft.fft2(K_big) * fft.fft2(P_conv_big)))
            s = P_D.sum()
            if s > 0:
                P_D /= s
            return P_D

    @staticmethod
    def compute_decorr_device(diff, K_sfft, fwhm_convd_side_px, skysig_sci, skysig_ref, conv_side="REF",
                              VERBOSE_LEVEL=0):
        """compute_decorr for a difference image already on the GPU (CuPy, N0 x N1, any layout): the same steps
        on the device, with the sky sigmas given, and S returned on the device. Only the [::7, ::7] sample of S
        that the S sky level is measured on comes back to the host."""
        import cupy as xp
        from sfft.utils.DeCorrelationCalculator import DeCorrelation_Calculator

        nan_mask = ~xp.isfinite(diff)
        diff_filled = xp.where(nan_mask, xp.float32(0), diff).astype(np.float32, copy=False)
        if diff.flags.f_contiguous and not diff.flags.c_contiguous:
            # NOTE in the image's memory order, as compute_decorr hands it to the FFT (the same transform values)
            diff_filled = xp.asfortranarray(diff_filled)
        if conv_side == "REF":
            skysig_conv, skysig_unconv = skysig_ref, skysig_sci
        else:
            skysig_conv, skysig_unconv = skysig_sci, skysig_ref
        KDeCo = DeCorrelation_Calculator.DCC(
            MK_JLst=[None], SkySig_JLst=[skysig_unconv],
            MK_ILst=[K_sfft], SkySig_ILst=[skysig_conv], MK_Fin=None,
            VERBOSE_LEVEL=VERBOSE_LEVEL)

        gaussian_half = max(5, int(np.ceil(5.0 * fwhm_convd_side_px * FWHM_TO_SIGMA)))
        p_conv = _gaussian_kernel_2d(fwhm_convd_side_px, gaussian_half)
        K_big = _embed_centered_then_shift_device(xp, K_sfft, diff.shape)
        P_conv_big = _embed_centered_then_shift_device(xp, p_conv, diff.shape)
        P_D = xp.real(xp.fft.ifft2(xp.fft.fft2(K_big) * xp.fft.fft2(P_conv_big)))
        del K_big, P_conv_big
        s = P_D.sum()
        if s > 0:
            P_D = P_D / s

        FKDECO = xp.fft.fft2(_embed_centered_then_shift_device(xp, KDeCo, diff.shape))
        FPSF = xp.fft.fft2(P_D) * FKDECO
        del P_D
        FdDIFF = xp.fft.fft2(diff_filled) * FKDECO
        del diff_filled, FKDECO
        S = xp.fft.ifft2(FdDIFF * xp.conj(FPSF)).real
        del FdDIFF, FPSF
        skysig_S = SkyLevel_Estimator.SLE(PixA_obj=xp.asnumpy(S[::7, ::7]))[1]
        if skysig_S > 0:
            S /= S.dtype.type(skysig_S)    # NOTE as numpy divides float32 S by the (float64) sky sigma
        if nan_mask.any():
            S = xp.where(nan_mask, xp.float32(np.nan), S)
        return S

    @staticmethod
    def sky_sigma(PixA):
        return SkyLevel_Estimator.SLE(
            PixA_obj=np.nan_to_num(np.asarray(PixA[::7, ::7], dtype=np.float64)))[1]

    @staticmethod
    def compute_decorr(diff, K_sfft, fwhm_convd_side_px, PixA_SCI, PixA_REF,
                       conv_side, VERBOSE_LEVEL=2, skysig_sci=None, skysig_ref=None):
        import logging as _lg
        import time as _tm
        from sfft.utils.DeCorrelationCalculator import DeCorrelation_Calculator

        try:
            import cupy as xp
            on_gpu = True
        except Exception:
            xp = np
            on_gpu = False

        _p = {}
        _prev = _tm.perf_counter()

        def _lap(name, sync=False):
            nonlocal _prev
            if sync and on_gpu:
                xp.cuda.Stream.null.synchronize()
            now = _tm.perf_counter()
            _p[name] = now - _prev
            _prev = now

        nan_mask = ~np.isfinite(diff)
        diff_filled = np.where(nan_mask, 0.0, diff).astype(np.float32, copy=False)
        _lap("prep")

        if skysig_sci is None:
            skysig_sci = Scorr_Calculator.sky_sigma(PixA_SCI)
        _lap("sle_sci")
        if skysig_ref is None:
            skysig_ref = Scorr_Calculator.sky_sigma(PixA_REF)
        _lap("sle_ref")
        if conv_side == "REF":
            skysig_conv, skysig_unconv = skysig_ref, skysig_sci
        else:
            skysig_conv, skysig_unconv = skysig_sci, skysig_ref

        KDeCo = DeCorrelation_Calculator.DCC(
            MK_JLst=[None], SkySig_JLst=[skysig_unconv],
            MK_ILst=[K_sfft], SkySig_ILst=[skysig_conv], MK_Fin=None,
            VERBOSE_LEVEL=VERBOSE_LEVEL)
        _lap("dcc")

        P_D = Scorr_Calculator.build_diff_psf(K_sfft, fwhm_convd_side_px, diff.shape)
        _lap("psf", sync=True)

        FKDECO = xp.fft.fft2(xp.asarray(_embed_centered_then_shift(KDeCo, diff.shape)))
        FPSF = xp.fft.fft2(xp.asarray(P_D)) * FKDECO
        FdDIFF = xp.fft.fft2(xp.asarray(diff_filled)) * FKDECO
        S = xp.fft.ifft2(FdDIFF * xp.conj(FPSF)).real
        S = xp.asnumpy(S) if on_gpu else S
        _lap("fft_main", sync=True)

        skysig_S = SkyLevel_Estimator.SLE(PixA_obj=S[::7, ::7])[1]
        _lap("sle_S")
        if skysig_S > 0:
            S /= skysig_S     # NOTE S is this function's own array
        if nan_mask.any():
            S = np.where(nan_mask, np.nan, S)
        _lap("norm")
        _p["total"] = sum(_p.values())
        _lg.getLogger("turbo_pipeline.gpu_service.scorr").info(
            "scorr_prof prep=%(prep).3f sle_sci=%(sle_sci).3f sle_ref=%(sle_ref).3f "
            "dcc=%(dcc).3f psf=%(psf).3f fft_main=%(fft_main).3f sle_S=%(sle_S).3f "
            "norm=%(norm).3f total=%(total).3f" % _p)
        return S, S

    @staticmethod
    def from_subtraction(PixA_DIFF, Solution, SFFTConfig0, ConvdSide, FWHM_ConvdSide,
                         PixA_REF, PixA_SCI, VERBOSE_LEVEL=2, **_ignored):
        """Realize the matching kernel from the sfft solution, then compute Scorr."""
        N0, N1 = SFFTConfig0['N0'], SFFTConfig0['N1']
        L0, L1 = SFFTConfig0['L0'], SFFTConfig0['L1']
        DK, Fpq = SFFTConfig0['DK'], SFFTConfig0['Fpq']
        XY_ctr = np.array([[N0 / 2.0, N1 / 2.0]]) + 0.5
        K = Realize_MatchingKernel(XY_q=XY_ctr).FromArray(Solution=Solution,
            N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq)[0]
        return Scorr_Calculator.compute_decorr(diff=PixA_DIFF, K_sfft=K,
            fwhm_convd_side_px=FWHM_ConvdSide, PixA_SCI=PixA_SCI, PixA_REF=PixA_REF,
            conv_side=ConvdSide, VERBOSE_LEVEL=VERBOSE_LEVEL)
