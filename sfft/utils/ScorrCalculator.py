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
    big[sy: sy + nh, sx: sx + nw] = small
    return fft.fftshift(big)


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
    def compute_decorr(diff, K_sfft, fwhm_convd_side_px, PixA_SCI, PixA_REF,
                       conv_side, VERBOSE_LEVEL=2):
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
        diff_filled = np.where(nan_mask, 0.0, diff).astype(np.float32)
        _lap("prep")

        skysig_sci = SkyLevel_Estimator.SLE(
            PixA_obj=np.nan_to_num(np.asarray(PixA_SCI[::3, ::3], dtype=np.float64)))[1]
        _lap("sle_sci")
        skysig_ref = SkyLevel_Estimator.SLE(
            PixA_obj=np.nan_to_num(np.asarray(PixA_REF[::3, ::3], dtype=np.float64)))[1]
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

        skysig_S = SkyLevel_Estimator.SLE(PixA_obj=S[::3, ::3])[1]
        _lap("sle_S")
        if skysig_S > 0:
            S = S / skysig_S
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
