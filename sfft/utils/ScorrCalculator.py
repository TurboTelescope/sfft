import numpy as np
from astropy.io import fits
from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
from sfft.utils.SFFTSolutionReader import Realize_MatchingKernel
# version: Jun 14, 2026

try:
    import pyfftw.interfaces.numpy_fft as fft
except ImportError:
    from numpy import fft

__author__ = "Lei Hu <leihu@andrew.cmu.edu> (sfft); ZOGY Scorr (Zackay, Ofek & Gal-Yam 2016)"
__version__ = "v0.3"

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _gaussian_kernel_2d(fwhm_px, half_size):
    """Unit-sum 2D Gaussian kernel of shape (2*half_size+1, 2*half_size+1)."""
    sigma = fwhm_px * FWHM_TO_SIGMA
    y, x = np.mgrid[-half_size: half_size + 1, -half_size: half_size + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return g / g.sum()


def _embed_centered_then_shift(small, target_shape):
    """Place a small kernel at the center of a zero-padded array and fftshift to origin.

    The output is suitable for direct FFT -- its peak ends up at (0, 0), so the
    convolution kernel does not introduce a positional shift.
    """
    big = np.zeros(target_shape, dtype=np.float64)
    nh, nw = small.shape
    cy, cx = target_shape[0] // 2, target_shape[1] // 2
    sy, sx = cy - nh // 2, cx - nw // 2
    big[sy: sy + nh, sx: sx + nw] = small
    return fft.fftshift(big)


class Scorr_Calculator:

    """
    # * Purpose
    #   Build a ZOGY corrected score image (Scorr) on top of an sfft difference
    #   image. Scorr is the matched-filter detection statistic expressed in units
    #   of sigma -- threshold it directly (e.g. |Scorr| > 5) to find transients.
    #
    #   This is the FULL ZOGY Scorr (Zackay, Ofek & Gal-Yam 2016, ApJ 830:27):
    #       S      = P_D (cross-correlate) D                  [Eq. 17, score image]
    #       V_S    = V_S_N + V_S_R + V_ast                    [Eqs. 26-32, per-pixel variance]
    #       Scorr  = S / sqrt(V_S)
    #   Unlike a background-limited matched filter, the per-pixel variance carries
    #   the source-shot-noise and astrometric-noise terms, so Scorr correctly
    #   suppresses bright-source residuals and dipoles from astrometric jitter.
    #
    # * The three ingredients, and how this wrapper gets them
    #   1. Matching kernel K        -- computed internally by sfft (Realize_MatchingKernel).
    #   2. Difference-image PSF P_D -- sfft is PSF-free, so P_D is reconstructed as
    #                                  P_D = K (conv) Gaussian(FWHM of the convolved side).
    #                                  K already absorbs any non-Gaussian features of the
    #                                  real convolved-side PSF (it is fit to the pixel data),
    #                                  so a Gaussian for the convolved side is the right input.
    #   3. Per-pixel noise          -- propagated analytically from sigma maps of the two
    #                                  inputs, sigma = sqrt(max(counts - sky, 0)/gain + sky_sig**2).
    #
    # * Which side is which (DIFF 'CONVD' keyword), with N == SCI, R == REF
    #       ConvdSide == 'REF'  ->  D = N - K(conv)R     ->  k_n = P_D,        k_r = K(conv)P_D
    #       ConvdSide == 'SCI'  ->  D = K(conv)N - R     ->  k_n = K(conv)P_D, k_r = P_D
    #   FWHM of the CONVOLVED side feeds P_D = K (conv) Gaussian(FWHM_convolved) ~ PSF of the
    #   unconvolved side, which is the profile of a real point source in D.
    #
    # Reference: Zackay, Ofek & Gal-Yam 2016, ApJ 830:27 ("Proper Image Subtraction").
    """

    @staticmethod
    def build_diff_psf(K_sfft, fwhm_convd_side_px, image_shape, gaussian_half=None):
        """Construct the diff PSF P_D = K_sfft (conv) Gaussian(fwhm_convd_side_px).

        Returns a 2D ndarray of `image_shape`, fftshifted to the origin (so its FFT
        has the canonical "kernel" phase). Sum is normalized to 1.
        """
        if gaussian_half is None:
            gaussian_half = max(5, int(np.ceil(5.0 * fwhm_convd_side_px * FWHM_TO_SIGMA)))
        p_conv = _gaussian_kernel_2d(fwhm_convd_side_px, gaussian_half)
        K_big = _embed_centered_then_shift(K_sfft, image_shape)
        P_conv_big = _embed_centered_then_shift(p_conv, image_shape)
        P_D = np.real(fft.ifft2(fft.fft2(K_big) * fft.fft2(P_conv_big)))
        s = P_D.sum()
        if s > 0:
            P_D /= s
        return P_D

    @staticmethod
    def compute(diff, K_sfft, fwhm_convd_side_px, sigma_map_sci, sigma_map_ref, \
        conv_side='REF', dx=0.0, dy=0.0, use_strict_v_ast=False, renormalize=True, \
        sci_image=None, ref_image=None, VERBOSE_LEVEL=2):

        """Compute the ZOGY Scorr from sfft outputs.

        # * Inputs
        #   -diff                          # sfft difference image D. NaN-safe (NaN pixels propagate to S/Scorr).
        #   -K_sfft                        # the realized sfft matching kernel (2D array).
        #   -fwhm_convd_side_px            # FWHM (px) of the side sfft convolved (FWHM_REF if conv_side='REF').
        #   -sigma_map_sci, -sigma_map_ref # per-pixel sigma (NOT variance) of science / reference, diff shape.
        #   -conv_side ['REF'/'SCI']       # which side sfft convolved (DIFF 'CONVD' keyword).
        #   -dx, -dy [0.0]                 # astrometric uncertainty (1-sigma) in pixels.
        #   -use_strict_v_ast [False]      # strict ZOGY V_ast (4 extra image-size FFTs) vs the
        #                                  #   gradient-of-S shortcut (much faster, negligible loss).
        #   -sci_image, -ref_image [None]  # required only if use_strict_v_ast=True.
        #   -renormalize [True]            # rescale Scorr by its empirical MAD-sigma noise floor so the
        #                                  #   noise reads exactly N(0, 1), absorbing residual noise-model
        #                                  #   mismatch (pixel correlation, background residuals, ...).
        #
        # * Returns
        #   PixA_SCORE                     # matched-filter signal S
        #   PixA_SCORR                     # corrected score Scorr (each pixel ~ N(0, 1) under the null)
        """

        diff = np.asarray(diff, dtype=np.float64)
        H, W = diff.shape

        if sigma_map_sci.shape != (H, W) or sigma_map_ref.shape != (H, W):
            raise Exception("MeLOn ERROR: sigma maps must match diff shape %s; got "
                            "sci=%s, ref=%s." % ((H, W), sigma_map_sci.shape, sigma_map_ref.shape))
        if conv_side not in ('REF', 'SCI'):
            raise Exception("MeLOn ERROR: conv_side must be 'REF' or 'SCI'; got %r." % conv_side)

        nan_mask = ~np.isfinite(diff)
        diff_filled = np.where(nan_mask, 0.0, diff)
        sigma_n = np.where(np.isfinite(sigma_map_sci), sigma_map_sci, 0.0).astype(np.float64)
        sigma_r = np.where(np.isfinite(sigma_map_ref), sigma_map_ref, 0.0).astype(np.float64)

        # Diff PSF: P_D = K (conv) Gaussian(FWHM of convolved side)
        P_D = Scorr_Calculator.build_diff_psf(K_sfft, fwhm_convd_side_px, diff.shape)
        P_D_hat = fft.fft2(P_D)
        K_big = _embed_centered_then_shift(K_sfft, diff.shape)
        K_hat = fft.fft2(K_big)

        # Matched-filter signal (cross-correlation of D with P_D)
        D_hat = fft.fft2(diff_filled)
        S = np.real(fft.ifft2(D_hat * np.conj(P_D_hat)))

        # Per-input noise kernels: k_n (SCI -> S), k_r (REF -> S). See class docstring.
        K_conv_P_D = np.real(fft.ifft2(K_hat * P_D_hat))
        if conv_side == 'REF':
            k_n, k_r = P_D, K_conv_P_D
        else:
            k_n, k_r = K_conv_P_D, P_D

        # Source/background noise propagation (ZOGY Eqs. 26-27)
        V_N = sigma_n ** 2
        V_R = sigma_r ** 2
        V_S_N = np.real(fft.ifft2(fft.fft2(V_N) * fft.fft2(k_n ** 2)))
        V_S_R = np.real(fft.ifft2(fft.fft2(V_R) * fft.fft2(k_r ** 2)))

        # Astrometric noise (ZOGY Eqs. 30-33)
        if use_strict_v_ast:
            if sci_image is None or ref_image is None:
                raise Exception("MeLOn ERROR: use_strict_v_ast=True requires sci_image and ref_image.")
            sci = np.where(np.isfinite(sci_image), sci_image, 0.0).astype(np.float64)
            ref = np.where(np.isfinite(ref_image), ref_image, 0.0).astype(np.float64)
            k_n_hat = np.conj(fft.fft2(k_n))
            k_r_hat = np.conj(fft.fft2(k_r))
            S_N_strict = np.real(fft.ifft2(k_n_hat * fft.fft2(sci)))
            S_R_strict = np.real(fft.ifft2(k_r_hat * fft.fft2(ref)))
            dSNdx = S_N_strict - np.roll(S_N_strict, 1, axis=1)
            dSNdy = S_N_strict - np.roll(S_N_strict, 1, axis=0)
            dSRdx = S_R_strict - np.roll(S_R_strict, 1, axis=1)
            dSRdy = S_R_strict - np.roll(S_R_strict, 1, axis=0)
            V_ast = (dx ** 2) * (dSNdx ** 2 + dSRdx ** 2) + (dy ** 2) * (dSNdy ** 2 + dSRdy ** 2)
        else:
            # Shortcut: gradient of the matched-filter signal directly.
            dSdx = S - np.roll(S, 1, axis=1)
            dSdy = S - np.roll(S, 1, axis=0)
            V_ast = (dx ** 2) * dSdx ** 2 + (dy ** 2) * dSdy ** 2

        V_total = np.maximum(V_S_N + V_S_R + V_ast, 1e-30)
        S_corr = S / np.sqrt(V_total)

        if renormalize:
            # Empirical noise calibration: 5 passes of 3-sigma clipping isolate the
            # noise-only pixels (real sources clipped away), then divide by the
            # measured MAD-sigma so the noise floor reads N(0, 1) regardless of how
            # accurate the analytic V_S was.
            sample = S_corr[np.isfinite(S_corr)]
            for _ in range(5):
                med = np.median(sample)
                mad_sigma = 1.4826 * np.median(np.abs(sample - med))
                if mad_sigma <= 0:
                    break
                keep = np.abs(sample - med) < 3.0 * mad_sigma
                if keep.sum() < 100:
                    break
                sample = sample[keep]
            med = float(np.median(sample))
            mad_sigma = float(1.4826 * np.median(np.abs(sample - med)))
            if VERBOSE_LEVEL in [1, 2]:
                print('MeLOn CheckPoint: Scorr empirical noise floor med=%.4f, MAD-sigma=%.4f '
                      '(rescaled to unit sigma)' % (med, mad_sigma))
            if mad_sigma > 0:
                S_corr = (S_corr - med) / mad_sigma

        if nan_mask.any():
            S = np.where(nan_mask, np.nan, S)
            S_corr = np.where(nan_mask, np.nan, S_corr)
        return S, S_corr

    @staticmethod
    def _sigma_map(PixA, gain):
        """Per-pixel sigma map: sqrt(max(counts - sky, 0)/gain + sky_sig**2).

        Works whether or not the image was background-subtracted (sky ~ 0 if it was).
        NaNs are filled with the sky level before estimation and left finite here;
        diff NaNs are what ultimately mask Scorr.
        """
        P = np.asarray(PixA, dtype=np.float64)
        m = ~np.isfinite(P)
        if m.any():
            P = P.copy()
            P[m] = np.nanmedian(P)
        sky_lvl, sky_sig = SkyLevel_Estimator.SLE(PixA_obj=P)
        if gain is None or gain <= 0:
            gain = 1.0
        source_counts = np.maximum(P - sky_lvl, 0.0)
        return np.sqrt(source_counts / gain + sky_sig ** 2)

    @staticmethod
    def from_subtraction(PixA_DIFF, Solution, SFFTConfig0, ConvdSide, FWHM_ConvdSide, \
        PixA_REF, PixA_SCI, GAIN_REF, GAIN_SCI, dx=0.0, dy=0.0, \
        use_strict_v_ast=False, renormalize=True, VERBOSE_LEVEL=2):

        """
        # * High-level ZOGY Scorr from an sfft subtraction result. Used by the subtraction packets.
        #
        #   Steps:
        #     1. Realize the matching kernel K at the stamp center from (Solution, SFFTConfig0).
        #     2. Build per-pixel sigma maps for SCI and REF from their pixel data, gains and sky sigma.
        #     3. Build the diff PSF P_D = K (conv) Gaussian(FWHM_ConvdSide).
        #     4. Run the full ZOGY matched filter + noise propagation (compute()).
        #
        # * Inputs
        #   -SFFTConfig0                   # the SFFTConfig[0] dict (has N0,N1,L0,L1,DK,Fpq).
        #   -ConvdSide ['REF'/'SCI']       # which side was convolved.
        #   -FWHM_ConvdSide                # FWHM (px) of the convolved side (FWHM_REF if ConvdSide='REF').
        #   -PixA_REF / -PixA_SCI          # input images (sfft pixel orientation), for sigma maps.
        #   -GAIN_REF / -GAIN_SCI          # gains (e-/ADU) of REF / SCI, for the Poisson term.
        #   -dx, -dy [0.0]                 # astrometric uncertainty (1-sigma, px). 0 ignores the V_ast term.
        #
        # * Returns
        #   PixA_SCORE, PixA_SCORR         # score and corrected-score (sigma units) images
        """

        N0, N1 = SFFTConfig0['N0'], SFFTConfig0['N1']
        L0, L1 = SFFTConfig0['L0'], SFFTConfig0['L1']
        DK, Fpq = SFFTConfig0['DK'], SFFTConfig0['Fpq']

        # 1. realize the matching kernel K at the stamp center
        XY_ctr = np.array([[N0 / 2.0, N1 / 2.0]]) + 0.5
        K = Realize_MatchingKernel(XY_q=XY_ctr).FromArray(Solution=Solution, \
            N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq)[0]

        # 2. per-pixel sigma maps (native units of each input image)
        sigma_map_sci = Scorr_Calculator._sigma_map(PixA_SCI, GAIN_SCI)
        sigma_map_ref = Scorr_Calculator._sigma_map(PixA_REF, GAIN_REF)

        # 3.+4. ZOGY matched filter + noise propagation
        PixA_SCORE, PixA_SCORR = Scorr_Calculator.compute(diff=PixA_DIFF, K_sfft=K, \
            fwhm_convd_side_px=FWHM_ConvdSide, sigma_map_sci=sigma_map_sci, sigma_map_ref=sigma_map_ref, \
            conv_side=ConvdSide, dx=dx, dy=dy, use_strict_v_ast=use_strict_v_ast, \
            renormalize=renormalize, sci_image=PixA_SCI, ref_image=PixA_REF, VERBOSE_LEVEL=VERBOSE_LEVEL)

        return PixA_SCORE, PixA_SCORR

    @staticmethod
    def FromFITS(FITS_DIFF, FITS_SCI, FITS_REF, FITS_Solution, FWHM_ConvdSide, \
        GAIN_KEY='GAIN', dx=0.0, dy=0.0, use_strict_v_ast=False, renormalize=True, VERBOSE_LEVEL=2):

        """
        # Convenience: read the DIFF (and its 'CONVD' keyword), the SCI/REF images, gains and the
        #   sfft Solution from FITS, then compute the ZOGY Scorr. Returns (PixA_SCORE, PixA_SCORR).
        """

        hdr = fits.getheader(FITS_DIFF, ext=0)
        if 'CONVD' not in hdr:
            raise Exception("MeLOn ERROR: DIFF header has no 'CONVD' keyword; pass ConvdSide via compute().")
        ConvdSide = hdr['CONVD']

        PixA_DIFF = fits.getdata(FITS_DIFF, ext=0).T   # match sfft's pixel orientation convention
        PixA_SCI = fits.getdata(FITS_SCI, ext=0).T
        PixA_REF = fits.getdata(FITS_REF, ext=0).T
        GAIN_SCI = float(fits.getheader(FITS_SCI, ext=0).get(GAIN_KEY, 1.0))
        GAIN_REF = float(fits.getheader(FITS_REF, ext=0).get(GAIN_KEY, 1.0))

        shdr = fits.getheader(FITS_Solution, ext=0)
        N0, N1 = int(shdr['N0']), int(shdr['N1'])
        XY_ctr = np.array([[N0 / 2.0, N1 / 2.0]]) + 0.5
        K = Realize_MatchingKernel(XY_q=XY_ctr).FromFITS(FITS_Solution=FITS_Solution)[0]

        sigma_map_sci = Scorr_Calculator._sigma_map(PixA_SCI, GAIN_SCI)
        sigma_map_ref = Scorr_Calculator._sigma_map(PixA_REF, GAIN_REF)

        return Scorr_Calculator.compute(diff=PixA_DIFF, K_sfft=K, fwhm_convd_side_px=FWHM_ConvdSide, \
            sigma_map_sci=sigma_map_sci, sigma_map_ref=sigma_map_ref, conv_side=ConvdSide, \
            dx=dx, dy=dy, use_strict_v_ast=use_strict_v_ast, renormalize=renormalize, \
            sci_image=PixA_SCI, ref_image=PixA_REF, VERBOSE_LEVEL=VERBOSE_LEVEL)
