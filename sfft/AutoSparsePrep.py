import warnings
import numba as nb
import numpy as np
from astropy.io import fits
from astropy.table import Column, hstack
from sfft.utils.pyAstroMatic.PYSEx import PY_SEx
from sfft.utils.SymmetricMatch import Symmetric_Match
from sfft.utils.HoughMorphClassifier import Hough_MorphClassifier
from sfft.utils.WeightedQuantile import TopFlatten_Weighted_Quantile
# version: Feb 10, 2023

__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.4"

@nb.njit(nogil=True)
def _signed_label(a, swap, LUT):
    # a segmentation label (its raw 32 bits, byte swapped when swap) relabelled: LUT[a] (the SubSource's new
    # label for an SS-island, -a for any other), -a beyond the LUT
    a = np.int64(a)
    if swap:
        a = ((a & 0xFF) << 24) | (((a >> 8) & 0xFF) << 16) | (((a >> 16) & 0xFF) << 8) | ((a >> 24) & 0xFF)
    if a >= 2147483648:
        a -= 4294967296
    if 0 <= a < LUT.shape[0]:
        return LUT[a]
    return -a


@nb.njit(nogil=True)
def _lmap_dilate(SEGr, SEGs, swap_r, swap_s, LUTr, LUTs, n, PZX, use_pzx, BAN, use_ban, L, PZ, banned):
    """One pass over the (y, x) label maps: the ProhibitedZone PZ (a non-SS island on either side, or PZX), the merged
    label map (the larger relabelled label, 0 in PZ) dilated by the L1 ball of radius n into L, and banned[v]
    set for each label v > 0 that L holds on a banned pixel (BAN) outside PZ. The dilation keeps, per row,
    its maxima over |dx| <= k for k = 0..n in a ring of 2n + 1 rows; L at (y, x) is the maximum over |dy| <= n
    of row y + dy's |dx| <= n - |dy| maximum (pixels outside the map ignored, as grey_dilation's reflected
    border only repeats nearer pixels)."""
    ny, nx = L.shape
    R = 2 * n + 1
    H = np.zeros((R, n + 1, nx), np.int32)
    for y in range(ny + n):
        if y < ny:
            h = H[y % R]
            for x in range(nx):
                r = _signed_label(SEGr[y, x], swap_r, LUTr)
                s = _signed_label(SEGs[y, x], swap_s, LUTs)
                pz = r < 0 or s < 0 or (use_pzx and PZX[y, x])
                PZ[y, x] = pz
                h[0, x] = 0 if pz else max(r, s)
            for k in range(1, n + 1):
                for x in range(nx):
                    v = h[k - 1, x]
                    if x > 0 and h[k - 1, x - 1] > v:
                        v = h[k - 1, x - 1]
                    if x < nx - 1 and h[k - 1, x + 1] > v:
                        v = h[k - 1, x + 1]
                    h[k, x] = v
        yo = y - n
        if yo < 0:
            continue
        first = True
        for dy in range(-n, n + 1):
            yy = yo + dy
            if yy < 0 or yy >= ny:
                continue
            g = H[yy % R, n - abs(dy)]
            if first:
                for x in range(nx):
                    L[yo, x] = g[x]
                first = False
            else:
                for x in range(nx):
                    if g[x] > L[yo, x]:
                        L[yo, x] = g[x]
        if use_ban:
            for x in range(nx):
                v = L[yo, x]
                if v > 0 and BAN[yo, x] and not PZ[yo, x]:
                    banned[v] = True


@nb.njit(nogil=True)
def _lmap_final(L, PZ, banned, use_ban, A):
    # L to -128 in PZ and -64 on the banned labels, A = L > 0; returns the count of active pixels
    ny, nx = L.shape
    cnt = 0
    for y in range(ny):
        for x in range(nx):
            v = L[y, x]
            if PZ[y, x]:
                v = -128
            elif use_ban and v > 0 and banned[v]:
                v = -64
            L[y, x] = v
            A[y, x] = v > 0
            cnt += v > 0
    return cnt


def _label_bits(a):
    # an integer label map as its raw 32-bit words (uint32 view, no copy) and whether they need a byte swap
    if a.dtype.kind not in 'iu':
        raise TypeError('segmentation maps must be integer, got %s' % a.dtype)
    if a.dtype.itemsize != 4:
        if a.size and (a.max() >= 2**31 or a.min() < -2**31):
            raise ValueError('segmentation labels must fit 32 bits')
        a = a.astype(np.int32)
    return a.view(a.dtype.newbyteorder('=').str[0] + 'u4'), not a.dtype.isnative


def _label_lut(old, new):
    # LUT[k] = -k, except LUT[old[i]] = new[i] (later pairs win, as in a dict)
    old = np.asarray(old, dtype=np.int64)
    if old.size and old.min() < 0:
        raise ValueError('segmentation labels of SubSources must be >= 0')
    size = int(old.max()) + 1 if old.size else 1
    LUT = -np.arange(size, dtype=np.int32)
    LUT[old] = np.asarray(new, dtype=np.int32)
    return LUT

class Auto_SparsePrep:
    def __init__(self, FITS_REF, FITS_SCI, GAIN_KEY='GAIN', SATUR_KEY='ESATUR', BACK_TYPE='MANUAL', \
        BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=2.0, ANALYSIS_THRESH=2.0, DETECT_MINAREA=5, \
        DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', ONLY_FLAGS=[0], BoundarySIZE=30, \
        PixA_REF_input=None, PixA_SCI_input=None, SATLEVEL_REF_input=None, SATLEVEL_SCI_input=None, VERBOSE_LEVEL=2):

        self.FITS_REF = FITS_REF
        self.FITS_SCI = FITS_SCI

        self.PixA_REF_input = PixA_REF_input
        self.PixA_SCI_input = PixA_SCI_input
        self.SATLEVEL_REF_input = SATLEVEL_REF_input
        self.SATLEVEL_SCI_input = SATLEVEL_SCI_input

        self.GAIN_KEY = GAIN_KEY
        self.SATUR_KEY = SATUR_KEY

        self.BACK_TYPE = BACK_TYPE
        self.BACK_VALUE = BACK_VALUE
        self.BACK_SIZE = BACK_SIZE
        self.BACK_FILTERSIZE = BACK_FILTERSIZE

        self.DETECT_THRESH = DETECT_THRESH
        self.ANALYSIS_THRESH = ANALYSIS_THRESH
        self.DETECT_MINAREA = DETECT_MINAREA
        self.DETECT_MAXAREA = DETECT_MAXAREA
        self.DEBLEND_MINCONT = DEBLEND_MINCONT
        self.BACKPHOTO_TYPE = BACKPHOTO_TYPE

        self.ONLY_FLAGS = ONLY_FLAGS
        self.BoundarySIZE = BoundarySIZE
        self.VERBOSE_LEVEL = VERBOSE_LEVEL

    def run_image_mask(self, AstSEx_SS, PixA_SEGr, PixA_SEGs, StarExt_iter, XY_PriorBan, BuildMaskedImages=True, \
        SatMasks=True, InputsFinite=False, PriorBanMask=None, ActiveMaskOut=None):
        # NOTE SatMasks=False leaves the saturation masks None (they are read only for MaskSatContam);
        #      InputsFinite=True: the caller guarantees NaN-free images, so the NaN masks are not built
        
        if self.PixA_REF_input is not None:
            PixA_REF = self.PixA_REF_input.T
        else:
            PixA_REF = fits.getdata(self.FITS_REF, ext=0).T
        if np.issubdtype(PixA_REF.dtype, np.integer):
            PixA_REF = PixA_REF.astype(np.float64)

        if self.PixA_SCI_input is not None:
            PixA_SCI = self.PixA_SCI_input.T
        else:
            PixA_SCI = fits.getdata(self.FITS_SCI, ext=0).T
        if np.issubdtype(PixA_SCI.dtype, np.integer):
            PixA_SCI = PixA_SCI.astype(np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.SATLEVEL_REF_input is not None:
                SATLEVEL_REF = self.SATLEVEL_REF_input
            else:
                SATLEVEL_REF = fits.getheader(self.FITS_REF, ext=0)[self.SATUR_KEY]
            if self.SATLEVEL_SCI_input is not None:
                SATLEVEL_SCI = self.SATLEVEL_SCI_input
            else:
                SATLEVEL_SCI = fits.getheader(self.FITS_SCI, ext=0)[self.SATUR_KEY]
            SatMask_REF = PixA_REF >= SATLEVEL_REF if SatMasks else None
            SatMask_SCI = PixA_SCI >= SATLEVEL_SCI if SatMasks else None

        # * Relabel SS-islands & Flip labels of NotSS-islands, Define ProhibitedZone, Make SFFTLmap
        #   NOTE: in two passes over the (y, x) memory order of the label maps, which are left unchanged:
        #   an SS-island takes its SubSource's label, any other island's label is negated; the ProhibitedZone
        #   is where either side is negative (or NaN); SFFTLmap is the larger label, 0 in the ProhibitedZone,
        #   grey-dilated by the StarExt_iter-iterated cross (the L1 ball of that radius, 1 when below 2 as in
        #   ndimage.iterate_structure), then -128 in the ProhibitedZone and -64 on Prior-Banned SubSources.
        SEGL_SSr = np.array(AstSEx_SS['SEGLABEL_REF']).astype(int)
        SEGL_SSs = np.array(AstSEx_SS['SEGLABEL_SCI']).astype(int)
        SEGL_SS = np.array(AstSEx_SS['SEGLABEL']).astype(int)

        NaNmask_U = None
        if not InputsFinite:
            NaNmask_REF = np.isnan(PixA_REF)
            NaNmask_SCI = np.isnan(PixA_SCI)
            if NaNmask_REF.any() or NaNmask_SCI.any():
                NaNmask_U = np.logical_or(NaNmask_REF, NaNmask_SCI)

        SEGr, swap_r = _label_bits(PixA_SEGr.T)
        SEGs, swap_s = _label_bits(PixA_SEGs.T)
        L = np.empty(SEGr.shape, dtype=np.int32)
        PZ = np.empty(SEGr.shape, dtype=bool)
        BAN = np.zeros((1, 1), dtype=bool) if PriorBanMask is None else PriorBanMask.T
        banned = np.zeros(max(int(SEGL_SS.max()) if SEGL_SS.size else 0, 0) + 1, dtype=bool)
        _lmap_dilate(SEGr, SEGs, swap_r, swap_s, _label_lut(SEGL_SSr, SEGL_SS), _label_lut(SEGL_SSs, SEGL_SS), \
            max(int(StarExt_iter), 1), np.zeros((1, 1), dtype=bool) if NaNmask_U is None else NaNmask_U.T, \
            NaNmask_U is not None, BAN, PriorBanMask is not None, L, PZ, banned)
        SFFTLmap, ProZone = L.T, PZ.T

        if XY_PriorBan is not None or PriorBanMask is not None:
            # ** Identify the Prior-Banned SubSources and decorate AstSEx_SS
            #    NOTE: Unlike Prior-Selection, the Prior-Ban coordinates do not have to be accurate.
            #    WARNING NOTE: AstSEx_SS will be updated in this step!
            #    NOTE: PriorBanMask (the tile's shape, True on banned pixels) bans the same as the coordinates
            #          of its True pixels given as XY_PriorBan (x + 1, y + 1)

            NUM_PB = 0
            if XY_PriorBan is not None:
                _XY_PB = np.asarray(XY_PriorBan)
                _ix, _iy = (_XY_PB[:, 0] - 0.5).astype(int), (_XY_PB[:, 1] - 0.5).astype(int)
                _v = SFFTLmap[_ix, _iy]
                banned[_v[(_v > 0) & ~ProZone[_ix, _iy]]] = True
                NUM_PB += XY_PriorBan.shape[0]
            if PriorBanMask is not None:
                NUM_PB += int(np.count_nonzero(PriorBanMask))
            PBMASK_SS = np.zeros(len(SEGL_SS), dtype=bool)
            _in = (SEGL_SS > 0) & (SEGL_SS < banned.size)
            PBMASK_SS[_in] = banned[SEGL_SS[_in]]
            AstSEx_SS.add_column(Column(PBMASK_SS, name='MASK_PriorBan'))
            
            if self.VERBOSE_LEVEL in [1, 2]:
                _message = 'Find / Given [%d / %d] Prior-Banned ' %(np.sum(PBMASK_SS), NUM_PB)
                _message += 'in current [%d] SubSources!' %(len(AstSEx_SS))
                print('MeLOn CheckPoint: %s' %_message)
        else:
            if self.VERBOSE_LEVEL in [1, 2]:
                print('MeLOn CheckPoint: SKIP, NO Prior-Banned Coordinates Given!')

        # * Update Label Map (SFFTLmap) by inactivating the Prior-Banned SubSources & Create ActiveMask
        #   NOTE: The preparation can guarantee that mREF & mSCI are NaN-Free !
        #   NOTE: ActiveMaskOut (bool, the tile's shape) receives the mask instead of a new array
        A = np.empty(L.shape, dtype=bool) if ActiveMaskOut is None else ActiveMaskOut.T
        NUM_ACTIVE = _lmap_final(L, PZ, banned, XY_PriorBan is not None or PriorBanMask is not None, A)
        ActiveMask = A.T
        
        if self.VERBOSE_LEVEL in [1, 2]:
            ActivePROP = NUM_ACTIVE / (ActiveMask.shape[0] * ActiveMask.shape[1])
            print('MeLOn CheckPoint: Active-Mask (Non-Prior-Banned SubSources) Pixel Proportion [%s]' \
                   %('{:.2%}'.format(ActivePROP)))

        # * Produce masked images PixA_mREF & PixA_mSCI
        #   NOTE: BuildMaskedImages=False leaves them to the caller (the images with zeros off the Active-Mask)
        PixA_mREF, PixA_mSCI = None, None
        if BuildMaskedImages:
            PixA_mREF = PixA_REF.copy()
            PixA_mSCI = PixA_SCI.copy()
            PixA_mREF[~ActiveMask] = 0.0      # NOTE REF has been sky-subtracted
            PixA_mSCI[~ActiveMask] = 0.0      # NOTE SCI has been sky-subtracted

        # * Create sfft-prep master dictionary
        SFFTPrepDict = {}
        SFFTPrepDict['PixA_REF'] = PixA_REF
        SFFTPrepDict['PixA_SCI'] = PixA_SCI
        SFFTPrepDict['REF-SAT-Mask'] = SatMask_REF
        SFFTPrepDict['SCI-SAT-Mask'] = SatMask_SCI
        SFFTPrepDict['Union-NaN-Mask'] = NaNmask_U

        SFFTPrepDict['SATLEVEL_REF'] = SATLEVEL_REF
        SFFTPrepDict['SATLEVEL_SCI'] = SATLEVEL_SCI

        SFFTPrepDict['SExCatalog-SubSource'] = AstSEx_SS
        SFFTPrepDict['SFFT-LabelMap'] = SFFTLmap
        SFFTPrepDict['Active-Mask'] = ActiveMask
        SFFTPrepDict['PixA_mREF'] = PixA_mREF
        SFFTPrepDict['PixA_mSCI'] = PixA_mSCI

        return SFFTPrepDict

    def HoughAutoMask(self, Hough_MINFR=0.1, Hough_PeakClip=0.7, BeltHW=0.2, PointSource_MINELLIP=0.3, \
        MatchTol=None, MatchTolFactor=3.0, COARSE_VAR_REJECTION=True, CVREJ_MAGD_THRESH=0.12, \
        CVREJ_NSIGMA=None, MIN_SUBSOURCE=None, ELABO_VAR_REJECTION=False, EVREJ_RATIO_THREH=5.0, \
        EVREJ_SAFE_MAGDEV=0.04, StarExt_iter=4, XY_PriorBan=None):

        # - Hough Transform is used to determine subtraction-sources
        # TODO: change parameter names
        # TODO: add a robust clipping on SNR & FR, use the DECam examples to determine ...
        # TODO: allows PointSource_ELLIP_Thresh being None?

        # * Use Hough-MorphClassifer to identify GoodSources in REF & SCI
        def main_hough(FITS_obj):
            # NOTE: SEGMENTATION map is sensitive to DETECT_THRESH
            Hmc = Hough_MorphClassifier.MakeCatalog(FITS_obj=FITS_obj, GAIN_KEY=self.GAIN_KEY, \
                SATUR_KEY=self.SATUR_KEY, BACK_TYPE=self.BACK_TYPE, BACK_VALUE=self.BACK_VALUE, \
                BACK_SIZE=self.BACK_SIZE, BACK_FILTERSIZE=self.BACK_FILTERSIZE, \
                DETECT_THRESH=self.DETECT_THRESH, ANALYSIS_THRESH=self.ANALYSIS_THRESH, \
                DETECT_MINAREA=self.DETECT_MINAREA, DETECT_MAXAREA=self.DETECT_MAXAREA, \
                DEBLEND_MINCONT=self.DEBLEND_MINCONT, BACKPHOTO_TYPE=self.BACKPHOTO_TYPE, \
                CHECKIMAGE_TYPE='SEGMENTATION', AddRD=False, ONLY_FLAGS=self.ONLY_FLAGS, \
                BoundarySIZE=self.BoundarySIZE, AddSNR=False, VERBOSE_LEVEL=self.VERBOSE_LEVEL)
            PixA_SEG = Hmc[1][0].astype(int)

            Hc = Hough_MorphClassifier.Classifier(AstSEx=Hmc[0], Hough_MINFR=Hough_MINFR, \
                Hough_PeakClip=Hough_PeakClip, BeltHW=BeltHW, PointSource_MINELLIP=PointSource_MINELLIP, \
                VERBOSE_LEVEL=self.VERBOSE_LEVEL)
            FHWM = Hc[5]
            AstSEx_GS = Hmc[0][Hc[3]]
            return AstSEx_GS, FHWM, PixA_SEG
        
        AstSEx_GSr, FWHM_REF, PixA_SEGr = main_hough(self.FITS_REF)
        AstSEx_GSs, FWHM_SCI, PixA_SEGs = main_hough(self.FITS_SCI)

        return self.match_and_mask(AstSEx_GSr=AstSEx_GSr, FWHM_REF=FWHM_REF, PixA_SEGr=PixA_SEGr, \
            AstSEx_GSs=AstSEx_GSs, FWHM_SCI=FWHM_SCI, PixA_SEGs=PixA_SEGs, MatchTol=MatchTol, \
            MatchTolFactor=MatchTolFactor, COARSE_VAR_REJECTION=COARSE_VAR_REJECTION, \
            CVREJ_MAGD_THRESH=CVREJ_MAGD_THRESH, CVREJ_NSIGMA=CVREJ_NSIGMA, MIN_SUBSOURCE=MIN_SUBSOURCE, \
            ELABO_VAR_REJECTION=ELABO_VAR_REJECTION, \
            EVREJ_RATIO_THREH=EVREJ_RATIO_THREH, EVREJ_SAFE_MAGDEV=EVREJ_SAFE_MAGDEV, \
            StarExt_iter=StarExt_iter, XY_PriorBan=XY_PriorBan)

    @staticmethod
    def magd_surface(X, Y, MAGD, FLUXr, FLUXERRr, FLUXs, FLUXERRs, deg, SNR_MIN=20.0, NITER=5, NSIG=3.0, MAX_DEV=0.3):
        # Robust 2D polynomial of the SCI-REF magnitude difference over the tile, so that variable rejection
        # compares each source with the local flux ratio instead of one constant MAG_OFFSET.
        u = 2.0 * (X - X.min()) / max(X.max() - X.min(), 1.0) - 1.0
        v = 2.0 * (Y - Y.min()) / max(Y.max() - Y.min(), 1.0) - 1.0
        A = np.array([u**i * v**(n - i) for n in range(deg+1) for i in range(n+1)]).T
        with np.errstate(divide='ignore', invalid='ignore'):
            sig = 1.0857 * np.sqrt((FLUXERRr / FLUXr)**2 + (FLUXERRs / FLUXs)**2)
            snr = np.minimum(FLUXr / FLUXERRr, FLUXs / FLUXERRs)
        use = np.isfinite(MAGD) & np.isfinite(sig) & (snr > SNR_MIN)
        # fit only sources near the typical ratio: saturated or blended sources sit far off and would drag the surface
        med = np.median(MAGD[use]) if use.any() else 0.0
        use &= np.abs(MAGD - med) < MAX_DEV
        # sparse tiles: lower the degree until there are 5 bright sources per term; below that, no surface
        while deg > 0 and use.sum() < 5 * (deg + 1) * (deg + 2) // 2:
            deg -= 1
        A = A[:, :(deg + 1) * (deg + 2) // 2]
        if use.sum() < 5 * A.shape[1]:
            return None
        w = 1.0 / (sig**2 + 0.01**2)
        for _ in range(NITER):
            sw = np.sqrt(w[use])
            coef = np.linalg.lstsq(A[use] * sw[:, None], MAGD[use] * sw, rcond=None)[0]
            r = MAGD - A @ coef
            mad = 1.4826 * np.median(np.abs(r[use] - np.median(r[use])))
            new = use & (np.abs(r) <= NSIG * max(mad, 0.005))
            if new.sum() < 5 * A.shape[1] or np.array_equal(new, use):
                break
            use = new
        surf = A @ coef
        # a surface that leaves the fitted range is not trusted
        if np.any(np.abs(surf[use] - med) > MAX_DEV):
            return None
        return surf

    def match_and_mask(self, AstSEx_GSr, FWHM_REF, PixA_SEGr, AstSEx_GSs, FWHM_SCI, PixA_SEGs, \
        MatchTol=None, MatchTolFactor=3.0, COARSE_VAR_REJECTION=True, CVREJ_MAGD_THRESH=0.12, \
        CVREJ_NSIGMA=None, MIN_SUBSOURCE=None, ELABO_VAR_REJECTION=False, EVREJ_RATIO_THREH=5.0, \
        EVREJ_SAFE_MAGDEV=0.04, StarExt_iter=4, XY_PriorBan=None, VARREJ_SPDEG=None, BuildMaskedImages=True, \
        SatMasks=True, InputsFinite=False, PriorBanMask=None, ActiveMaskOut=None):
        # Reusable core shared by HoughAutoMask (internal SExtractor+Hough detection) and
        # externally-injected detections (turbo SubtractionPrep): cross-match per-side
        # GoodSources, reject variables, build the SFFT masks. Each AstSEx_GS must carry
        # SEGLABEL/X_IMAGE/Y_IMAGE/MAG_AUTO/FLUX_AUTO/FLUXERR_AUTO; PixA_SEG is its raw
        # integer segmentation map.

        if self.VERBOSE_LEVEL in [1, 2]:
            _message = 'Estimated [FWHM_REF = %.3f pix] & [FWHM_SCI = %.3f pix]!' %(FWHM_REF, FWHM_SCI)
            print('\nMeLOn CheckPoint: %s' %_message)

        # * Determine Matched-GoodSources (abbr. MGS)
        XY_GSr = np.array([AstSEx_GSr['X_IMAGE'], AstSEx_GSr['Y_IMAGE']]).T
        XY_GSs = np.array([AstSEx_GSs['X_IMAGE'], AstSEx_GSs['Y_IMAGE']]).T
        
        UMatchTol = MatchTol
        if MatchTol is None:
            # - Given precise WCS, one can use a high MatchTolFactor ~ 3.0
            # - For very sparse fields where WCS is inaccurate, one can loosen MatchTolFactor to ~ 1.0
            # - WARNING NOTE: it can go wrong when the estimated FWHM is highly inaccurate.
            UMatchTol = np.sqrt((FWHM_REF / MatchTolFactor)**2 + (FWHM_SCI / MatchTolFactor)**2)

        if self.VERBOSE_LEVEL in [2]:
            print('MeLOn CheckPoint: Tolerance [%.3f pix] For Source Cross-Match!' %UMatchTol)

        Symm = Symmetric_Match.SM(XY_A=XY_GSr, XY_B=XY_GSs, tol=UMatchTol, return_distance=False)
        AstSEx_MGSr = AstSEx_GSr[Symm[:, 0]]
        AstSEx_MGSs = AstSEx_GSs[Symm[:, 1]]
        NUM_MGS = Symm.shape[0]

        # * Determine Magnitude Offset from MGS
        MAG_MGSr = np.array(AstSEx_MGSr['MAG_AUTO'])
        MAG_MGSs = np.array(AstSEx_MGSs['MAG_AUTO'])
        FLUX_MGSr = np.array(AstSEx_MGSr['FLUX_AUTO'])
        FLUX_MGSs = np.array(AstSEx_MGSs['FLUX_AUTO'])

        MAGD_MGS = MAG_MGSs - MAG_MGSr
        MAG_OFFSET0 = np.median(MAGD_MGS)

        NTE_4MO = 30  # NUM_TOP_END for MAG_OFFSET, default value used herein.
        MAG_OFFSETr = TopFlatten_Weighted_Quantile.TFWQ(values=MAGD_MGS, weights=FLUX_MGSr, \
            quantiles=[0.5], NUM_TOP_END=NTE_4MO)[0]
        
        MAG_OFFSETs = TopFlatten_Weighted_Quantile.TFWQ(values=MAGD_MGS, weights=FLUX_MGSs, \
            quantiles=[0.5], NUM_TOP_END=NTE_4MO)[0]
        MAG_OFFSET = (MAG_OFFSETr + MAG_OFFSETs)/2.0
        
        if np.abs(MAG_OFFSET - MAG_OFFSET0) > 0.05:
            if self.VERBOSE_LEVEL in [0, 1, 2]:
                _warn_message = 'Magnitude Offset Estimate --- WEIGHTED-MEDIAN IS HIGHLY DEVIATED FROM MEDIAN '
                _warn_message += '[median: %.3f mag] >>> [weighted-median: %.3f mag]!' %(MAG_OFFSET0, MAG_OFFSET)
                warnings.warn('\nMeLOn WARNING: %s' %_warn_message)
        
        if self.VERBOSE_LEVEL in [2]:
            _message = 'Magnitude Offset Estimate --- '
            _message += '[median: %.3f mag] >>> [weighted-median: %.3f mag]!' %(MAG_OFFSET0, MAG_OFFSET)
            print('\nMeLOn CheckPoint: %s' %_message)

        # * Reference level for variable rejection: the constant MAG_OFFSET, or (VARREJ_SPDEG) a smooth surface
        MAGD_REF_MGS = np.full(NUM_MGS, MAG_OFFSET)
        if VARREJ_SPDEG:
            _surf = self.magd_surface(X=np.array(AstSEx_MGSs['X_IMAGE'], dtype=float), \
                Y=np.array(AstSEx_MGSs['Y_IMAGE'], dtype=float), MAGD=MAGD_MGS, \
                FLUXr=FLUX_MGSr, FLUXERRr=np.array(AstSEx_MGSr['FLUXERR_AUTO']), \
                FLUXs=FLUX_MGSs, FLUXERRs=np.array(AstSEx_MGSs['FLUXERR_AUTO']), deg=int(VARREJ_SPDEG))
            if _surf is None:
                print('\nMeLOn CheckPoint: too few bright Matched-GoodSources for a variable-rejection surface, '
                      'using the constant MAG_OFFSET!')
            else:
                MAGD_REF_MGS = _surf
            if self.VERBOSE_LEVEL in [1, 2] and _surf is not None:
                print('\nMeLOn CheckPoint: Variable-Rejection reference surface [max degree %d] spans [%.3f, %.3f] mag!' \
                    %(int(VARREJ_SPDEG), np.percentile(_surf, 1), np.percentile(_surf, 99)))

        _dev_MGS = MAGD_MGS - MAGD_REF_MGS

        # * Apply a coarse variable rejection (abbr. CVREJ)
        if COARSE_VAR_REJECTION:
            if CVREJ_NSIGMA is not None:
                _sig = 1.4826 * np.median(np.abs(_dev_MGS - np.median(_dev_MGS)))
                CVREJ_THRESH = CVREJ_NSIGMA * _sig if _sig > 0 else CVREJ_MAGD_THRESH
            else:
                CVREJ_THRESH = CVREJ_MAGD_THRESH
            CVREJ_MASK = np.abs(_dev_MGS) > CVREJ_THRESH

            if self.VERBOSE_LEVEL in [1, 2]:
                if CVREJ_NSIGMA is not None:
                    _message = 'Coarse Variable Rejection [%.2f sigma = %.3f mag] ' %(CVREJ_NSIGMA, CVREJ_THRESH)
                else:
                    _message = 'Coarse Variable Rejection [magnitude deviation > %.3f mag] ' %CVREJ_THRESH
                _message += 'on Matched-GoodSources [%d / %d]!' %(np.sum(CVREJ_MASK), NUM_MGS)
                print('\nMeLOn CheckPoint: %s' %_message)
        else:
            CVREJ_MASK = np.zeros(NUM_MGS, dtype=bool)
            if self.VERBOSE_LEVEL in [1, 2]:
                print('\nMeLOn CheckPoint: SKIP Coarse Variable Rejection!')

        AstSEx_iSSr = AstSEx_MGSr[~CVREJ_MASK]
        AstSEx_iSSs = AstSEx_MGSs[~CVREJ_MASK]
        _cvrej_keep_idx = np.where(~CVREJ_MASK)[0]
        REJECT_MASK = CVREJ_MASK.copy()

        # * Apply a more elaborate variable rejection (abbr. EVREJ)
        if ELABO_VAR_REJECTION:

            if self.VERBOSE_LEVEL in [2]:
                warnings.warn('\nMeLOn WARNING: Elaborate Variable Rejection requires CORRECT GAIN in FITS HEADER!')

            """
            # Remarks on the Elaborate Variable Rejection
            # [1] The coarse variable rejection by CVREJ_MAGD_THRESH can only exclude the most drastic variable sources.
            #     As a result, the possible missing varaibles, especially at the bright end, can strongly affect the subtraction. 
            #
            # [2] Elaborate Variable Rejection (EVREJ) is designed to further identify variables by taking their 
            #     photometric uncertainties into account. For stationary objects, the flux change between two images 
            #     is a predictable probability distribution propagated from original photometric uncertatinties.
            #     EVREJ identify the outliers of the distribution as a variable. 
            #     NOTE: Bright transients (compared with its host) can be also identified by EVREJ.
            #
            # [3] However, the scenario in [2] is very ideal so that EVREJ can overkill. This motivated us to add a
            #     protection on EVREJ by -EVREJ_SAFE_MAGDEV, which is used to handle the imperfections listed below.
            #
            #     (a) distribution mean: The photometric scaling can vary across the field (says, ~1%, or ~0.01mag). 
            #         However, we only use a constant MAG_OFFSET for EVREJ. The error of MAG_OFFSET, including the ignorance 
            #         of spatial variation and the uncertainty of the constant MAG_OFFSET itself, makes the calculated flux 
            #         change only an approximation of the 'real' value.
            #
            #     (b) distribution variance: i) inaccurate SExtractor FLUXERR, especially at the bright end 
            #         (Poission Noise Dominated). e.g., A mosaic input image used an average GAIN value, 
            #         making FLUXERR calculation in SExtractor not accurate. ii) the error of MAG_OFFSET 
            #         will affect the scaling of one FLUXERR.
            #
            """

            MAG_iSSr = np.array(AstSEx_iSSr['MAG_AUTO'])
            MAG_iSSs = np.array(AstSEx_iSSs['MAG_AUTO'])
            FLUX_iSSr = np.array(AstSEx_iSSr['FLUX_AUTO'])
            FLUX_iSSs = np.array(AstSEx_iSSs['FLUX_AUTO'])
            FLUXERR_iSSr = np.array(AstSEx_iSSr['FLUXERR_AUTO'])
            FLUXERR_iSSs = np.array(AstSEx_iSSs['FLUXERR_AUTO'])
            MAGD_iSS = MAG_iSSs - MAG_iSSr

            # *** Note on the flux scaling *** 
            # (1) MAG_OFFSET = MAG_SCI - MAG_REF = -2.5 * np.log10(FLUX_SCI/FLUX_REF)
            # (2) FLUX_SCAL = FLUX_SCI/FLUX_REF = 10**(MAG_OFFSET/-2.5)

            MAGD_REF_iSS = MAGD_REF_MGS[_cvrej_keep_idx]
            FLUX_SCAL = 10**(MAGD_REF_iSS/-2.5)
            sFLUX_iSSr = FLUX_SCAL * FLUX_iSSr
            sFLUXERR_iSSr = FLUX_SCAL * FLUXERR_iSSr

            _DATA = FLUX_iSSs - sFLUX_iSSr
            _SIGMA = np.sqrt(sFLUXERR_iSSr**2 + FLUXERR_iSSs**2)
            _OUTMASK = np.abs(_DATA) > EVREJ_RATIO_THREH * _SIGMA
            _SAFEMASK = np.abs(MAGD_iSS - MAGD_REF_iSS) <= EVREJ_SAFE_MAGDEV
            
            EVREJ_MASK = np.logical_and(_OUTMASK, ~_SAFEMASK)
            REJECT_MASK[_cvrej_keep_idx[EVREJ_MASK]] = True

            EVREJ_PERC = np.sum(EVREJ_MASK) / NUM_MGS

            if self.VERBOSE_LEVEL in [1, 2]:
                _message = 'Elaborate Variable Rejection [flux deviation > %.2f sigma] & ' %EVREJ_RATIO_THREH
                _message += '[magnitude deviation > %.3f mag] ' %EVREJ_SAFE_MAGDEV
                _message += 'on Matched-GoodSources [%d / %d] ' %(np.sum(EVREJ_MASK), NUM_MGS)
                _message += 'or [%s]!' %('{:.2%}'.format(EVREJ_PERC))
                print('\nMeLOn CheckPoint: %s' %_message)

            if EVREJ_PERC > 0.1:
                if self.VERBOSE_LEVEL in [0, 1, 2]:
                    _warn_message = '[%s] Matched-GoodSources ARE REJECTED BY EVREJ!' %('{:.2%}'.format(EVREJ_PERC))
                    warnings.warn('\nMeLOn IMPORTANT WARNING: %s' %_warn_message)
        else:
            if self.VERBOSE_LEVEL in [1, 2]:
                print('\nMeLOn CheckPoint: SKIP Elaborate Variable Rejection!')

        # * Conditioning guard: never starve the kernel solve below MIN_SUBSOURCE
        _keep = ~REJECT_MASK
        _floor = None if MIN_SUBSOURCE is None else min(MIN_SUBSOURCE, NUM_MGS)
        if _floor is not None and 0 < int(_keep.sum()) < _floor:
            _rej_idx = np.where(REJECT_MASK)[0]
            _addback = _rej_idx[np.argsort(np.abs(_dev_MGS[_rej_idx]))[:_floor - int(_keep.sum())]]
            _keep[_addback] = True
            if self.VERBOSE_LEVEL in [1, 2]:
                print('\nMeLOn CheckPoint: Conditioning-Guard re-instated [%d] least-deviant sources to floor [%d]!' %(len(_addback), _floor))

        AstSEx_SSr = AstSEx_MGSr[_keep]
        AstSEx_SSs = AstSEx_MGSs[_keep]

        # * Combine catalogs for survived MGS (i.e., SubSources)
        for coln in AstSEx_SSr.colnames:
            AstSEx_SSr[coln].name = coln + '_REF'
            AstSEx_SSs[coln].name = coln + '_SCI'
        AstSEx_SS = hstack([AstSEx_SSr, AstSEx_SSs])
        
        # create unified segmentation label
        AstSEx_SS.add_column(Column(1+np.arange(len(AstSEx_SS)), name='SEGLABEL'), index=0)
        if self.VERBOSE_LEVEL in [1, 2]:
            print('\nMeLOn CheckPoint: SubSources out of Matched-GoodSources [%d / %d]!' %(len(AstSEx_SS), NUM_MGS))
        
        # * Run Image Masking (P.S. some SubSources might be marked as invalid by the Prior-Ban list)
        SFFTPrepDict = self.run_image_mask(AstSEx_SS=AstSEx_SS, PixA_SEGr=PixA_SEGr, \
            PixA_SEGs=PixA_SEGs, StarExt_iter=StarExt_iter, XY_PriorBan=XY_PriorBan, \
            BuildMaskedImages=BuildMaskedImages, SatMasks=SatMasks, InputsFinite=InputsFinite, \
            PriorBanMask=PriorBanMask, ActiveMaskOut=ActiveMaskOut)
        
        # * Append Additional information
        SFFTPrepDict['MAG_OFFSET'] = MAG_OFFSET
        SFFTPrepDict['FWHM_REF'] = FWHM_REF
        SFFTPrepDict['FWHM_SCI'] = FWHM_SCI

        return SFFTPrepDict

    def SemiAutoMask(self, XY_PriorSelect=None, MatchTol=None, MatchTolFactor=3.0, \
        StarExt_iter=4, XY_PriorBan=None):
        # - We directly use given Prior-Selection to determine subtraction-sources

        def func4phot(FITS_obj):
            # run SExtractor photometry
            SExParam = ['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', \
                        'MAGERR_AUTO', 'FLAGS', 'FLUX_RADIUS', 'FWHM_IMAGE', 'A_IMAGE', 'B_IMAGE']
            
            PYSEX_OP = PY_SEx.PS(FITS_obj=FITS_obj, SExParam=SExParam, GAIN_KEY=self.GAIN_KEY, \
                SATUR_KEY=self.SATUR_KEY, BACK_TYPE=self.BACK_TYPE, BACK_VALUE=self.BACK_VALUE, \
                BACK_SIZE=self.BACK_SIZE, BACK_FILTERSIZE=self.BACK_FILTERSIZE, DETECT_THRESH=self.DETECT_THRESH, \
                ANALYSIS_THRESH=self.ANALYSIS_THRESH, DETECT_MINAREA=self.DETECT_MINAREA, \
                DETECT_MAXAREA=self.DETECT_MAXAREA, DEBLEND_MINCONT=self.DEBLEND_MINCONT, \
                BACKPHOTO_TYPE=self.BACKPHOTO_TYPE, CHECKIMAGE_TYPE='SEGMENTATION', AddRD=False, \
                ONLY_FLAGS=self.ONLY_FLAGS, XBoundary=self.BoundarySIZE, YBoundary=self.BoundarySIZE, \
                MDIR=None, VERBOSE_LEVEL=self.VERBOSE_LEVEL)
            AstSEx, PixA_SEG = PYSEX_OP[0], PYSEX_OP[1][0].astype(int)
            
            # *** Note on the crude estimate of FWHM ***
            # our strategy is to calculate a median FWHM_IMAGE weighted by the source flux, meanwhile, 
            # we modulate the weights by applying a penalty on the sources with large FWHM_IMAGE.

            NTE_4FWHM = 30  # NUM_TOP_END for FWHM, default value used herein.
            _values, _weights = np.array(AstSEx['FWHM_IMAGE']), np.array(AstSEx['FLUX_AUTO'])
            _weights /= np.clip(_values, a_min=1.0, a_max=None)**2

            FWHM = TopFlatten_Weighted_Quantile.TFWQ(values=_values, weights=_weights, \
                quantiles=[0.5], NUM_TOP_END=NTE_4FWHM)[0]
            return AstSEx, FWHM, PixA_SEG
        
        AstSExr, FWHM_REF, PixA_SEGr = func4phot(self.FITS_REF)
        AstSExs, FWHM_SCI, PixA_SEGs = func4phot(self.FITS_SCI)
        
        if self.VERBOSE_LEVEL in [1, 2]:
            _message = 'Estimated [FWHM_REF = %.3f pix] & [FWHM_SCI = %.3f pix]!' %(FWHM_REF, FWHM_SCI)
            print('\nMeLOn CheckPoint: %s' %_message)

        # * Cross Match REF & SCI catalogs
        XYr = np.array([AstSExr['X_IMAGE'], AstSExr['Y_IMAGE']]).T
        XYs = np.array([AstSExs['X_IMAGE'], AstSExs['Y_IMAGE']]).T

        UMatchTol = MatchTol
        if MatchTol is None:
            # - Given precise WCS, one can use a high MatchTolFactor ~ 3.0
            # - For very sparse fields where WCS is inaccurate, one can loosen MatchTolFactor to ~ 1.0
            # - WARNING NOTE: it can go wrong when the estimated FWHM is highly inaccurate.
            UMatchTol = np.sqrt((FWHM_REF / MatchTolFactor)**2 + (FWHM_SCI / MatchTolFactor)**2)
        
        if self.VERBOSE_LEVEL in [2]:
            print('MeLOn CheckPoint: Tolerance [%.3f pix] For Source Cross-Match!' %UMatchTol)

        _Symm = Symmetric_Match.SM(XY_A=XYr, XY_B=XYs, tol=UMatchTol, return_distance=False)
        AstSEx_Mr = AstSExr[_Symm[:, 0]]
        AstSEx_Ms = AstSExs[_Symm[:, 1]]

        # * Determine Magnitude Offset from Matched Sources
        MAG_Mr = np.array(AstSEx_Mr['MAG_AUTO'])
        MAG_Ms = np.array(AstSEx_Ms['MAG_AUTO'])
        FLUX_Mr = np.array(AstSEx_Mr['FLUX_AUTO'])
        FLUX_Ms = np.array(AstSEx_Ms['FLUX_AUTO'])

        MAGD_M = MAG_Ms - MAG_Mr
        MAG_OFFSET0 = np.median(MAGD_M)

        NTE_4MO = 30  # NUM_TOP_END for MAG_OFFSET, default value used herein.
        MAG_OFFSETr = TopFlatten_Weighted_Quantile.TFWQ(values=MAGD_M, weights=FLUX_Mr, \
            quantiles=[0.5], NUM_TOP_END=NTE_4MO)[0]
        
        MAG_OFFSETs = TopFlatten_Weighted_Quantile.TFWQ(values=MAGD_M, weights=FLUX_Ms, \
            quantiles=[0.5], NUM_TOP_END=NTE_4MO)[0]
        MAG_OFFSET = (MAG_OFFSETr + MAG_OFFSETs)/2.0
        
        if np.abs(MAG_OFFSET - MAG_OFFSET0) > 0.05:
            if self.VERBOSE_LEVEL in [0, 1, 2]:
                _warn_message = 'Magnitude Offset Estimate --- HIGHLY DEVIATED FROM DIRECT MEDIAN '
                _warn_message += '[%.3f mag -> %.3f mag]!' %(MAG_OFFSET0, MAG_OFFSET)
                warnings.warn('\nMeLOn WARNING: %s' %_warn_message)

        if self.VERBOSE_LEVEL in [2]:
            _message = 'Magnitude Offset Estimate --- '
            _message += '[%.3f mag -> %.3f mag]!' %(MAG_OFFSET0, MAG_OFFSET)
            print('\nMeLOn CheckPoint: %s' %_message)
        
        # * Combine catalogs
        for coln in AstSEx_Mr.colnames:
            AstSEx_Mr[coln].name = coln + '_REF'
            AstSEx_Ms[coln].name = coln + '_SCI'
        AstSEx_iSS = hstack([AstSEx_Mr, AstSEx_Ms])

        X_IMAGE_REF_SCI_MEAN = np.array(AstSEx_iSS['X_IMAGE_REF'] + AstSEx_iSS['X_IMAGE_SCI'])/2.0
        Y_IMAGE_REF_SCI_MEAN = np.array(AstSEx_iSS['Y_IMAGE_REF'] + AstSEx_iSS['Y_IMAGE_SCI'])/2.0
        AstSEx_iSS.add_column(Column(X_IMAGE_REF_SCI_MEAN, name='X_IMAGE_REF_SCI_MEAN'))
        AstSEx_iSS.add_column(Column(Y_IMAGE_REF_SCI_MEAN, name='Y_IMAGE_REF_SCI_MEAN'))

        # * Only Preserve the Objects in the Prior-Selection (i.e., SubSources)
        #   WARNING NOTE: here we still use the same cross-match tolerance!
        #   WARNING NOTE: we match the prior selection to the mean image coordinates.

        XY_iSS = np.array([AstSEx_iSS['X_IMAGE_REF_SCI_MEAN'], AstSEx_iSS['Y_IMAGE_REF_SCI_MEAN']]).T
        _Symm = Symmetric_Match.SM(XY_A=XY_PriorSelect, XY_B=XY_iSS, tol=UMatchTol, return_distance=False)
        AstSEx_SS = AstSEx_iSS[_Symm[:, 1]]
        
        # record matching and create unified segmentation label
        AstSEx_SS.add_column(Column(_Symm[:, 0], name='INDEX_PRIOR_SELECTION'), index=0)
        AstSEx_SS.add_column(Column(1+np.arange(len(AstSEx_SS)), name='SEGLABEL'), index=0)

        if self.VERBOSE_LEVEL in [1, 2]:
            print('MeLOn CheckPoint: Find / Given [%d / %d] Prior-Selected in [%d] Matched-Sources!' \
                   %(len(AstSEx_SS), XY_PriorSelect.shape[0], len(AstSEx_iSS)))
        
        # * Run Image Masking (P.S. some SubSources might be marked as invalid by the Prior-Ban list)
        SFFTPrepDict = self.run_image_mask(AstSEx_SS=AstSEx_SS, PixA_SEGr=PixA_SEGr, \
            PixA_SEGs=PixA_SEGs, StarExt_iter=StarExt_iter, XY_PriorBan=XY_PriorBan)
        
        SFFTPrepDict['MAG_OFFSET'] = MAG_OFFSET
        SFFTPrepDict['FWHM_REF'] = FWHM_REF
        SFFTPrepDict['FWHM_SCI'] = FWHM_SCI

        return SFFTPrepDict
