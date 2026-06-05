"""Slim vendored copy of rdmpy/calibrate.py — only ``get_rdm_psfs``.

Vendored from rdmpy-main/rdmpy/calibrate.py.  The original module exported
``calibrate_rdm`` / ``calibrate_sdm`` / ``get_sdm_psfs`` / ``get_calib_info`` /
``isolate_psf`` / ``get_psf_centers`` as well, but those depend on
``_src/opt.py`` (which in turn requires ``kornia``) and on
``skimage.feature.corner_peaks`` / ``skimage.morphology``.  ``get_rdm_psfs``
itself uses none of that, so we drop everything else along with the
``opt`` / ``corner_peaks`` / ``erosion`` / ``disk`` / ``matplotlib`` / ``tqdm``
imports they pulled in.
"""

import gc

import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F

from ._src import psf_model


def get_rdm_psfs(
    seidel_coeffs,
    dim,
    model,
    patch_size=0,
    sys_params={},
    downsample=1,
    higher_order=None,
    verbose=True,
    device=torch.device("cpu"),
):
    """Return PSF ROFTs for a given set of Seidel coefficients.

    This function takes a set of Seidel coefficients and returns the PSF ROFTs for the optical system. There is one
    PSF ROFT for each radius in the image. The PSF ROFTs can be used directly in ring deconvolution.

    Parameters
    ----------
    seidel_coeffs : torch.Tensor
        Seidel coefficients of the optical system. Should be (6,1) with coefficients: sphere, coma, astigmatism, field curvature, distortion, defocus.

    dim : int
        Desired sidelength of each PSF image. Note that it enforces square images.

    model : str
        Either 'lsi' or 'lri' for the type of PSF model to use. LSI model will return a single PSF at the center of the image,
        while LRI model will return a stack of PSF RoFTs.

    patch_size : int
        Size of the isoplanatic annuli. If 0, no patching will be done. If > 0, the PSFs will be computed in patches.

    sys_params : dict
        Parameters for the optical system.

    downsample : int
        Factor by which to downsample the PSFs after fitting. Useful for saving memory.

    higher_order : bool
        Whether to include higher order aberrations in the PSF computation. If True, will include an additional 8 coefficients.

    verbose : bool
        Whether to print out progress.

    device : torch.device

    Returns
    -------
    psf_data : torch.Tensor
        PSFs of the optical system. If `model` is 'lsi', this is a single PSF.
        If `model` is 'lri', this is a stack of PSF RoFTs.

    """

    # default parameters which describe the optical system.
    def_sys_params = {
        "samples": dim,
        "L": 0,
        "lamb": 0.55e-6,
        "NA": 0.5,
    }
    radius_over_z = np.tan(np.arcsin(def_sys_params["NA"]))
    def_sys_params["L"] = ((dim) * (def_sys_params["lamb"])) / (4 * (radius_over_z))
    def_sys_params.update(sys_params)

    patch_based = patch_size > 0 and patch_size <= dim // abs(downsample)

    if not torch.is_tensor(seidel_coeffs):
        seidel_coeffs = torch.tensor(seidel_coeffs).to(device)

    if model == "lsi":
        point_list = [(0, 0)]  # just the center PSF
    elif model == "lri":
        rs = np.linspace(
            0, (dim / 2), int(dim // abs(downsample)), endpoint=False, retstep=False
        )

        if patch_based:
            rs = rs[::patch_size]

        point_list = [(r, -r) for r in rs]  # radial line of PSFs
    else:
        raise (NotImplementedError)

    if verbose:
        print("rendering PSFs...")

    if model == "lri":
        buffer = 2
    else:
        buffer = 0

    psf_data = psf_model.compute_rdm_psfs(
        seidel_coeffs,
        point_list,
        sys_params=def_sys_params,
        polar=(model == "lri" and not patch_based),
        stack=True,
        buffer=buffer,
        shift=not patch_based,
        downsample=downsample,
        higher_order=higher_order,
        verbose=verbose,
        device=device,
    )

    # prep the PSFs for outputing to the user
    if model == "lsi":
        psf_data = psf_data[0].to(device)
    if model == "lri":
        # here compute the RoFT of each PSF in-place (torch.rfft is memory inefficient)

        if patch_based:
            pad = dim // 2
            psf_data = F.pad(psf_data, (0, pad, 0, pad))
            for i in range(psf_data.shape[0]):
                temp_rft = fft.rfft2(psf_data[i, :, 0:-buffer])
                psf_data[i, :, 0 : psf_data.shape[-1] // 2] = torch.real(temp_rft)
                psf_data[i, :, psf_data.shape[-1] // 2 :] = torch.imag(temp_rft)
        else:
            for i in range(psf_data.shape[0]):
                temp_rft = fft.rfft(psf_data[i, 0:-2, :], dim=0)
                psf_data[i, 0 : psf_data.shape[1] // 2, :] = torch.real(temp_rft)
                psf_data[i, psf_data.shape[1] // 2 :, :] = torch.imag(temp_rft)

            del temp_rft
        gc.collect()

    return psf_data
