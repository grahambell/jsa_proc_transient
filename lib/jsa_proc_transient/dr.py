from starlink import kappa, cupid, smurf
import os
import logging
import re
logging.basicConfig(level=logging.DEBUG)
import sys
import tempfile

from astropy.io import fits
from starlink.ndfpack import Ndf

from transientclumps.TCOffsetFunctions import source_match
from transientclumps.TCGaussclumpsFunctions import run_gaussclumps
from transientclumps.TCPrepFunctions import prepare_image

data_dir = '/net/kamaka/export/data/jsa_proc/data/M16AL001'

# Dictionary of whether to supply 'mask2' values, by run name.
maskdict = {
    'R1': False,
    'R2': False,
    'R3': True,
    'R4': True,
}

# Dictionary of dimmconfig files, by run name.
dimmconfigdict = {
    'R1': '$ORAC_CAL_ROOT/scuba2/dimmconfig_M16AL001_R1.lis',
    'R2': '$ORAC_CAL_ROOT/scuba2/dimmconfig_M16AL001_R2.lis',
    'R3': '$ORAC_CAL_ROOT/scuba2/dimmconfig_M16AL001_R3.lis',
    'R4': '$ORAC_CAL_ROOT/scuba2/dimmconfig_M16AL001_R4.lis',
}

# Find "transientclumps" configuration files.
config_dir = os.path.dirname(sys.modules['transientclumps'].__file__)

param_file = os.path.join(config_dir, 'data', 'parameters', 'GCParms.txt')

kernel = os.path.join(config_dir, 'data', 'kernels', 'TCgauss_6.sdf')




def transient_analysis(inputfiles, reductiontype):
    """
    Take in a list of input files from a single observation and
    the reduction type (e.g. 'R1', 'R2' etc).

    Returns the filename of the reduced maps (with and without
    pointing corrections) and the output source catalog (after
    pointing corrections).

    """

    if not os.path.exists(param_file):
        raise Exception('Configuration file "{}" not found'.format(param_file))

    if not os.path.exists(kernel):
        raise Exception('Kernel file "{}" not found'.format(kernel))
    kernel_fwhm = float(kernel[:-4].split('_')[-1])

    # Get source, utdate, obsnum and fiter.
    header = fits.Header.fromstring(''.join(Ndf(inputfiles[0]).head['FITS']))
    source = safe_object_name(header['OBJECT'])
    date = header['UTDATE']
    obsnum = header['OBSNUM']
    filter_ = header['FILTER']

    # Get dimmconfig, reference and masks.
    dimmconfig = os.path.expandvars(dimmconfigdict[reductiontype])
    if not os.path.exists(dimmconfig):
        raise Exception('Dimmconfig file "{}" not found'.format(dimmconfig))

    mask2 = '!'
    if maskdict[reductiontype]:
        mask2 = get_filename_mask(source, reductiontype)
        if not os.path.exists(mask2):
            raise Exception('Mask file "{}" not found'.format(mask2))

    reference = get_filename_reference(source, filter_)
    if not os.path.exists(reference):
        raise Exception('Reference file "{}" not found'.format(reference))

    refcat = get_filename_ref_cat(source, filter_)
    if not os.path.exists(refcat):
        raise Exception('Reference catalog "{}" not found'.format(refcat))

    # Create output file name.
    out = get_filename_output(
        source, date, obsnum, filter_, reductiontype, False)

    # Create list of input files
    filelist = tempfile.NamedTemporaryFile(
        mode='w', prefix='tmpList', delete=False)
    filelist.file.writelines([i + '\n' for i in inputfiles])
    filelist.file.close()

    # run makemap
    makemapres = smurf.makemap(in_='^' + filelist.name,
                               config='^' + dimmconfig,
                               out=out,
                               ref=reference,
                               mask2=mask2)

    if not os.path.exists(out):
        raise Exception('MAKEMAP did not generate output "{}"'.format(out))

    # Prepare the image (smoothing etc) by running J. Lane's
    # prepare image routine.
    prepare_image(out, kernel, kernel_fwhm)
    prepared_file = out[:-4]+'_crop_smooth_jypbm.sdf'

    # Identify the sources run J. Lane's run_gaussclumps routine.
    run_gaussclumps(prepared_file, param_file)
    sourcecatalog = prepared_file[:-4] + '_log.FIT'

    if not os.path.exists(sourcecatalog):
        raise Exception(
            'CUPID did not generate catalog "{}"'.format(sourcecatalog))

    # Calculate offsets with J. Lane's source_match
    results = source_match(
        sourcecatalog, refcat, minpeak=0.2, maxrad=30, maxsep=10,
        cutoff=4, pix_scale=3.0)
    xoffset = results[0][1]
    yoffset = results[0][2]

    if (xoffset is None) or (yoffset is None):
        raise Exception('Pointing offsets not found')

    # Create the pointing offset file.
    offsetsfile = create_pointing_offsets(xoffset, yoffset, system='TRACKING')

    # Re reduce map with pointing offset.
    out_a = get_filename_output(
        source, date, obsnum, filter_, reductiontype, True)
    makemapres2 = smurf.makemap(in_='^' + filelist.name,
                                config='^' + dimmconfig,
                                out=out,
                                out=out_a,
                                ref=reference,
                                mask2=mask2,
                                pointing=offsetsfile)

    if not os.path.exists(out_a):
        raise Exception('MAKEMAP did not generate output "{}"'.format(out_a))

    # Re run Lane's smoothing and gauss clumps routine.
    prepare_image(out_a)
    prepared_file = out_a[:-4]+'_crop_smooth_jypbm.sdf'

    # Identify the sources run J. Lane's run_gaussclumps routine.
    run_gaussclumps(prepared_file, param_file)
    sourcecatalog_a = prepared_file[:-4] + '_log.FIT'

    if not os.path.exists(sourcecatalog_a):
        raise Exception(
            'CUPID did not generate catalog "{}"'.format(sourcecatalog_a))

    return out_a, sourcecatalog


def create_pointing_offsets(x, y, system='TRACKING'):
    offsetfile = 'pointing_offset.txt'
    with open(offsetfile, 'w') as f:
        f.write('# SYSTEM={}\n'.format(system))
        f.write('#TAI DLON DLAT\n')
        f.write('1 {} {}\n'.format(x, y))
        f.write('10000000 {} {}\n'.format(x, y))
    return offsetfile


def safe_object_name(name):
    """
    Make safe version of an object name for use in the construction of
    file names, attempting to follow the scheme used by the survey.
    """

    # Remove spaces before numbers.
    name = re.sub(' +(?=[0-9])', '', name)

    # Remove unexpected characters.
    name = re.sub('[^-_A-Za-z0-9]', '_', name)

    # Return in upper case.
    return name.upper()


def get_filename_mask(source, reductiontype):
    return os.path.join(
        data_dir, 'mask',
        '{}_{}_extmask.sdf'.format(source, reductiontype))


def get_filename_reference(source, filter_):
    return os.path.join(
        data_dir, 'reference',
        '{}_reference_{}.sdf'.format(source, filter_))


def get_filename_ref_cat(source, filter_):
    return os.path.join(
        data_dir, 'cat',
        '{}_reference_cat_{}.fits'.format(source, filter_))


def get_filename_output(source, date, obsnum, filter_, reductiontype, aligned):
    # Change 1st letter of reduction type ('R') to 'A' for aligned map.
    if aligned:
        reductiontype = 'A' + reductiontype[1:]

    # Add 'E' (for EAO) prefix to reduction type.
    return '{}_{}_{:05d}_{}_E{}.sdf'.format(
        source, date, obsnum, filter_, reductiontype)
