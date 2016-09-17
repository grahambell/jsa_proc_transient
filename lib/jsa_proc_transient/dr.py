from codecs import ascii_decode
import os
import logging
import re
import subprocess
import sys
import tempfile

from astropy.io import fits
from starlink.ndfpack import Ndf

from transientclumps.TCOffsetFunctions import source_match
from transientclumps.TCGaussclumpsFunctions import run_gaussclumps
from transientclumps.TCPrepFunctions import prepare_image

logger = logging.getLogger(__name__)

data_dir = '/net/kamaka/export/data/jsa_proc/data/M16AL001'

# Dictionary of which mask to supply as 'mask2' values, by run name.
# If a reduction is not listed here, it does not use an external mask.
maskdict = {
    'R3': 'R1',
    'R4': 'R2',
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


def transient_analysis_all(inputfiles):
    outputfiles = []

    for type_ in ['R1', 'R2']:
        outputfiles.extend(transient_analysis(inputfiles, type_))

    return outputfiles


def transient_analysis(inputfiles, reductiontype):
    """
    Take in a list of input files from a single observation and
    the reduction type (e.g. 'R1', 'R2' etc).

    Returns a list of output files.
    """

    logger.debug('Checking configuration file "%s" exists', param_file)
    if not os.path.exists(param_file):
        raise Exception('Configuration file "{}" not found'.format(param_file))

    logger.debug('Checking kernel file "%s" exists', kernel)
    if not os.path.exists(kernel):
        raise Exception('Kernel file "{}" not found'.format(kernel))
    kernel_fwhm = float(kernel[:-4].split('_')[-1])

    # Get source, utdate, obsnum and fiter.
    logger.debug('Reading header from file "%s"', inputfiles[0])
    header = fits.Header.fromstring(''.join(
        ascii_decode(x)[0] for x in Ndf(inputfiles[0]).head['FITS']))
    source = safe_object_name(header['OBJECT'])
    date = header['UTDATE']
    obsnum = header['OBSNUM']
    filter_ = header['FILTER']

    logger.info('Performing %sum %s reduction for %s on %s (observation %i)',
                filter_, reductiontype, source, date, obsnum)

    # Set wavelength-dependent parameters up.
    if filter_ == '850':
        beam_fwhm = 14.5
        fcf_arcsec = 2.34
    else:
        raise Exception('Wavelength "{}" not recognised'.format(filter_))
    jypbm_conv = fcf_arcsec * 1.133 * (beam_fwhm ** 2.0)
    prepare_kwargs = {
        'kern_name': kernel,
        'kern_fwhm': kernel_fwhm,
        'jypbm_conv': jypbm_conv,
        'beam_fwhm': beam_fwhm,
    }

    # Get dimmconfig, reference and masks.
    logger.debug('Identifying dimmconfig, mask and reference files')
    dimmconfig = os.path.expandvars(dimmconfigdict[reductiontype])
    if not os.path.exists(dimmconfig):
        raise Exception('Dimmconfig file "{}" not found'.format(dimmconfig))

    mask2 = '!'
    mask_reductiontype = maskdict.get(reductiontype)
    if mask_reductiontype is not None:
        mask2 = get_filename_mask(source, mask_reductiontype)
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
    logger.debug('Running MAKEMAP, output: "%s"', out)
    subprocess.check_call(
        [
            os.path.expandvars('$SMURF_DIR/makemap'),
            'in=^{}'.format(filelist.name),
            'config=^{}'.format(dimmconfig),
            'out={}'.format(out),
            'ref={}'.format(reference),
            'mask2={}'.format(mask2),
            'msg_filter=none',
        ],
        shell=False)

    if not os.path.exists(out):
        raise Exception('MAKEMAP did not generate output "{}"'.format(out))

    # Prepare the image (smoothing etc) by running J. Lane's
    # prepare image routine.
    logger.debug('Preparing image')
    prepare_image(out, **prepare_kwargs)
    prepared_file = out[:-4]+'_crop_smooth_jypbm.sdf'

    # Identify the sources run J. Lane's run_gaussclumps routine.
    logger.debug('Running CUPID')
    run_gaussclumps(prepared_file, param_file)
    sourcecatalog = prepared_file[:-4] + '_log.FIT'

    if not os.path.exists(sourcecatalog):
        raise Exception(
            'CUPID did not generate catalog "{}"'.format(sourcecatalog))

    # Calculate offsets with J. Lane's source_match
    logger.debug('Performing source match')
    results = source_match(
        sourcecatalog, refcat, minpeak=0.2, maxrad=30, maxsep=10,
        cutoff=4, pix_scale=3.0)
    xoffset = results[0][1]
    yoffset = results[0][2]

    if (xoffset is None) or (yoffset is None):
        raise Exception('Pointing offsets not found')

    # Create the pointing offset file.
    offsetsfile = out[:-4] + '_offset.txt'
    create_pointing_offsets(offsetsfile, xoffset, yoffset, system='TRACKING')

    # Re reduce map with pointing offset.
    out_a = get_filename_output(
        source, date, obsnum, filter_, reductiontype, True)

    logger.debug('Running MAKEMAP, output: "%s"', out_a)
    subprocess.check_call(
        [
            os.path.expandvars('$SMURF_DIR/makemap'),
            'in=^{}'.format(filelist.name),
            'config=^{}'.format(dimmconfig),
            'out={}'.format(out_a),
            'ref={}'.format(reference),
            'mask2={}'.format(mask2),
            'pointing={}'.format(offsetsfile),
            'msg_filter=none',
        ],
        shell=False)

    if not os.path.exists(out_a):
        raise Exception('MAKEMAP did not generate output "{}"'.format(out_a))

    # Re run Lane's smoothing and gauss clumps routine.
    logger.debug('Preparing image')
    prepare_image(out_a, **prepare_kwargs)
    prepared_file = out_a[:-4]+'_crop_smooth_jypbm.sdf'

    # Identify the sources run J. Lane's run_gaussclumps routine.
    logger.debug('Running CUPID')
    run_gaussclumps(prepared_file, param_file)
    sourcecatalog_a = prepared_file[:-4] + '_log.FIT'

    if not os.path.exists(sourcecatalog_a):
        raise Exception(
            'CUPID did not generate catalog "{}"'.format(sourcecatalog_a))

    return [out, sourcecatalog, out_a, sourcecatalog_a, offsetsfile]


def create_pointing_offsets(offsetfile, x, y, system='TRACKING'):
    with open(offsetfile, 'w') as f:
        f.write('# SYSTEM={}\n'.format(system))
        f.write('#TAI DLON DLAT\n')
        f.write('1 {} {}\n'.format(x, y))
        f.write('10000000 {} {}\n'.format(x, y))


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
