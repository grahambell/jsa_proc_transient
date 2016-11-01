from __future__ import absolute_import

from codecs import ascii_decode
import os
import logging
import re
import shutil
import subprocess
import sys
import tempfile

from astropy.io import fits
from starlink.ndfpack import Ndf

from transientclumps.TCOffsetFunctions import source_match
from transientclumps.TCGaussclumpsFunctions import run_gaussclumps
from transientclumps.TCPrepFunctions import prepare_image

from .gbs import get_field_name as gbs_field_name

logger = logging.getLogger(__name__)

data_dir = '/net/kamaka/export/data/jsa_proc/data/M16AL001'

# Dictionary of which mask to supply as 'REF' values in place of the
# normal reference image, by run name.
# If a reduction is not listed here, it does not use an external mask,
# and should just be given the normal reference image.
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
kernel_fwhm = float(kernel[:-4].split('_')[-1])


def transient_analysis(inputfiles, reductiontype, no_450_cat, as_ref_cat,
                       dimmconfig_850=None, dimmconfig_450=None):
    """
    Take in a list of input files from a single observation and
    the reduction type (e.g. 'R1', 'R2' etc).

    Returns a list of output files.
    """

    # Check reduction type is valid.
    if reductiontype not in dimmconfigdict:
        raise Exception(
            'Unrecognised reduction type "{}"'.format(reductiontype))

    logger.debug('Checking kernel file "%s" exists', kernel)
    if not os.path.exists(kernel):
        raise Exception('Kernel file "{}" not found'.format(kernel))

    # Organize input files into subsystems.
    logger.debug('Organizing input files by subsystem')
    subsystems = {450: [], 850: []}

    for file_ in inputfiles:
        file_basename = os.path.basename(file_)
        if file_basename.startswith('s4'):
            subsystems[450].append(file_)
        elif file_basename.startswith('s8'):
            subsystems[850].append(file_)
        else:
            raise Exception(
                'Did not recognise raw file name "{}"'.format(file_basename))

    if not subsystems[850]:
        raise Exception('No 850um data files given')

    output_files = []

    logger.debug('Performing 850um analysis')
    output_files.extend(transient_analysis_subsystem(
        subsystems[850], reductiontype, '850', None,
        install_catalog_as_ref=as_ref_cat,
        dimmconfig=dimmconfig_850))

    if subsystems[450]:
        # Offsets file should have been first given.
        offsetsfile = output_files[0]

        if not offsetsfile.endswith('_offset.txt'):
            raise Exception(
                'File "{}" does not look like an offsets file'.format(
                    offsetsfile))

        output_files.extend(transient_analysis_subsystem(
            subsystems[450], reductiontype, '450', offsetsfile,
            expect_missing_catalog=no_450_cat,
            dimmconfig=dimmconfig_450))

    return output_files


def transient_analysis_subsystem(inputfiles, reductiontype, filter_,
                                 offsetsfile, expect_missing_catalog=False,
                                 install_catalog_as_ref=False,
                                 dimmconfig=None):
    """
    Take in a list of input files from one subsystem of a single observation
    and the reduction type (e.g. 'R1', 'R2' etc).

    Returns a list of output files.

    If an offsetsfile is not given then an initial reduction will be done
    to determine the offsets.  In this case the newly created offsetsfile
    will be the first file returned.
    """

    # Get source, utdate, obsnum and fiter.
    logger.debug('Reading header from file "%s"', inputfiles[0])
    header = fits.Header.fromstring(''.join(
        ascii_decode(x)[0] for x in Ndf(inputfiles[0]).head['FITS']))
    raw_source = header['OBJECT']
    source = safe_object_name(raw_source)
    date = header['UTDATE']
    obsnum = header['OBSNUM']
    if filter_ != header['FILTER']:
        raise Exception('Unexpected value of FILTER header')

    project = header['PROJECT']
    if project == 'M16AL001':
        is_gbs = False
        field_name = source
    elif project.startswith('MJLSG'):
        is_gbs = True
        field_name = gbs_field_name(raw_source)
    else:
        raise Exception('Unexpected project value "{}"'.format(project))

    logger.info(
        'Performing %sum %s reduction for %s on %s (observation %i, %s)',
        filter_, reductiontype, source, date, obsnum,
        ('GBS' if is_gbs else 'transient'))

    # Set wavelength-dependent parameters up.
    prepare_kwargs = get_prepare_parameters(filter_)
    match_kwargs = get_match_parameters(filter_)

    # Get dimmconfig, reference and masks.
    logger.debug('Identifying dimmconfig, mask and reference files')
    if dimmconfig is None:
        dimmconfig = dimmconfigdict[reductiontype]
    dimmconfig = os.path.expandvars(dimmconfig)
    if not os.path.exists(dimmconfig):
        raise Exception('Dimmconfig file "{}" not found'.format(dimmconfig))

    mask_reductiontype = maskdict.get(reductiontype)
    if mask_reductiontype is not None:
        # Use the appropriate mask as the reference image.
        reference = get_filename_mask(field_name, filter_, mask_reductiontype)
        if not os.path.exists(reference):
            raise Exception('Mask file "{}" not found'.format(reference))
    else:
        # Use the general reference image as we don't need a mask.
        reference = get_filename_reference(field_name, filter_)
        if not os.path.exists(reference):
            raise Exception('Reference file "{}" not found'.format(reference))
    reference = shutil.copy(reference, '.')

    logger.debug('Checking configuration file "%s" exists', param_file)
    if not os.path.exists(param_file):
        raise Exception('Configuration file "{}" not found'.format(param_file))
    param_file_copy = shutil.copy(param_file, '.')

    # Create list of input files
    filelist = tempfile.NamedTemporaryFile(
        mode='w', prefix='tmpList', delete=False)
    filelist.file.writelines([i + '\n' for i in inputfiles])
    filelist.file.close()

    # Prepare environment.
    os.environ['SMURF_THREADS'] = '16'

    output_files = []

    if offsetsfile is None:
        # Identify reference catalog.
        refcat = get_filename_ref_cat(field_name, filter_, reductiontype)
        if os.path.exists(refcat):
            # If it already exists, don't try to install a new one.
            install_catalog_as_ref = False
            refcat = shutil.copy(refcat, '.')
        elif not install_catalog_as_ref:
            # Raise an exception only if we're not making a new ref catalog.
            raise Exception('Reference catalog "{}" not found'.format(refcat))

        # Create output file name.
        out = get_filename_output(
            source, date, obsnum, filter_, reductiontype, False, is_gbs)

        # run makemap
        logger.debug('Running MAKEMAP, output: "%s"', out)
        subprocess.check_call(
            [
                os.path.expandvars('$SMURF_DIR/makemap'),
                'in=^{}'.format(filelist.name),
                'config=^{}'.format(dimmconfig),
                'out={}'.format(out),
                'ref={}'.format(reference),
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
        run_gaussclumps(prepared_file, param_file_copy)
        sourcecatalog = prepared_file[:-4] + '_log.FIT'

        if not os.path.exists(sourcecatalog):
            raise Exception(
                'CUPID did not generate catalog "{}"'.format(sourcecatalog))

        if install_catalog_as_ref:
            # Install source catalog as reference.
            shutil.copyfile(sourcecatalog, refcat)

            # Use as the reference in this run.
            refcat = sourcecatalog

        # Calculate offsets with J. Lane's source_match
        logger.debug('Performing source match')
        results = source_match(sourcecatalog, refcat, **match_kwargs)
        xoffset = results[0][0]
        yoffset = results[0][1]

        if (xoffset is None) or (yoffset is None):
            raise Exception('Pointing offsets not found')

        # Create the pointing offset file.
        offsetsfile = out[:-4] + '_offset.txt'
        create_pointing_offsets(
            offsetsfile, xoffset, yoffset, system='TRACKING')

        # Apply FCF calibration.
        out_cal = out[:-4] + '_cal.sdf'
        logger.debug('Calibrating file "%s" (making "%s")', out, out_cal)
        subprocess.check_call(
            [
                os.path.expandvars('$KAPPA_DIR/cmult'),
                'in={}'.format(out),
                'out={}'.format(out_cal),
                'scalar={}'.format(get_fcf_arcsec(filter_) * 1000.0),
            ],
            shell=False)

        output_files.extend([offsetsfile, out_cal, sourcecatalog])

        output_files.extend(create_png_previews(out))

    # Re reduce map with pointing offset.
    out = get_filename_output(
        source, date, obsnum, filter_, reductiontype, True, is_gbs)

    logger.debug('Running MAKEMAP, output: "%s"', out)
    subprocess.check_call(
        [
            os.path.expandvars('$SMURF_DIR/makemap'),
            'in=^{}'.format(filelist.name),
            'config=^{}'.format(dimmconfig),
            'out={}'.format(out),
            'ref={}'.format(reference),
            'pointing={}'.format(offsetsfile),
            'msg_filter=none',
        ],
        shell=False)

    if not os.path.exists(out):
        raise Exception('MAKEMAP did not generate output "{}"'.format(out))

    # Re run Lane's smoothing and gauss clumps routine.
    logger.debug('Preparing image')
    prepare_image(out, **prepare_kwargs)
    prepared_file = out[:-4]+'_crop_smooth_jypbm.sdf'

    # Identify the sources run J. Lane's run_gaussclumps routine.
    logger.debug('Running CUPID')
    run_gaussclumps(prepared_file, param_file_copy)
    sourcecatalog = prepared_file[:-4] + '_log.FIT'

    if os.path.exists(sourcecatalog):
        if expect_missing_catalog:
            raise Exception(
                'CUPID unexpectedly generated catalog "{}"'.format(
                    sourcecatalog))
        output_files.append(sourcecatalog)

    elif not expect_missing_catalog:
        raise Exception(
            'CUPID did not generate catalog "{}"'.format(sourcecatalog))

    # Apply FCF calibration.
    out_cal = out[:-4] + '_cal.sdf'
    logger.debug('Calibrating file "%s" (making "%s")', out, out_cal)
    subprocess.check_call(
        [
            os.path.expandvars('$KAPPA_DIR/cmult'),
            'in={}'.format(out),
            'out={}'.format(out_cal),
            'scalar={}'.format(get_fcf_arcsec(filter_) * 1000.0),
        ],
        shell=False)

    output_files.append(out_cal)

    output_files.extend(create_png_previews(out))

    return output_files


def get_fcf_arcsec(filter_):
    if filter_ == '850':
        fcf_arcsec = 2.34
    elif filter_ == '450':
        fcf_arcsec = 4.71
    else:
        raise Exception('Wavelength "{}" not recognised'.format(filter_))

    return fcf_arcsec


def get_prepare_parameters(filter_):
    fcf_arcsec = get_fcf_arcsec(filter_)

    if filter_ == '850':
        beam_fwhm = 14.5
    elif filter_ == '450':
        beam_fwhm = 9.8
    else:
        raise Exception('Wavelength "{}" not recognised'.format(filter_))

    jypbm_conv = fcf_arcsec * 1.133 * (beam_fwhm ** 2.0)

    kernel_copy = shutil.copy(kernel, '.')

    return {
        'kern_name': kernel_copy,
        'kern_fwhm': kernel_fwhm,
        'jypbm_conv': jypbm_conv,
        'beam_fwhm': beam_fwhm,
    }


def get_match_parameters(filter_):
    if filter_ == '850':
        pix_scale = 3.0
    elif filter_ == '450':
        pix_scale = 2.0
    else:
        raise Exception('Wavelength "{}" not recognised'.format(filter_))

    return {
        'maxrad': 30,
        'pix_scale': pix_scale,
    }


def create_pointing_offsets(offsetfile, x, y, system='TRACKING'):
    with open(offsetfile, 'w') as f:
        f.write('# SYSTEM={}\n'.format(system))
        f.write('#TAI DLON DLAT\n')
        f.write('1 {} {}\n'.format(x, y))
        f.write('10000000 {} {}\n'.format(x, y))


def create_png_previews(filename, resolutions=[64, 256, 1024], tries=10):
    previews = []

    for resolution in resolutions:
        preview = '{}_{}.png'.format(filename[:-4], resolution)

        logger.debug(
            'Making preview (%i) of file %s', resolution, filename)

        subprocess.check_call(
            [
                '/bin/bash',
                os.path.expandvars('$ORAC_DIR/etc/picard_start.sh'),
                'CREATE_PNG',
                '--recpars=RESOLUTION={}'.format(resolution),
                '--log', 's', '--nodisp',
                filename,
            ],
            shell=False)

        if not os.path.exists(preview):
            raise Exception(
                'Failed to make preview {}'.format(preview))

        previews.append(preview)

    return previews


def safe_object_name(name):
    """
    Make safe version of an object name for use in the construction of
    file names, attempting to follow the scheme used by the survey.
    """

    # Remove spaces before numbers.
    name = re.sub(' +(?=[0-9])', '', name)

    # Remove unexpected characters.
    name = re.sub('[^_A-Za-z0-9]', '_', name)

    # Adjust name of OMC2-3.
    name = re.sub('^OMC2_3', 'OMC23', name)

    # Return in upper case.
    return name.upper()


def get_filename_mask(source, filter_, reductiontype):
    return os.path.join(
        data_dir, 'mask',
        '{}_extmask_{}_{}.sdf'.format(source, filter_, reductiontype))


def get_filename_reference(source, filter_):
    return os.path.join(
        data_dir, 'reference',
        '{}_reference_{}.sdf'.format(source, filter_))


def get_filename_ref_cat(source, filter_, reductiontype):
    return os.path.join(
        data_dir, 'cat',
        '{}_reference_cat_{}_{}.fits'.format(source, filter_, reductiontype))


def get_filename_output(source, date, obsnum, filter_, reductiontype, aligned,
                        is_gbs):
    # Change 1st letter of reduction type ('R') to 'A' for aligned map.
    if aligned:
        reductiontype = 'A' + reductiontype[1:]

    # Add 'E' (for EAO) prefix to reduction type, or 'G' for GBS data.
    survey_code = 'G' if is_gbs else 'E'

    return '{}_{}_{:05d}_{}_{}{}.sdf'.format(
        source, date, obsnum, filter_, survey_code, reductiontype)
