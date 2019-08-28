from __future__ import absolute_import

from codecs import ascii_decode
from collections import defaultdict
import json
import os
import logging
import re
import shutil
import subprocess
import sys
import tempfile

from astropy.io import ascii as ap_ascii
from astropy.io import fits
from starlink.ndfpack import Ndf

from transientclumps.TCOffsetFunctions import source_match
from transientclumps.TCGaussclumpsFunctions import run_gaussclumps
from transientclumps.TCPrepFunctions import prepare_image, \
    crop_image, unitconv_image
from transientfluxcal.fluxcal import \
    find_calibration_factors, merge_catalog, extract_brightnessess_from_cat
from transientfluxcal.trigger import \
    analyse_sources, extract_central_var, extract_pixel_values, \
    make_metadata_table, merge_yso_catalog

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
                       dimmconfig_850=None, dimmconfig_450=None,
                       fixed_dra=None, fixed_ddec=None, mask_suffix=None):
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

    # Check for fixed offsets.
    offsetsfile = None
    is_kevin = False
    if (fixed_dra is not None) and (fixed_ddec is not None):
        offsetsfile = 'fixed_offset.txt'
        create_pointing_offsets(
            offsetsfile, fixed_dra, fixed_ddec, system='TRACKING')
        is_kevin = True

    elif (fixed_dra is not None) or (fixed_ddec is not None):
        raise Exception('Only one of "dra" and "ddec" was supplied')

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

    output_files = []

    if subsystems[850]:
        logger.debug('Performing 850um analysis')
        output_files.extend(transient_analysis_subsystem(
            subsystems[850], reductiontype, '850', offsetsfile,
            install_catalog_as_ref=as_ref_cat,
            dimmconfig=dimmconfig_850, is_kevin=is_kevin))

    elif offsetsfile is None:
        raise Exception('No 850um data files or fixed offsets given')

    if offsetsfile is None:
        # Offsets file should have been first given.
        offsetsfile = output_files[0]

    if subsystems[450]:
        logger.debug('Performing 450um analysis')
        output_files.extend(transient_analysis_subsystem(
            subsystems[450], reductiontype, '450', offsetsfile,
            expect_missing_catalog=no_450_cat,
            dimmconfig=dimmconfig_450, is_kevin=is_kevin))

    return output_files


def transient_analysis_subsystem(inputfiles, reductiontype, filter_,
                                 offsetsfile, expect_missing_catalog=False,
                                 install_catalog_as_ref=False,
                                 dimmconfig=None, is_kevin=False):
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

    is_gbs = False
    survey_code = None
    project = header['PROJECT']
    if (project == 'M16AL001') or re.match('M\d\d[AB]EC30', project):
        field_name = source
    elif project.startswith('MJLSG'):
        is_gbs = True
        field_name = gbs_field_name(raw_source)
        survey_code = 'G'
    elif project in ('M17BP054', 'M18AP017'):
        field_name = source
        survey_code = 'H'
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
        reference = get_filename_mask(
            source, filter_, mask_reductiontype, is_gbs, suffix=mask_suffix)
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
    os.environ['ADAM_EXIT'] = '1'

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
            source, date, obsnum, filter_, reductiontype, False,
            survey_code, is_kevin)

        # run makemap
        logger.debug('Running MAKEMAP, output: "%s"', out)
        sys.stderr.flush()
        subprocess.check_call(
            [
                os.path.expandvars('$SMURF_DIR/makemap'),
                'in=^{}'.format(filelist.name),
                'config=^{}'.format(dimmconfig),
                'out={}'.format(out),
                'ref={}'.format(reference),
                'msg_filter=normal',
            ],
            shell=False,
            stdout=sys.stderr)

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
        sys.stderr.flush()
        subprocess.check_call(
            [
                os.path.expandvars('$KAPPA_DIR/cmult'),
                'in={}'.format(out),
                'out={}'.format(out_cal),
                'scalar={}'.format(get_fcf_arcsec(filter_) * 1000.0),
            ],
            shell=False,
            stdout=sys.stderr)
        subprocess.check_call(
            [
                os.path.expandvars('$KAPPA_DIR/setunits'),
                'ndf={}'.format(out_cal),
                'units=mJy/arcsec**2',
            ],
            shell=False,
            stdout=sys.stderr)

        output_files.extend([offsetsfile, out_cal, sourcecatalog])

        output_files.extend(create_png_previews(out))

    elif not offsetsfile.endswith('_offset.txt'):
        raise Exception(
            'File "{}" does not look like an offsets file'.format(
                offsetsfile))

    # Re reduce map with pointing offset.
    out = get_filename_output(
        source, date, obsnum, filter_, reductiontype, True, survey_code,
        is_kevin)

    logger.debug('Running MAKEMAP, output: "%s"', out)
    sys.stderr.flush()
    subprocess.check_call(
        [
            os.path.expandvars('$SMURF_DIR/makemap'),
            'in=^{}'.format(filelist.name),
            'config=^{}'.format(dimmconfig),
            'out={}'.format(out),
            'ref={}'.format(reference),
            'pointing={}'.format(offsetsfile),
            'msg_filter=normal',
        ],
        shell=False,
        stdout=sys.stderr)

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
    sys.stderr.flush()
    subprocess.check_call(
        [
            os.path.expandvars('$KAPPA_DIR/cmult'),
            'in={}'.format(out),
            'out={}'.format(out_cal),
            'scalar={}'.format(get_fcf_arcsec(filter_) * 1000.0),
        ],
        shell=False,
        stdout=sys.stderr)
    subprocess.check_call(
        [
            os.path.expandvars('$KAPPA_DIR/setunits'),
            'ndf={}'.format(out_cal),
            'units=mJy/arcsec**2',
        ],
        shell=False,
        stdout=sys.stderr)

    output_files.append(out_cal)

    output_files.extend(create_png_previews(out))

    return output_files


def transient_flux_calibration(inputfiles, filter_='850'):
    pattern = re.compile('^(.*)_(\d{8})_(\d{5})_(850|450)_E(A\d)$')
    r_pattern = re.compile('^(.*)_(\d{8})_(\d{5})_(850|450)_E(R\d)$')
    output_files = []
    survey_code = 'E'

    # Determine calibration "method" based on filter.
    method_cat = True
    date_cutoff = None
    if filter_ == '450':
        method_cat = False
        date_cutoff = '20190815'
    elif filter_ == '850':
        date_cutoff = '20170301'

    # Read "family" file.
    with open(get_filename_cal_family(filter_), 'r') as f:
        family_data = json.load(f)

    if method_cat:
        with open(get_filename_id_mapping(), 'r') as f:
            family_data_mapping = json.load(f)
    else:
        # We don't use an existing catalog, so there is no mapping.
        family_data_mapping = None

    prepare_kwargs = get_prepare_parameters(filter_, fcf_arcsec=0.001)

    # Find source lists used by the "trigger" routines.
    disk_cat = get_filename_source_list('disks')
    if not os.path.exists(disk_cat):
        raise Exception('Disk source list "{}" not found'.format(disk_cat))
    disk_cat = shutil.copy(disk_cat, '.')

    prot_cat = get_filename_source_list('protostars')
    if not os.path.exists(prot_cat):
        raise Exception('Protostar source list "{}" not found'.format(prot_cat))
    prot_cat = shutil.copy(prot_cat, '.')

    with open(get_filename_special_names(), 'r') as f:
        special_names = json.load(f)

    # Organize files into SDF and catalog lists.
    input_map = {}
    input_cat = {}
    input_off = {}

    for file_ in inputfiles:
        target = None
        any_filter = False

        if file_.endswith('_cat.fits'):
            target = input_cat
            match = pattern.match(file_[:-9])

        elif file_.endswith('.sdf'):
            target = input_map
            match = pattern.match(file_[:-4])

        elif file_.endswith('_offset.txt'):
            target = input_off
            match = r_pattern.match(file_[:-11])
            # There is only one offset file -- use it for both filters.
            any_filter = True

        else:
            raise Exception('Unexpected file type: "{}"'.format(file_))

        if not match:
            raise Exception('Did not understand file name "{}"'.format(file_))

        if target is None:
            raise Exception('Target not set')

        if (match.group(4) != filter_) and not any_filter:
            continue

        target[match.groups()] = file_

    # Pair up the inputs and organize by field
    inputs = defaultdict(list)
    inv_var_sum = 0.0
    for (key, map_) in sorted(input_map.items()):
        if method_cat:
            cat = input_cat.pop(key, None)
            if cat is None:
                raise Exception('File "{}" has no matching catalog'.format(map_))
        else:
            cat = None

        info = {'map': map_, 'cat': cat}
        info.update(zip((
            'field_name', 'date', 'obsnum', 'filter', 'reductiontype',
        ), key))

        reductiontype_orig = re.sub('^A', 'R', info['reductiontype'])
        key_off = key[:-2] + ('850', reductiontype_orig,)
        off = input_off.pop(key_off, None)
        if off is None:
            raise Exception('File "{}" has no matching offset file'.format(map_))

        info.update(read_pointing_offsets(off))

        if not method_cat:
            var = extract_central_var(Ndf(map_), int(filter_))
            logger.info('Estimated variance of file %s: %f', map_, var)
            info['var'] = var

            # Only add to sum if before the cut-off date.
            if info['date'] <= date_cutoff:
                inv_var_sum += 1.0 / var

        key = (info.pop('field_name'), info.pop('reductiontype'))

        inputs[key].append(info)

    var_thresh = None
    if inv_var_sum != 0.0:
        var_coadd = 1.0 / inv_var_sum
        # 9 is factor by which this method seems to underestimate the noise in the coadd
        var_thresh = 9 * 11.111 * var_coadd
        logger.info('Variance threshold: %f', var_thresh)

    if method_cat and input_cat:
        raise Exception('Catalogs {} have no matching image'.format(repr(list(input_cat.keys()))))

    for (key, input_) in inputs.items():
        (field_name, reductiontype) = key

        # Get published catalog.
        pub_cat = get_filename_pub_cat(field_name, filter_)
        if not os.path.exists(pub_cat):
            raise Exception('Published catalog "{}" not found'.format(pub_cat))
        pub_cat = shutil.copy(pub_cat, '.')

        pub_cat_data = fits.getdata(pub_cat)

        # Wheech out noisy maps.
        if var_thresh is not None:
            input_filtered = []
            for info in input_:
                if info['var'] > var_thresh:
                    logger.info('Skipping map %s: variance too large', info['map'])
                else:
                    input_filtered.append(info)

            input_ = input_filtered

        # Sort inputs by date and obsnum.
        input_.sort(key=lambda x: x['obsnum'])
        input_.sort(key=lambda x: x['date'])

        # Get calibration source list.
        good_sources = family_data[field_name][reductiontype]

        if method_cat:
            (fluxcal_outputs, culled, all_peak_brightnesses) = perform_fluxcal_from_cat(
                input_, good_sources, field_name, filter_, reductiontype, survey_code)

            output_files.extend(fluxcal_outputs)
            input_ = culled

        else:
            all_peak_brightnesses = perform_fluxcal_via_extract(
                input_, good_sources, filter_, pub_cat_data)

        if not input_:
            continue

        # Perform flux calibration.
        (calibration_factors, calibration_factor_errors) = \
            find_calibration_factors(
                observations=input_,
                all_peak_brightnesses=all_peak_brightnesses,
                date_cutoff=date_cutoff)

        log_lines = []
        maps_cal = []
        maps_smooth = []
        observations_calibrated = []
        valid_calibration_factors = []
        valid_calibration_factor_errors = []

        for (observation, calibration_factor, calibration_factor_error) in zip(
                input_, calibration_factors, calibration_factor_errors):
            if calibration_factor is None:
                logger.warning('No calibration factor for "%s"', map)
                continue

            if not calibration_factor > 0:
                logger.error('Calibration failed for "%s"', map)
                continue

            map_ = observation['map']
            map_cal = map_[:-4] + '_cal.sdf'
            logger.debug('Calibrating file "%s" (making "%s")', map_, map_cal)
            sys.stderr.flush()
            subprocess.check_call(
                [
                    os.path.expandvars('$KAPPA_DIR/cdiv'),
                    'in={}'.format(map_),
                    'out={}'.format(map_cal),
                    'scalar={}'.format(calibration_factor),
                ],
                shell=False,
                stdout=sys.stderr)

            # Apply units correction -- some of the products in the system
            # were reduced before we started setting these units.  Set them
            # now so that this product and the coadd get the correct units.
            # (We can remove this if were were to re-reduce everything with
            # units set in the products.)
            subprocess.check_call(
                [
                    os.path.expandvars('$KAPPA_DIR/setunits'),
                    'ndf={}'.format(map_cal),
                    'units=mJy/arcsec**2',
                ],
                shell=False,
                stdout=sys.stderr)

            if filter_ != '450':
                prepare_image(map_cal, **prepare_kwargs)
            else:
                prepare_image_via_gaussian_smooth(
                    map_cal, pixels =2, kern_fwhm=4,
                    jypbm_conv=prepare_kwargs['jypbm_conv'],
                    beam_fwhm=prepare_kwargs['beam_fwhm'])

            map_smooth = map_cal[:-4]+'_crop_smooth_jypbm.sdf'

            subprocess.check_call(
                [
                    os.path.expandvars('$KAPPA_DIR/setunits'),
                    'ndf={}'.format(map_smooth),
                    'units=Jy/beam',
                ],
                shell=False,
                stdout=sys.stderr)

            maps_cal.append(map_cal)
            maps_smooth.append(map_smooth)
            output_files.extend([map_cal, map_smooth])

            observations_calibrated.append(observation)
            valid_calibration_factors.append(calibration_factor)
            valid_calibration_factor_errors.append(calibration_factor_error)

            log_lines.append([
                field_name,
                observation['date'],
                observation['obsnum'],
                filter_,
                reductiontype,
                calibration_factor,
                calibration_factor_error,
            ])

        log_file = '{}_{}_{}{}_cal_factor.txt'.format(
            field_name, filter_, survey_code, reductiontype)

        with open(log_file, 'w') as f:
            for line in log_lines:
                print(*line, file=f)

        output_files.append(log_file)

        # Create co-adds of calibrated and smoothed maps.
        coadd_cal = '{}_{}_{}{}_cal_coadd.sdf'.format(
            field_name, filter_, survey_code, reductiontype)
        create_coadded_map(maps_cal, coadd_cal)
        output_files.append(coadd_cal)
        output_files.extend(create_png_previews(coadd_cal))

        coadd_smooth = '{}_{}_{}{}_cal_smooth_coadd.sdf'.format(
            field_name, filter_, survey_code, reductiontype)
        create_coadded_map(maps_smooth, coadd_smooth)
        output_files.append(coadd_smooth)
        output_files.extend(create_png_previews(coadd_smooth))

        # Begin "trigger" analysis -- starting with metadata table.
        metadata = make_metadata_table(
            maps=maps_smooth, observations=observations_calibrated,
            calibration_factors=valid_calibration_factors,
            calibration_factor_errors=valid_calibration_factor_errors,
            field_name=field_name, reduction_type=reductiontype, filter_=filter_)

        metadata_file = '{}_{}_{}{}_cal_metadata.txt'.format(
            field_name, filter_, survey_code, reductiontype)
        ap_ascii.write(metadata, output=metadata_file, format='commented_header')
        output_files.append(metadata_file)

        # Prepare merged YSO catalog.
        yso_cat = merge_yso_catalog(pub_cat_data, disk_cat, prot_cat)
        ap_ascii.write(yso_cat, output='yso_compare.txt', format='commented_header')

        # Extract fluxes of sources from the smoothed maps.
        map_fluxes = []

        for (observation, map_smooth) in zip(observations_calibrated, maps_smooth):
            map_smooth_ndf = Ndf(map_smooth)

            map_fluxes.append(
                extract_pixel_values(map_smooth_ndf, pub_cat_data, filter_))

        # Run triggering analysis.
        (trigger_table, trigger_sources, trigger_text,
            lightcurves, lightcurves_triggered) = analyse_sources(
                field_name=field_name,
                observations=observations_calibrated,
                metadata=metadata,
                calibration_factors=valid_calibration_factors,
                calibration_factor_errors=valid_calibration_factor_errors,
                yso_cat=yso_cat,
                map_fluxes=map_fluxes,
                filter_=filter_,
                special_names=special_names,
                calibration_ids=good_sources,
                calibration_id_mapping=(None if family_data_mapping is None
                    else family_data_mapping[field_name]),
                lightcurve_prefix='{}_{}_{}{}'.format(
                    field_name, filter_, survey_code, reductiontype))

        output_files.extend(lightcurves)

        message_file = '{}_{}_{}{}_variables.txt'.format(
            field_name, filter_, survey_code, reductiontype)
        with open(message_file, 'w') as f:
            for line in trigger_text:
                print(line, file=f)

        message_file_attach = message_file[:-4] + '_attach.txt'
        with open(message_file_attach, 'w') as f:
            for lightcurve in lightcurves_triggered:
                print(lightcurve, file=f)

        output_files.extend([message_file, message_file_attach])

        source_info_file = '{}_{}_{}{}_source_info.txt'.format(
            field_name, filter_, survey_code, reductiontype)
        ap_ascii.write(trigger_table, output=source_info_file, format='commented_header')
        output_files.append(source_info_file)

    return output_files


def perform_fluxcal_from_cat(input_, good_sources, field_name, filter_, reductiontype, survey_code):
    output_files = []
    culled = []

    # Merge each catalog with the reference.
    for info in input_:
        obsnum = int(info['obsnum'])

        (cat_match, cat_cull) = merge_observation_catalogs(
            field_name=field_name, date=info['date'],
            obsnum=obsnum, filter_=filter_,
            reductiontype=reductiontype, cat=info['cat'],
            survey_code=survey_code)

        info['culled'] = cat_cull
        culled.append(info)
        output_files.append(cat_match)
        output_files.append(cat_cull)

    (culled, all_peak_brightnesses) = extract_brightnessess_from_cat(culled, good_sources)

    return (output_files, culled, all_peak_brightnesses)


def perform_fluxcal_via_extract(input_, good_sources, filter_, cat):
    all_peak_brightnesses = []

    for observation in input_:
        # Smooth to 4" (this is 2 x 2" pixels for the 450um maps).
        smooth = apply_gaussian_smooth(observation['map'], pixels=2, suffix='smoothuncal')

        # Extract pixels.
        values = extract_pixel_values(Ndf(smooth), cat, filter_)

        # Get "good" brightness values.
        brightnesses = []
        for id_ in good_sources:
            brightnesses.append(values[id_])

        all_peak_brightnesses.append(brightnesses)

    return all_peak_brightnesses


def merge_observation_catalogs(
        field_name, date, obsnum, filter_, reductiontype, cat, survey_code):
    reductiontype_orig = re.sub('^A', 'R', reductiontype)
    if reductiontype_orig not in dimmconfigdict:
        raise Exception(
            'Unrecognised reduction type "{}"'.format(reductiontype))

    logger.info(
        'Performing %sum %s catalog merge for %s on %s (observation %i)',
        filter_, reductiontype, field_name, date, obsnum)

    # Re-do source match to gather information.
    refcat = get_filename_ref_cat(field_name, filter_, reductiontype_orig)
    if not os.path.exists(refcat):
        raise Exception('Reference catalog "{}" not found'.format(refcat))

    refcat = shutil.copy(refcat, '.')

    match_kwargs = get_match_parameters(filter_)

    match_results = source_match(cat, refcat, **match_kwargs)

    cat_match = '{}_{}_{:05d}_{}_{}{}_match.FIT'.format(
        field_name, date, obsnum, filter_, survey_code, reductiontype)

    merge_catalog(
        cat, refcat, date, obsnum, cat_match,
        ref_index=match_results[4],
        cat_index=match_results[5])

    cat_cull = '{}_{}_{:05d}_{}_{}{}_cull.FIT'.format(
        field_name, date, obsnum, filter_, survey_code, reductiontype)

    merge_catalog(
        cat, refcat, date, obsnum, cat_cull,
        ref_index=match_results[6],
        cat_index=match_results[7])

    return (cat_match, cat_cull)


def get_fcf_arcsec(filter_):
    if filter_ == '850':
        fcf_arcsec = 2.34
    elif filter_ == '450':
        fcf_arcsec = 4.71
    else:
        raise Exception('Wavelength "{}" not recognised'.format(filter_))

    return fcf_arcsec


def get_prepare_parameters(filter_, fcf_arcsec=None):
    if fcf_arcsec is None:
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


def read_pointing_offsets(offsetfile):
    system = None
    x = None
    y = None

    with open(offsetfile, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('#'):
                m = re.match('# SYSTEM=(.*)', line)
                if m:
                    system = m.group(1)

            else:
                words = line.split(' ')
                if len(words) == 3:
                    x = float(words[1])
                    y = float(words[2])

    if system is None or x is None or y is None:
        raise Exception(
            'Did not understand offset file "{}"'.format(offsetfile))

    return {
        "offset_system": system,
        "offset_x": x,
        "offset_y": y,
    }


def apply_gaussian_smooth(filename, pixels, suffix='smooth'):
    out = filename[:-4] + '_{}.sdf'.format(suffix)

    logger.debug('Running GAUSMOOTH, output: "%s"', out)
    sys.stderr.flush()
    subprocess.check_call(
        [
            os.path.expandvars('$KAPPA_DIR/gausmooth'),
            'in={}'.format(filename),
            'out={}'.format(out),
            'fwhm={}'.format(pixels),
        ],
        shell=False,
        stdout=sys.stderr)

    if not os.path.exists(out):
        raise Exception('GAUSMOOTH did not generate output "{}"'.format(out))

    return out


# Version of TCPrepFunctions.prepare image which uses apply_gaussian_smooth
# instead of convolving with a kernel.
def prepare_image_via_gaussian_smooth(
        img_name, pixels, kern_fwhm, jypbm_conv,
        beam_fwhm, crop_radius=1200, crop_method='CIRCLE'):
    #Crop the image.
    crop_image(img_name, crop_radius, crop_method)

    #Smooth the image.
    img_name = img_name[:-4]+'_crop.sdf'
    img_name = apply_gaussian_smooth(img_name, pixels=pixels)

    #Convert the units. First account for the smoothing by dividing new beam
    # area by old beam area and multiplying to the Jy/Beam conversion factor.
    jypbm_conv *= ((beam_fwhm**2)+(kern_fwhm**2))/(beam_fwhm**2)
    unitconv_image(img_name, jypbm_conv)


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
            shell=False,
            stdout=sys.stderr)

        if not os.path.exists(preview):
            raise Exception(
                'Failed to make preview {}'.format(preview))

        previews.append(preview)

    return previews


def create_coadded_map(maps, filename):
    logger.debug('Creating co-add: %s', filename)

    filelist = tempfile.NamedTemporaryFile(
        mode='w', prefix='tmpList', delete=False)
    filelist.file.writelines([x + '\n' for x in maps])
    filelist.file.close()

    sys.stderr.flush()
    subprocess.check_call(
        [
            os.path.expandvars('$KAPPA_DIR/wcsmosaic'),
            'in=^{}'.format(filelist.name),
            'out={}'.format(filename),
            'lbnd=!',
            'ubnd=!',
            'ref=!',
        ],
        shell=False,
        stdout=sys.stderr)


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


def get_filename_mask(source, filter_, reductiontype, is_gbs, suffix=None):
    pattern = '{}_GBS_extmask_{}_{}{}.sdf' if is_gbs else '{}_extmask_{}_{}{}.sdf'

    suffix = '' if suffix is None else '_{}'.format(suffix)

    return os.path.join(
        data_dir, 'mask',
        pattern.format(source, filter_, reductiontype, suffix))


def get_filename_reference(source, filter_):
    return os.path.join(
        data_dir, 'reference',
        '{}_reference_{}.sdf'.format(source, filter_))


def get_filename_ref_cat(source, filter_, reductiontype):
    return os.path.join(
        data_dir, 'cat',
        '{}_reference_cat_{}_{}.fits'.format(source, filter_, reductiontype))


def get_filename_pub_cat(source, filter_, pub=None):
    if pub is None:
        if filter_ == '850':
            pub = 'dij2017'
        else:
            pub = 'stm450'

    if pub == 'dij2017':
        cat_dir = 'cat_dij2017'
        suffix = '20170616'
        extra = ''
    elif pub == 'stm450':
        cat_dir = 'cat_stm_450'
        suffix = '20190815'
        extra = '_450'
    else:
        raise Exception('Unknown publication "{}"'.format(pub))

    return os.path.join(
        data_dir, cat_dir,
        '{}{}_sourcecat_{}.fits'.format(source, extra, suffix))


def get_filename_source_list(source_type):
    return os.path.join(
        data_dir, 'trigger',
        '{}.txt'.format(source_type))


def get_filename_cal_family(filter_):
    if filter_ == '850':
        filename = 'family.json'
    else:
        filename = 'family_{}.json'.format(filter_)

    return os.path.join(data_dir, 'cal', filename)


def get_filename_special_names():
    return os.path.join(data_dir, 'trigger', 'names.json')


def get_filename_id_mapping():
    return os.path.join(data_dir, 'trigger', 'family_to_dij.json')


def get_filename_output(source, date, obsnum, filter_, reductiontype, aligned,
                        survey_code, is_kevin):
    if is_kevin:
        if aligned:
            # Change 1st letter of reduction type ('K') to 'A' for aligned map.
            reductiontype = 'K' + reductiontype[1:]
        else:
            raise Exception('Making non-aligned filename in "Kevin" mode')
    else:
        # Change 1st letter of reduction type ('R') to 'A' for aligned map.
        if aligned:
            reductiontype = 'A' + reductiontype[1:]

    # Add 'E' (for EAO) prefix to reduction type, or 'G' for GBS data.
    survey_code = 'E' if (survey_code is None) else survey_code

    return '{}_{}_{:05d}_{}_{}{}.sdf'.format(
        source, date, obsnum, filter_, survey_code, reductiontype)
