from starlink import kappa, cupid, smurf
import os
import logging
import re
logging.basicConfig(level=logging.DEBUG)

from astropy.io import fits
from starlink.ndfpack import Ndf

from transientclumps.TCOffsetFunctions import source_match
from transientclumps.TCGaussclumpsFunctions import run_gaussclumps
from transientclumps.TCPrepFunctions import prepare_image

# Look up information -- this needs full paths possibbly.
# This should probably be replaced by a better config system?

# Dictionary, source name as key. Value is reference map

refdict = {
    'IC348': 'referenceimages/IC348_reference_850.sdf',
}

refcatdict = {
    'IC348': 'referencecats/IC348_reference_cat_850.FITS',
}


# dictionary of 'mask2' values, by run name.
maskdict = {
    'IC348': {'R1': '!',
              'R2': '!',
              'R3': 'externalmasks/IC348_R3_extmask.sdf',
              'R4': 'externalmasks/IC348_R4_extmask.sdf',
              },
}


dimmconfigdict = {
    'R1': '$ORAC_CAL_ROOT/scuba2/dimmconfig_M16AL001_R1.lis',
    'R2': '$ORAC_CAL_ROOT/scuba2/dimmconfig_M16AL001_R2.lis',
    'R3': '$ORAC_CAL_ROOT/scuba2/dimmconfig_M16AL001_R3.lis',
    'R4': '$ORAC_CAL_ROOT/scuba2/dimmconfig_M16AL001_R4.lis',
}

outputfile1 = '{}_{}_{:05d}_8{}_pW_nopointcorr.sdf'
outputfile2 = '{}_{}_{:05d}_8{}_pW_aligned.sdf'

param_file = 'the parameter file?'

kernel = 'kernel.sdf'
kernel_fwhm = 5.0


def transient_analysis(inputfiles, reductiontype):
    """
    Take in a list of input files from a single 850um observation and
    the reduction type (e.g. 'R1', 'R2' etc).

    Returns the filename of the reduced maps (with and without
    pointing corrections) and the output source catalog (after
    pointing corrections).

    """

    # Get source, utdate and obsnum.
    header = fits.Header.fromstring(''.join(Ndf(inputfiles[0]).head['FITS']))
    source = safe_object_name(header['OBJECT'])
    date = header['UTDATE']
    obsnum = header['OBSNUM']

    # Get dimmconfig, reference and masks.
    dimmconfig = os.path.expandvars(dimmconfigdict[reductiontype])
    mask2 = maskdict[source][reductiontype]
    reference = refdict[source]
    refcat = refcatdict[source]

    # Create output file name.
    out = outputfile1.format(source, date, obsnum, reductiontype)

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

    # Prepare the image (smoothing etc) by running J. Lane's
    # prepare image routine.
    prepare_image(out, kernel, kernel_fwhm)
    prepared_file = out[:-4]+'_smooth_jybpm.sdf'

    # Identify the sources run J. Lane's run_gaussclumps routine.
    run_gaussclumps(prepared_file, param_file)
    sourcecatalog = prepared_file[:-4] + '_log.FIT'

    # Calculate offsets with J. Lane's source_match
    results = source_match(
        sourcecatalog, refcat, minpeak=0.2, maxrad=30, maxsep=10,
        cutoff=4, pix_scale=3.0)
    xoffset = results[0][1]
    yoffset = results[0][2]

    # Create the pointing offset file.
    offsetsfile = create_pointing_offsets(xoffset, yoffset, system='TRACKING')

    # Re reduce map with pointing offset.
    out = outputfile2.format(source, date, obsnum, reductiontype)
    makemapres2 = smurf.makemap(in_='^' + filelist.name,
                                config='^' + dimmconfig,
                                out=out,
                                ref=reference,
                                mask2=mask2,
                                pointing=offsetsfile)

    # Re run Lane's smoothing and gauss clumps routine.
    prepare_image(out)
    prepared_file = out[:-4]+'_smooth+jybpm.sdf'

    # Identify the sources run J. Lane's run_gaussclumps routine.
    run_gaussclumps(prepared_file, param_file)
    sourcecatalog = prepared_file[:-4] + '_log.FIT'

    return out, sourcecatalog


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
