from starlink import kappa, cupid, smurf
import logging
logging.basicConfig(level=logging.DEBUG)


from TCOffsetFunctions import source_match
from TCGaussclumpsFunctions import run_gaussclumps
from TCPrepFunctions import prepare_image

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
    'R1': r1dimmconfig,
    'R2': r2dimmconfig,
    'R3': r3dimmconfig,
    'R4': r4dimmconfig,
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
    # TODO: switch this to use starlink.ndf (more efficient).
    res = kappa.fitsval(ndf=inputfiles[0], keyword='OBJECT')
    source = res.value

    res = kappa.fitsval(ndf=inputfiles[0], keyword='UTDATE')
    date = res.value

    res = kappa.fitsval(ndf=inputfiles[0], keyword='OBSNUM')
    obsnum = res.value

    # Get dimmconfig, reference and masks.
    dimmconfig = dimmconfigdict[reductiontype]
    mask2 = maskdict[source][reductiontype]
    reference = refdict[source]
    refcat = refcatdict[source]

    # Create output file name.
    out = outputfile1.format(source, date, obsnum, reductiontype)

    # Create list of input files
    filelist = tempfile.NamedTemporaryFile(mode='w', prefix='tmpList', delete=False)
    filelist.file.writelines([i + '\n' for i in inputfiles])
    filelist.file.close()

    # run makemap
    makemapres = smurf.makemap(in_='^' + filelist.name,
                               config='^' + dimmconfig,
                               out=out,
                               ref=reference,
                               mask2=mask2)

    # Prepare the image (smoothing etc) by running J. Lane's prepare image routine.
    prepare_image(out, kernel, kernel_fwhm)
    prepared_file = out[:-4]+'_smooth_jybpm.sdf'

    # Identify the sources run J. Lane's run_gaussclumps routine.
    run_gaussclumps(prepared_file, param_file)
    sourcecatalog = prepared_file[:-4] + '_log.FIT'


    # Calculate offsets with J. Lane's source_match
    results = source_match(sourcecatalog, refcat, minpeak=0.2, maxrad=30, maxsep=10, cutoff=4, pix_scale=3.0)
    xoffset = results[0][1]
    yoffset = results[0][2]

    # Create the pointing offset file.
    offsetsfile = create_pointing_offsets(xoffset, yoffset, system='TRACKING')

    # Re reduce map with pointing offset.
    out = outputfile2.format(source, date, obsnum, reductiontype)
    makemapres2 = smurf.makemap(in'^' + filelist.name,
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
    f = open(offsetfile, 'w')
    f.write('# SYSTEM={}\n'.format(system))
    f.write('#TAI DLON DLAT\n')
    f.write('1 {} {}\n'.format(x, y))
    f.write('10000000 {} {}\n'.format(x, y))
    f.close()
    return offsetfile
