from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from datetime import datetime, timedelta
import logging
# Set a matplotlib backend before importing pyplot in case Tk isn't installed.
import matplotlib; matplotlib.use('PDF'); import matplotlib.pyplot as plt
import os
import re
import sys
import subprocess

from astropy.io import fits
import numpy as np

logger = logging.getLogger(__name__)


def make_metadata_table(
        maps, observations, calibration_factors, calibration_factor_errors,
        field_name, reduction_type, filter_):
    rows = []
    i = 0

    for (map_, observation, calibration_factor, calibration_factor_error) in zip(
            maps, observations, calibration_factors, calibration_factor_errors):
        i += 1
        stats = get_map_stats(map_)

        rows.append([
            i,
            '{}_{}_{}_{}'.format(
                field_name, observation['date'], observation['obsnum'], filter_),
            stats['UT'],
            stats['JD'],
            observation['obsnum'],
            stats['Elev'],
            stats['Tau225'],
            stats['RMS'],
            stats['RMS_unit'],
            calibration_factor,
            calibration_factor_error,
            observation['offset_x'],
            observation['offset_y'],
        ])

    return Table(
        list(zip(*rows)),
        names=(
            'ID', 'Name', 'UT', 'JD', 'Obs', 'Elev', 'Tau225',
            'RMS', 'RMS_unit', 'Cal_f', 'Cal_f_err', 'Offset_x', 'Offset_y'),
        meta={'name': 'Meta Data Table'})


def get_map_stats(filename):
    logger.debug(
        'Getting SCUBA-2 map stats for file %s', filename)

    log_file = 'log.mapstats'
    if os.path.exists(log_file):
        os.remove(log_file)

    subprocess.check_call(
        [
            '/bin/bash',
            os.path.expandvars('$ORAC_DIR/etc/picard_start.sh'),
            'SCUBA2_MAPSTATS',
            '--log', 's', '--nodisp',
            filename,
        ],
        shell=False,
        stdout=sys.stderr)

    metadata = np.loadtxt(log_file, dtype={
        'names': (
            'UT', 'HST', 'Obs', 'Source', 'Mode', 'Filter',
            'El', 'Airmass', 'Trans', 'Tau225', 'Tau', 't_elapsed',
            't_exp', 'rms', 'rms_units', 'nefd', 'nefd_units', 'RA', 'DEC',
            'mapsize', 'pixscale', 'project', 'recipe', 'filename',
        ),
        'formats': (
            'f8', '<U23', 'f8', '<U10', '<U10', 'f8',
            'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
            'f8', 'f8', '<U11', 'f8', '<U8', '<U11', '<U11',
            'f8', 'f8', '<U8', '<U11', '<U34',
        )})

    (ut_int, ut_frac) = divmod(metadata['UT'], 1)
    ut = datetime.strptime(str(int(ut_int)), '%Y%m%d') \
        + timedelta(days=ut_frac)

    return {
        'UT': ut,
        'JD': Time(ut, scale='utc').jd,
        'Elev': metadata['El'].item(),
        'Tau225': metadata['Tau225'].item(),
        'RMS': metadata['rms'].item(),
        'RMS_unit': metadata['rms_units'].item(),
    }


def extract_pixel_values(ndf, cat, filter_=850):
    values = []

    for i in range(len(cat)):
        coord = SkyCoord(cat['RA'][i], cat['DEC'][i], frame='icrs', unit='deg')

        a = ndf.wcs.tran([
            [coord.ra.radian], [coord.dec.radian],
            [float(filter_) / 1000000]], False)

        (x, y, z) = (a - 1).flatten().round().astype(int)

        # Ignore 'z' axis value (from filter_) and assume we have a 2D
        # map with a size of 1 in 'z'.
        values.append(ndf.data[0, y, x].item())

    return values


def extract_central_var(ndf, filter_=850, halfwidth=100):
    prev_frame = ndf.wcs.Current

    # Select AXIS coordinates.
    ndf.wcs.Current = 3

    a = ndf.wcs.tran([[0], [0], [0]], False)
    (x, y, z) = (a - 1).flatten().round().astype(int)

    subset = ndf.var[0,(y-halfwidth):(y+halfwidth),(x-halfwidth):(x+halfwidth)]
    var_mean = subset.mean()

    # Restore the current frame.
    ndf.wcs.Current = prev_frame

    return var_mean.item()


def merge_yso_catalog(cat, disk_cat_file, prot_cat_file):
    cat_dtypes = [('index', 'i'), ('ra', 'f8'), ('dec', 'f8'), ('class', '<U2')]

    prot_cat = np.loadtxt(prot_cat_file, dtype=cat_dtypes)
    disk_cat = np.loadtxt(disk_cat_file, dtype=cat_dtypes)

    # Generate empty lists for the information we would like to collect
    peak_int_ind = []
    peak_ind = []
    ra = []
    dec = []
    prot_distance = []
    prot_class = []
    disk_distance = []
    disk_class = []

    for i in range(len(cat)):
        peak_int_ind.append(i)
        peak_ind.append(cat['ID'][i])
        ra.append(cat['RA'][i])
        dec.append(cat['DEC'][i])

        for (yso_cat, distances, classes) in (
                [
                    (prot_cat, prot_distance, prot_class),
                    (disk_cat, disk_distance, disk_class),
                ]):
            yso_distances = np.sqrt(
                np.abs((yso_cat['ra'] - cat['RA'][i]) * np.cos(cat['DEC'][i] * np.pi / 180.0)) ** 2.0
                + np.abs(yso_cat['dec'] - cat['DEC'][i]) ** 2.0)

            closest = yso_distances.argmin()

            distances.append(3600.0 * yso_distances[closest])
            classes.append(yso_cat['class'][closest])

    return Table(
        [peak_int_ind, peak_ind, ra, dec, prot_distance, prot_class, disk_distance, disk_class],
        names=('Index', 'ID', 'RA', 'DEC', 'Proto_Dist', 'Proto_Class', 'Disk_Dist', 'Disk_Class'),
        meta={'name': 'YSO Table'})


def analyse_sources(
        field_name, observations, metadata,
        calibration_factors, calibration_factor_errors,
        yso_cat, map_fluxes,
        calibration_ids, calibration_id_mapping,
        trigger_thresh=4.0, brightness_thresh=0.150, sd_thresh=1.5,
        filter_='850', special_names={}, lightcurve_prefix='index'):
    # Prepare an Astropy table.
    columns = [
        'Index', 'ID', 'RA', 'DEC', 'proto_dist', 'disk_dist',
        'mean_peak_flux', 'sd_peak_flux', 'sd_fiducial', 'sd/sd_fiducial',
        'slope (%/year)', 'delta_slope', 'abs(slope/delta_slope)',
        'intercept', 'delta_intercept'
    ]

    dtypes = [
        'i4', '<U23', 'f8', 'f8', 'f8', 'f8',
        'f8', 'f8', 'f8', 'f8',
        'f8', 'f8', 'f8',
        'f8', 'f8',
    ]

    for observation in observations:
        columns.append('f_{}_{}'.format(
            observation['date'], observation['obsnum']))
        dtypes.append('f8')

    for observation in observations:
        columns.append('abs(f_{}_{}-mean_peak_flux)/sd_peak_flux'.format(
            observation['date'], observation['obsnum']))
        dtypes.append('f8')

    t = Table(names=columns, dtype=dtypes)

    trigger_messages_fiducial = []

    trigger_messages_stochastic_new = []
    trigger_messages_stochastic_old = []

    lightcurves = []
    lightcurves_triggered = []
    triggered_sources = []

    n_obs = len(observations)
    n_source = len(yso_cat)
    jds = metadata['JD'].data

    if filter_ == '850':
        noise_faint = 0.014
        noise_bright = 0.02

    elif filter_ == '450':
        noise_faint = metadata['RMS'].data.mean()
        noise_bright = 0.055

    else:
        raise Exception('Filter not recognized')

    # For each source, calculate the mean flux and standard deviation.
    for (i_source, yso) in enumerate(yso_cat):
        id_        = yso['ID']
        index      = yso['Index']
        name       = special_names.get(id_, id_)

        ra         = yso['RA']
        dec        = yso['DEC']
        proto_dist = yso['Proto_Dist']
        disk_dist  = yso['Disk_Dist']

        coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
        ra_str = coord.ra.to_string(sep=':', unit='hour', precision=1)
        dec_str = coord.dec.to_string(sep=':', unit='deg', precision=0)

        fluxes = np.array([map_flux[i_source] for map_flux in map_fluxes])

        triggers = [0.0,] * n_obs

        mean = np.average(fluxes)
        sd = np.std(fluxes, ddof=1)

        sd_fiducial = np.sqrt(noise_faint ** 2.0 + (noise_bright * mean) ** 2.0)

        sd_fiducial_trigger = sd / sd_fiducial

        # The diagonal elements of cov are the variances of the coefficients
        # in z, i.e. np.sqrt(np.diag(cov)) gives you the standard deviations
        # of the coefficients.
        (p, cov) = np.polyfit(
            jds - jds[0],
            fluxes / mean,
            1, cov=True, full=False)

        slope           = p[0] * 36524.0  # Now in %/year
        delta_slope     = np.sqrt(np.diag(cov))[0] * 36524.0
        intercept       = p[1]
        delta_intercept = np.sqrt(np.diag(cov))[1]

        is_triggered = False
        is_triggered_not_old = False

        # Enumerate fluxes backwards so we find most recent trigger first.
        for (i_obs_rev, flux) in enumerate(reversed(fluxes)):
            i_obs = n_obs - i_obs_rev - 1

            jd = jds[i_obs]
            observation = observations[i_obs]

            other_fluxes = np.array([x for (i, x) in enumerate(fluxes) if i != i_obs])

            other_mean = np.average(other_fluxes)
            other_sd = np.std(other_fluxes, ddof=1)

            trigger = abs(flux - other_mean) / other_sd

            triggers[i_obs] = trigger

            # Check to see if any of the peak flux measurements on any date have a large variance

            if ((mean > brightness_thresh)
                    and (trigger >= trigger_thresh)
                    and not is_triggered):
                is_triggered = True

                trigger_message = [
                    '',
                    '####################',
                    '####################',
                    '',
                    'Source {} (index = {}) has abs(flux - flux_m)/SD = {:.2f} on JD: {} = {} (Epoch {}/{})'.format(
                        name, index, trigger, jd, observation['date'], i_obs + 1, n_obs),
                    '',
                    'This is greater than the current abs(flux - flux_m)/SD threshold: {}.'.format(trigger_thresh),
                    'Mean Source Brightness: {:.4f} Jy/beam.'.format(mean),
                    'This source is located at (RA, dec) = ({}, {}) = ({:.5f}, {:.5f})'.format(ra_str, dec_str, ra, dec),
                    'The nearest protostar is {:.2f}" away and the nearest disc is {:.2f}" away.'.format(proto_dist, disk_dist),
                    '',
                    'Peak Brightness      = {:.4f}'.format(flux),
                    'Mean Peak Brightness = {:.4f}'.format(mean),
                    'SD                   = {:.4f}'.format(sd),
                    'SD_fid               = {:.4f}'.format(sd_fiducial),
                    'SD/SD_fid            = {:.5f}'.format(sd_fiducial_trigger),
                    '',
                    '####################',
                    '####################',
                    '',
                ]

                if not i_obs_rev:
                    trigger_messages_stochastic_new.extend(trigger_message)
                    is_triggered_not_old = True

                else:
                    trigger_messages_stochastic_old.extend(trigger_message)


        # Now check for the other indicator of variability - a large relative standard deviation relative to the fiducial model in Doug's Midproject paper (like, for instance, EC53)
        # Also, a birghtness_threshold is coded in just in case we get too many spurious non-detections from faint sources and we want to get rid of those

        if ((mean > brightness_thresh)
                and (sd_fiducial_trigger > sd_thresh)):
            is_triggered = True
            is_triggered_not_old = True

            trigger_messages_fiducial.extend([
                '',
                '####################',
                '####################',
                '',
                'Source {} (index = {}) has an SD = {:.4f} over all peak flux measurements.'.format(name, index, sd),
                '',
                'This is greater than the fiducial SD model would predict for a mean brightness of {:.3f} Jy/beam by a factor of {:.2f}.'.format(mean, sd_fiducial_trigger),
                'The current SD/SD_fiducial threshold is set to {}.'.format(sd_thresh),
                'This source is located at (RA, dec) = ({}, {}) = ({:.5f}, {:.5f})'.format(ra_str, dec_str, ra, dec),
                'The nearest protostar is {:.2f}" away and the nearest disc is {:.2f}" away.'.format(proto_dist, disk_dist),
                '',
                'Peak Brightness      = {:.4f}'.format(fluxes[-1]),
                'Mean Peak Brightness = {:.4f}'.format(mean),
                'SD                   = {:.4f}'.format(sd),
                'SD_fid               = {:.4f}'.format(sd_fiducial),
                'SD/SD_fid            = {:.5f}'.format(sd_fiducial_trigger),
                '',
                '####################',
                '####################',
                '',
            ])

        if is_triggered:
            triggered_sources.append(id_)

        # Add row for this source to the table.
        row = [
            index, id_, ra, dec, proto_dist, disk_dist,
            mean, sd, sd_fiducial, sd_fiducial_trigger,
            slope, delta_slope, abs(slope/delta_slope),
            intercept, delta_intercept,
        ]

        row.extend(fluxes)

        row.extend(triggers)

        t.add_row(row)

        # Create the light curve plot.
        plt.scatter(
            jds,
            fluxes,
            label='{}: {}'.format(index, id_))

        # plt.legend(loc='lower left')
        plt.title('{}: {}'.format(index, id_))

        plt.xticks(rotation=20)

        plt.axhline(y=mean + sd, color='k', linestyle='dashed')
        plt.axhline(y=mean + sd_fiducial, color='b', linestyle='dotted')

        plt.axhline(y=mean, color='k', linestyle='solid')

        plt.axhline(y=mean - sd, color='k', linestyle='dashed')
        plt.axhline(y=mean - sd_fiducial, color='b', linestyle='dotted')

        lightcurve_filename = '{}_{:04d}_{}_lightcurve.pdf'.format(
            lightcurve_prefix, index, re.sub('[^-+_A-Za-z0-9]', '_', id_))

        plt.savefig(lightcurve_filename, format='pdf')
        plt.clf()

        lightcurves.append(lightcurve_filename)

        if is_triggered_not_old:
            lightcurves_triggered.append(lightcurve_filename)

    observation_latest = observations[-1]

    # Build message text.
    text = [
        'Hello Everyone,',
        '',
    ]

    if filter_ == '850':
        text.append('As of {} (JD {:.3f}), the {} region has {} Transient Survey epochs.'.format(
            observation_latest['date'], jds[-1], field_name, n_obs))
    else:
        text.append('As of {} (JD {:.3f}), the {} region has {} Transient Survey epochs which have a sufficiently low RMS at {}um to include in this analysis.'.format(
            observation_latest['date'], jds[-1], field_name, n_obs, filter_))

    text.extend([
        '',
        'The most recent observation had offsets of RA = {:.3f}", Dec = {:.3f}" and a calibration factor of {:.3f} +/- {:.3f}.'.format(
            observation_latest['offset_x'], observation_latest['offset_y'],
            calibration_factors[-1], calibration_factor_errors[-1]),
        '',
        'We are tracking {} sources in this region.'.format(n_source),
        'Here are the latest results from the automatic variability detection pipeline:',
    ])

    if not triggered_sources:
        text.extend([
            '',
            'No potential variables found so far.',
        ])

    else:
        if trigger_messages_stochastic_new:
            text.extend([
                '',
                'NEW INDIVIDUAL EPOCH OUTLIER DETECTIONS:',
                '',
            ])

            text.extend(trigger_messages_stochastic_new)

        if trigger_messages_fiducial:
            text.extend([
                '',
                'TIME SERIES STOCHASTICITY DETECTIONS:',
                '',
            ])

            text.extend(trigger_messages_fiducial)

        if trigger_messages_stochastic_old:
            text.extend([
                '',
                'OLD INDIVIDUAL EPOCH OUTLIER DETECTIONS:',
                '',
            ])

            text.extend(trigger_messages_stochastic_old)

    text.extend([
        '',
        'The sources used for calibration of this region are:',
        '',
    ])

    calibration_ids_by_index = {}
    calibration_ids_unmatched = []

    if calibration_id_mapping is None:
        for calibration_id in sorted(calibration_ids):
            text.append('Catalogue entry {}'.format(
                calibration_id))

    else:
        for calibration_id in calibration_ids:
            for (family_id, cat_index) in calibration_id_mapping:
                if calibration_id == family_id:
                    calibration_ids_by_index[cat_index] = calibration_id
                    break
            else:
                calibration_ids_unmatched.append(calibration_id)

        for cat_index in sorted(calibration_ids_by_index.keys()):
            text.append('Index {} (GaussClumps Catalogue Reference PIDENT {})'.format(
                cat_index, calibration_ids_by_index[cat_index]))

        for calibration_id in calibration_ids_unmatched:
            text.append('Non-matched GaussClumps Catalogue Reference PIDENT {}'.format(
                calibration_id))

    text.extend([
        '',
        'If you see any issues with this message, please contact Steve Mairs at s.mairs@eaobservatory.org.',
        '',
        'Have a great day!',
        'Steve (via the automated variability detection pipeline)',
    ])

    return (t, triggered_sources, text, lightcurves, lightcurves_triggered)
