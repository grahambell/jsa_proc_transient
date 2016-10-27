import logging

logger = logging.getLogger(__name__)

gbs_objects = {
    'IC348': [
        ('MJLSG38', 'IC348-E'),
        ('MJLSG38', 'IC348-C'),
    ],

    'NGC1333': [
        ('MJLSG38', 'NGC1333-N'),
        ('MJLSG38', 'NGC1333-S'),
        ('MJLSG22', 'NGC1333'),
    ],

    'OPH_CORE': [
        ('MJLSG32', 'L1688-1'),
        ('MJLSG32', 'L1688-2'),
        ('MJLSG32', 'L1688-3'),
        ('MJLSG32', 'L1688-4'),
    ],

    'Serpens_Main': [
        ('MJLSG33', 'SerpensMain1'),
    ],

    'Serpens_South': [
        ('MJLSG33', 'SerpensS-NE'),
        ('MJLSG33', 'SerpensS-NW'),
        ('MJLSG33', 'SerpensS-SE'),
        ('MJLSG33', 'SerpensS-SW'),
    ],

    'NGC2071': [
        ('MJLSG41', 'OrionBN_450_E'),
        ('MJLSG41', 'OrionBN_450_S'),
        ('MJLSG41', 'OrionBN_450_W'),
    ],

    'NGC2024': [
        ('MJLSG41', 'OrionBS_450_E'),
        ('MJLSG41', 'OrionBS_450_S'),
        ('MJLSG41', 'OrionBS_450_W'),
        ('MJLSG41', 'OrionBS_850_S'),
    ],

    'OMC23': [
        ('MJLSG31', 'OMC1 tile1'),
        ('MJLSG31', 'OMC1 tile2'),
        ('MJLSG31', 'OMC1 tile3'),
        ('MJLSG31', 'OMC1 tile4'),
        ('MJLSG22', 'OMC1 tile1'),
        ('MJLSG22', 'OMC1 tile2'),
    ],
}

def get_field_name(source_name):
    for (field, sources) in gbs_objects.items():
        for (project, field_source_name) in sources:
            if source_name == field_source_name:
                return field

    raise Exception('Unexpected source name "{}"'.format(source_name))
