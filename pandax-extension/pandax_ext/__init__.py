from .application import SimpleApp1


def _jupyter_server_extension_points():
    return [{"module": "pandax_ext.application", "app": SimpleApp1}]
