"""Jupyter server example application."""

import os

from jupyter_server.extension.application import ExtensionApp, ExtensionAppJinjaMixin

from .handlers import (
    PandaXHandler,
)

DEFAULT_STATIC_FILES_PATH = os.path.join(os.path.dirname(__file__), "static")
DEFAULT_TEMPLATE_FILES_PATH = os.path.join(os.path.dirname(__file__), "templates")


class SimpleApp1(ExtensionAppJinjaMixin, ExtensionApp):
    """A simple jupyter server application."""

    # The name of the extension.
    name = "simple_ext1"

    # The url that your extension will serve its homepage.
    extension_url = "/simple_ext1/default"

    # Should your extension expose other server extensions when launched directly?
    load_other_extensions = True

    # Local path to static files directory.
    static_paths = [DEFAULT_STATIC_FILES_PATH]

    # Local path to templates directory.
    template_paths = [DEFAULT_TEMPLATE_FILES_PATH]

    def initialize_handlers(self):
        """Initialize handlers."""
        self.handlers.extend(
            [
                (rf"/{self.name}/default", PandaXHandler),
            ]
        )

    def initialize_settings(self):
        """Initialize settings."""
        self.log.info(f"Config {self.config}")


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

main = launch_new_instance = SimpleApp1.launch_instance
