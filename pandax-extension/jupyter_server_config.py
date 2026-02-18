"""Configuration file for jupyter-server extensions."""

c = get_config()
# ------------------------------------------------------------------------------
# Application(SingletonConfigurable) configuration
# ------------------------------------------------------------------------------
# The date format used by logging formatters for %(asctime)s
c.Application.log_datefmt = (  # type:ignore[name-defined]
    "%Y-%m-%d %H:%M:%S Simple_Extensions_Example"
)

# ------------------------------------------------------------------------------
# Jupyter Server Configuration
# ------------------------------------------------------------------------------

# Set the root directory for notebooks and terminals.
# All file paths accessed via the Jupyter server will be relative to this directory.
c.ServerApp.notebook_dir = "/Users/yinganwang/Development/capstone/pandax-meng"

# Allow cross-origin requests from your frontend app running on localhost:3000.
# Necessary if you’re running a Next.js or React frontend separately.
c.ServerApp.allow_origin = "http://localhost:3000"

# Allow credentials (cookies, auth headers) to be sent with cross-origin requests.
c.ServerApp.allow_credentials = True

# Disable XSRF/CSRF protection.
# Useful in local development to simplify API calls, but not recommended in production.
c.ServerApp.disable_check_xsrf = True

# Enable terminal support in the Jupyter server.
# Required if you want to use the <Terminal> component from jupyter-react.
c.ServerApp.terminals_enabled = True

# Pass additional Tornado settings to the server.
# 'debug=True' enables Tornado debug mode (auto-reload, more verbose logs).
c.ServerApp.tornado_settings = {"debug": True}

# Set a fixed token for authentication when connecting to this Jupyter server.
# The frontend (React/Next.js) must use this token to authenticate API calls.
c.ServerApp.token = "pandax-local-dev"

# Set the logging level of the Jupyter server.
# 'ERROR' suppresses warnings and info messages, showing only errors.
c.ServerApp.log_level = "ERROR"
