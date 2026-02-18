# Jupyter Server Simple Extension Example

This folder contains example of simple extensions on top of Jupyter Server and review configuration aspects.

## Install

```bash
pip install -e '.[test]'

jupyter server extension list
```

If you see simple_ext1 with status OK, then it means you have succesfully installed the extension.

```bash
Config dir: /Users/yinganwang/.jupyter

Config dir: /opt/anaconda3/envs/pandax/etc/jupyter
    jupyter_server_terminals enabled
    - Validating jupyter_server_terminals...
      jupyter_server_terminals 0.5.3 OK
    pandax_ext enabled
    - Validating pandax_ext...
      pandax_ext  OK

Config dir: /usr/local/etc/jupyter
```

## Run jupyter server extension

```bash
# In root directory
chmod +x pandax-extension/run_pandax_extension_server.sh
./pandax-extension/run_pandax_jupyter.sh
```

### How the script works
In terminal, run
```bash
jupyter server --ServerApp.disable_check_xsrf=True --ServerApp.tornado_settings="{'debug': True}"
```

And you would see the port and the token in your terminal tab:

```bash
To access the server, open this file in a browser:
  file:///Users/yinganwang/Library/Jupyter/runtime/jpserver-1881-open.html
Or copy and paste one of these URLs:
  http://localhost:8888/?token=c18e55de2f1e0cc81f3170ffadd1f525d3ad0fde4e9bbdb8
  http://127.0.0.1:8888/?token=c18e55de2f1e0cc81f3170ffadd1f525d3ad0fde4e9bbdb8
```

Now you can use CURL to access the `GET` and `POST` endpoints.

```bash
curl -G "http://localhost:8888/simple_ext1/default" \
  --data-urlencode "path=/Users/yinganwang/Development/capstone/pandax-meng/notebooks/spscientist/student-performance-in-exams/src/small_bench_meng.ipynb" \
  -H "X-XSRFToken: <your_token>"

curl -X POST "http://localhost:8888/simple_ext1/default" \
  -H "Content-Type: application/json" \
  -H "X-XSRFToken: <your_token>" \
  -d '{"path": "/Users/yinganwang/Development/capstone/pandax-meng/notebooks/spscientist/student-performance-in-exams/src/small_bench_meng.ipynb"}'
```

Reference: https://github.com/jupyter-server/jupyter_server/tree/main/examples/simple
