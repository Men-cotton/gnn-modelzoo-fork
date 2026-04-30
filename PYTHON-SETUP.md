# Software setup

The get more info about software setup and dependencies, please visit our developer doc page [here](https://docs.cerebras.net/en/latest/wsc/getting-started/setup-environment.html).

## Cerebras Wafer-Scale Cluster instructions

After installing all the Cerebras packages distributed in the CSoft platform, to support all the functionalities in Model Zoo, please install other external packages to your python environment.

For a local Python environment, install with uv from the repository root:

```bash
./setup.sh
```

This creates `.venv`, installs the locked dependency set, and installs this
checkout in editable mode. To refresh `uv.lock` after changing dependencies,
run:

```bash
UV_UPDATE_LOCK=1 ./setup.sh
```

**NOTE:** The rest of this guide concerns to *GPU Python environment only*.

Along with the Cerebras Wafer-Scale Cluster, the Model Zoo allows for models to be run on GPUs as well. To run the model code on a GPU, certain packages need to be installed. This is done through the uv-managed environment created by `./setup.sh`.

Follow along below for setting up a GPU environment setup.

## GPU instructions

### CUDA requirements

To run on a GPU, the CUDA libraries must be installed on the system. This includes both the CUDA toolkit as well as the cuDNN libraries. To install these packages, please follow the instructions provided on the [CUDA website](https://developer.nvidia.com/cuda-zone). And make sure to also include the [cuDNN library installation](https://developer.nvidia.com/cudnn).

### PyTorch GPU setup

Currently, this Model Zoo checkout pins PyTorch version `2.4.0`.

Once all the CUDA requirements are installed, create the uv environment with
Python 3.11:

```bash
./setup.sh
```

To test if PyTorch is able to properly access the GPU, start a Python session through the virtual environment created above and run the following commands:

```bash
    $ source .venv/bin/activate
    $ python
    >>> import torch
    >>> torch.__version__
    '2.4.0+cu121' # Confirm that the PT version is `2.4.0`
    >>> torch.cuda.is_available()
    True # Should return `True`
    >>> torch.cuda.device_count()
    1 # Number of devices present
    >>> torch.cuda.get_device_name(0)
    # Should return the proper GPU type
```

**NOTE:** `cerebras_pytorch` is still required by the project metadata. If your
environment uses a private Cerebras package index, configure uv credentials for
that index before running `./setup.sh`.
