FROM nvcr.io/nvidia/pytorch:25.02-py3

# https://github.com/openucx/ucc/issues/476 - 'ImportError: /opt/hpcx/ucx/lib/libucs.so.0: undefined symbol: ucm_set_global_opts'
# Workaround: Error happens because compiler picks up libucm required by libucs from a different directory,
# i.e. libucs is taken from $UCX_HOME while libucm comes from HPCX installed in /opt. Proper container environment
# config should resolve the issue.
ENV LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH"

# allow more efficient management of memory segments
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /workspaces


# NOTE: Added libgl1 for cv2 'ImportError: libGL.so.1: cannot open shared object file: No such file or directory'
# libxkbfile1 is neeed for nsys-ui
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends --assume-yes \
      build-essential make gcc g++ gdb strace valgrind git clang-format \
      xauth libgl1 ffmpeg v4l-utils udev usbutils libusb-1.0-0-dev wget \
      x11-utils x11-xserver-utils x11-apps \
      python3-opencv libxkbfile1 nvtop tlp

# mcc daq https://github.com/mccdaq/uldaq?_ga=2.85905500.479671302.1736441555-1860231292.1736441555
# Required dependency for python library 'uldaq'
RUN mkdir mccdaq
RUN wget -N https://github.com/mccdaq/uldaq/releases/download/v1.2.1/libuldaq-1.2.1.tar.bz2 --directory-prefix=mccdaq/
RUN tar -xvjf mccdaq/libuldaq-1.2.1.tar.bz2 -C mccdaq/
RUN cd mccdaq/libuldaq-1.2.1 && ./configure && make && make install && cd ../..

RUN python -m pip install --upgrade pip

# ngc.nvidia pulls in opencv=4.7.0 which is not compatible with running graphical
# user interfaces within a docker container. An appropriate version of opencv is
# installed in 'requirements.txt'. However, if the container library is not explicitly removed
# imshow will fail with 'error: (-2:Unspecified error) The function is not implemented. Rebuild the library with ...'
RUN pip uninstall --yes opencv

COPY requirements.txt /workspaces/requirements.txt
RUN pip install -r requirements.txt

#----------------------
# profiling tools
# https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html
#----------------------

ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_2/
# cli-only
#ARG NSYS_PKG=NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb
# cli + ui
ARG NSYS_PKG=nsight-systems-2025.2.1_2025.2.1.130-1_amd64.deb
#RUN apt-get update && apt install -y libglib2.0-0 libxkbfile1
#RUN wget ${NSYS_URL}${NSYS_PKG} && dpkg -i $NSYS_PKG && rm $NSYS_PKG
RUN wget ${NSYS_URL}${NSYS_PKG} && apt install -y ./$NSYS_PKG && rm $NSYS_PKG
