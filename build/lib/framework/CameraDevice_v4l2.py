# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ctypes
import errno
import fcntl
import mmap
import os
import queue
import select
import threading
import time

import numpy as np
from numba import njit, prange

from framework.CameraUtils import get_index_from_model_name, set_control
from framework.Logger import logger

# the v4l2 python library is not compatible with modern
# versions of python. provide minimal definitions for
# v4l2. A C-wrapper may need to be provided to ensure
# compatibility across different architectures.
from framework.v4l2_defs import (
    PIXEL_FORMATS,
    V4L2_BUF_TYPE_VIDEO_CAPTURE,
    V4L2_CAP_VIDEO_CAPTURE,
    V4L2_MEMORY_MMAP,
    VIDIOC_DQBUF,
    VIDIOC_G_FMT,
    VIDIOC_G_PARM,
    VIDIOC_QBUF,
    VIDIOC_QUERYBUF,
    VIDIOC_QUERYCAP,
    VIDIOC_REQBUFS,
    VIDIOC_S_FMT,
    VIDIOC_S_PARM,
    VIDIOC_STREAMOFF,
    VIDIOC_STREAMON,
    decode_fourcc,
    v4l2_buffer,
    v4l2_capability,
    v4l2_format,
    v4l2_requestbuffers,
    v4l2_streamparm,
)

"""
NV12 is a YUV 4:2:0 format with:
- full resolution Y (luminance) plane
- UV plane with chroma subsampling (1 UV pair per 2x2 block)

Layout:
yuv_frame[:, :, 0] -> Y
yuv_frame[:, :, 1] -> UV interleaved, subsampled vertically (every 2 rows)
"""


@njit
def convert_nv12_to_rgb(yuv_frame, rgb_image):
    height, width = yuv_frame.shape[:2]
    y = yuv_frame[:, :, 0].astype(np.float32)
    uv = yuv_frame[0:height:2, :, 1].reshape(height // 2, width)

    u_sub = uv[:, 0::2]
    v_sub = uv[:, 1::2]

    u = np.repeat(np.repeat(u_sub, 2, axis=0), 2, axis=1)[:height, :width].astype(np.float32) - 128.0
    v = np.repeat(np.repeat(v_sub, 2, axis=0), 2, axis=1)[:height, :width].astype(np.float32) - 128.0

    y = 1.164 * (y - 16.0)

    r = y + 1.596 * v
    g = y - 0.392 * u - 0.813 * v
    b = y + 2.017 * u

    rgb_image[:, :, 0] = np.clip(r, 0, 255).astype(np.uint8)
    rgb_image[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)
    rgb_image[:, :, 2] = np.clip(b, 0, 255).astype(np.uint8)

    return rgb_image


@njit(parallel=True)
def convert_nv12_to_rgb_parallel(y_plane, uv_plane, rgb_image):
    height, width = y_plane.shape
    u_sub = uv_plane[:, 0::2].astype(np.float32) - 128.0  # (H//2, W//2)
    v_sub = uv_plane[:, 1::2].astype(np.float32) - 128.0
    y_full = 1.164 * (y_plane.astype(np.float32) - 16.0)

    for y in prange(height):
        for x in range(width):
            # corresponding UV index for 2x2 subsampling
            uv_y = y // 2
            uv_x = x // 2

            u = u_sub[uv_y, uv_x]
            v = v_sub[uv_y, uv_x]
            y_val = y_full[y, x]

            r = y_val + 1.596 * v
            g = y_val - 0.392 * u - 0.813 * v
            b = y_val + 2.017 * u

            rgb_image[y, x, 0] = min(255, max(0, int(r)))
            rgb_image[y, x, 1] = min(255, max(0, int(g)))
            rgb_image[y, x, 2] = min(255, max(0, int(b)))

    return rgb_image


"""
YUYV is a YUV 4:2:2 format with:
- full resolution Y (luminance) plane
- UV plane (chrominance) shared across every 2 pixels horizontally [Y0 U0 Y1 V0] [Y2 U1 Y3 V1] ...


Layout:
- each pair of pixels shares one U and one V value: Y0 and Y1 share U0 and V0, ...
"""


@njit
def convert_yuyv_to_rgb(yuv_frame, rgb_image):
    height, width = yuv_frame.shape[:2]
    for y in range(height):
        for x in range(0, width, 2):
            Y0 = float(yuv_frame[y, x, 0])
            U = float(yuv_frame[y, x, 1]) - 128.0
            Y1 = float(yuv_frame[y, x + 1, 0])
            V = float(yuv_frame[y, x + 1, 1]) - 128.0

            Y0 = 1.164 * (Y0 - 16.0)
            Y1 = 1.164 * (Y1 - 16.0)

            R0 = max(0, min(255, Y0 + 1.596 * V))
            G0 = max(0, min(255, Y0 - 0.392 * U - 0.813 * V))
            B0 = max(0, min(255, Y0 + 2.017 * U))

            R1 = max(0, min(255, Y1 + 1.596 * V))
            G1 = max(0, min(255, Y1 - 0.392 * U - 0.813 * V))
            B1 = max(0, min(255, Y1 + 2.017 * U))

            rgb_image[y, x, 0] = np.uint8(R0)
            rgb_image[y, x, 1] = np.uint8(G0)
            rgb_image[y, x, 2] = np.uint8(B0)

            rgb_image[y, x + 1, 0] = np.uint8(R1)
            rgb_image[y, x + 1, 1] = np.uint8(G1)
            rgb_image[y, x + 1, 2] = np.uint8(B1)

    return rgb_image


@njit(parallel=True)
def convert_yuyv_to_rgb_parallel(yuv_frame, rgb_image):
    height, width = yuv_frame.shape[:2]

    for y in prange(height):
        for x in range(0, width, 2):
            Y0 = float(yuv_frame[y, x, 0])
            U = float(yuv_frame[y, x, 1]) - 128.0
            Y1 = float(yuv_frame[y, x + 1, 0])
            V = float(yuv_frame[y, x + 1, 1]) - 128.0

            Y0 = 1.164 * (Y0 - 16.0)
            Y1 = 1.164 * (Y1 - 16.0)

            R0 = Y0 + 1.596 * V
            G0 = Y0 - 0.392 * U - 0.813 * V
            B0 = Y0 + 2.017 * U

            R1 = Y1 + 1.596 * V
            G1 = Y1 - 0.392 * U - 0.813 * V
            B1 = Y1 + 2.017 * U

            rgb_image[y, x, 0] = min(255, max(0, int(R0)))
            rgb_image[y, x, 1] = min(255, max(0, int(G0)))
            rgb_image[y, x, 2] = min(255, max(0, int(B0)))

            rgb_image[y, x + 1, 0] = min(255, max(0, int(R1)))
            rgb_image[y, x + 1, 1] = min(255, max(0, int(G1)))
            rgb_image[y, x + 1, 2] = min(255, max(0, int(B1)))

    return rgb_image


class CameraDevice_v4l2:
    def __init__(self, model_name, width, height, fps, buffer_size, codec, controls, threaded=False):
        self.device_idx = get_index_from_model_name(model_name)
        if self.device_idx < 0:
            raise ValueError(f"Camera with name={model_name} was not found.")

        self.device_path = f"/dev/video{self.device_idx}"
        if not os.path.exists(self.device_path):
            raise ValueError(f"Device {self.device_path} does not exist.")

        logger.debug(f"CameraDevice (v4l2): Initializing device '{model_name}' at {self.device_path}.")

        try:
            self.fd = os.open(self.device_path, os.O_RDWR, 0)
        except OSError as e:
            raise ValueError(f"Failed to open device: {e}")

        if self.fd < 0:
            raise ValueError(f"CameraDevice: Failed to open device {self.device_path}.")

        self.is_streaming = False
        self.buffer_pool = []
        self.buffer_count = 0

        self.last_frame_timestamp = None

        self.capabilities = v4l2_capability()
        fcntl.ioctl(self.fd, VIDIOC_QUERYCAP, self.capabilities)
        if not (self.capabilities.capabilities & V4L2_CAP_VIDEO_CAPTURE):
            raise ValueError(f"CameraDevice: Video capture not supported for {self.device_path}.")

        logger.debug("Applying camera controls")
        for ctl, val in controls.items():
            set_control(self.device_idx, ctl, val)

        # ctrls_dict = get_controls(self.device_idx)
        # logger.debug(ctrls_dict)

        logger.debug("Setting camera parameters")
        self.set_dimensions(width, height)
        self.set_fps(fps)
        self.set_codec(codec)

        self.set_buffersize(buffer_size)

        self.codec = self.get_codec()
        self.width, self.height = self.get_dimensions()
        self.fps = self.get_fps()
        self.buffersize = self.get_buffersize()
        logger.debug(
            f"Camera Parameters: {self.width}x{self.height} @ {self.fps} buffer={self.buffersize} codec={decode_fourcc(self.codec)}"
        )

        if self.fps != fps:
            raise ValueError(f"Camera FPS is {self.fps} not {fps}.")

        if codec == "YUYV" or codec == "NV12":
            self.color_conversion = PIXEL_FORMATS[codec]
        else:
            raise ValueError(f"Unsupported codec '{codec}'. Supported: {list(PIXEL_FORMATS.keys())}")

        # preallocate buffer for rgb conversion
        self.rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        self.frame_count = 0
        self.threaded = threaded
        self.running = False
        self.frame_queue = queue.Queue(maxsize=4)

        # start capture
        try:
            fcntl.ioctl(self.fd, VIDIOC_STREAMON, ctypes.c_uint32(V4L2_BUF_TYPE_VIDEO_CAPTURE))
            self.is_streaming = True
        except OSError as e:
            raise ValueError(f"Error starting video stream: {e}")

        self._start()

    def validate(self):
        for i in range(2):
            if self.get_frame()["frame"] is not None:
                return True
        return False

    def _start(self):
        if self.threaded:
            self.running = True
            self.camera_read_thread = threading.Thread(target=self._read_frames, daemon=True)
            self.camera_read_thread.start()

    def _stop(self):
        if self.threaded:
            self.running = False
            self.camera_read_thread.join()

    def shutdown(self):
        self._stop()

        if not hasattr(self, 'fd') or self.fd is None:
            logger.debug("shutdown: No fd available.")
            return

        # dequeue buffers still in-use
        if self.is_streaming:
            for i in range(self.buffer_count):
                try:
                    r, _, _ = select.select([self.fd], [], [], 1.0)
                    if not r:
                        logger.debug("shutdown: select() timeout, no buffer to dequeue")
                        break
                    buf, _ = self._dequeue_buffer()
                    if buf is None:
                        logger.debug("shutdown: dequeue returned NULL")
                    else:
                        logger.debug(f"shutdown: dequeued buf={buf.index}")
                except Exception as e:
                    logger.warning(f"shutdown: exception during dequeue: {e}")
                    break

            # stop capture before unmapping
            try:
                fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, ctypes.c_uint32(V4L2_BUF_TYPE_VIDEO_CAPTURE))
            except Exception as e:
                logger.warning(f"shutdown: Error setting VIDIOC_STREAMOFF: {e}")
            self.is_streaming = False
        else:
            logger.debug("shutdown: stream not active, skipping dequeue/stream-off")

        # clean up memory-mapped buffers
        self._shutdown_buffers()

        if self.fd is not None:
            try:
                os.close(self.fd)
                self.fd = None
            except Exception as e:
                logger.warning(f"shutdown: Failed to close fd: {e}")
            finally:
                self.fd = None

        logger.info("CameraDevice: shutdown is complete")

    def get_frame(self):
        if self.threaded:
            # NOTE: This could still block if camera processing can't keep up.
            return self.frame_queue.get()
        else:
            return self._read_frame()

    def _read_frames(self):
        if not self.threaded:
            return

        while self.running:
            try:
                frame_data = self._read_frame()

                # if the queue is full, don't wait for the consumer to grab a frame, discard the oldest frame
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame_data)
            except Exception as e:
                logger.warning(f"frame_queue: threw exception {e}.")

    # returns raw camera frame
    def _read_frame(self):
        assert self.fd

        # NOTE: blocking read
        frame, timestamp = self._read_frame_from_device()
        if frame is None:
            logger.warning("CameraDevice:_read_frame: failed to read camera frame.")

        frame_data = {"frame": frame, "frame_number": self.frame_count, "timestamp": timestamp}

        self.frame_count += 1
        return frame_data

    def _convert_frame_to_numpy(self, frame_data):
        if self.color_conversion == PIXEL_FORMATS["YUYV"]:
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((self.height, self.width, 2))
            return frame
        elif self.color_conversion == PIXEL_FORMATS["NV12"]:
            y_size = self.width * self.height
            # y-plane is full resolution
            y_plane = np.frombuffer(frame_data[:y_size], dtype=np.uint8).reshape((self.height, self.width))
            # uv-plane is interleaved, half vertical resolution
            uv_plane = np.frombuffer(frame_data[y_size:], dtype=np.uint8).reshape((self.height // 2, self.width))
            return y_plane, uv_plane
        else:
            logger.warning("Unsupported pixel format for conversion.")
            return None

    # perform colorspace conversion to RGB based on codec
    def convert_to_rgb(self, yuv_frame, copy=False):
        if self.color_conversion == PIXEL_FORMATS["NV12"]:
            y, uv = yuv_frame
            convert_nv12_to_rgb_parallel(y, uv, self.rgb_image)
        elif self.color_conversion == PIXEL_FORMATS["YUYV"]:
            convert_yuyv_to_rgb_parallel(yuv_frame, self.rgb_image)
        else:
            logger.warning("Unsupported pixel format for conversion.")
            return None

        return self.rgb_image.copy() if copy else self.rgb_image

    # extract grayscale (y/luminance) channel from the frame
    def convert_to_grayscale(self, frame):
        if self.color_conversion == PIXEL_FORMATS["NV12"]:
            if isinstance(frame, tuple):
                return frame[0]
            else:
                raise TypeError("Expected NV12 frame as (y_plane, uv_plane)")
        elif self.color_conversion == PIXEL_FORMATS["YUYV"]:
            if isinstance(frame, np.ndarray) and frame.shape[2] >= 1:
                return frame[:, :, 0]
            else:
                raise TypeError("Expected YUYV frame with shape (H, W, 2)")
        else:
            raise ValueError("Unsupported pixel format for grayscale extraction")

    def _read_frame_from_device(self, timeout=1.0):
        # Use select to block until the device signals a new frame is ready to
        # pace with the camera's configured frame rate.
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            logger.warning("_read_frame_from_device: timeout waiting for frame")
            return None, None

        buf, timestamp = self._dequeue_buffer()
        if buf is None:
            return None, None

        """
        if self.last_frame_timestamp is not None:
            delta = timestamp - self.last_frame_timestamp
            expected_interval = 1.0 / self.fps
            if delta < expected_interval * 0.95:
                logger.debug(f"read_frame_from_device: frame dt={delta:.4f}s (expected ~{expected_interval:.4f}s)")
        """

        self.last_frame_timestamp = timestamp
        buf_data = self.buffer_pool[buf.index].buffer
        # make a copy to avoid external references to the mmap
        converted = self._convert_frame_to_numpy(buf_data)
        if isinstance(converted, tuple):
            frame = tuple(arr.copy() for arr in converted)
        else:
            frame = converted.copy()

        self._enqueue_buffer(buf)
        return frame, timestamp

    def _dequeue_buffer(self):
        buf = v4l2_buffer()
        ctypes.memset(ctypes.byref(buf), 0, ctypes.sizeof(buf))
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP

        try:
            fcntl.ioctl(self.fd, VIDIOC_DQBUF, buf)
            # logger.debug(f"_dequeue_buffer: dequeued buffer {buf.index}")
        except OSError as e:
            if e.errno == errno.EAGAIN:
                logger.debug("_dequeue_buffer: No buffer available to dequeue")
                return None, None
            elif e.errno == errno.ENODEV:
                logger.debug("_dequeue_buffer: Device no longer available.")
                return None, None
            else:
                logger.error(f"_dequeue_buffer: Error dequeuing buffer: {e}")
                return None, None

        timestamp = buf.timestamp.secs + buf.timestamp.usecs / 1e6
        return buf, timestamp

    def _enqueue_buffer(self, buf):
        try:
            fcntl.ioctl(self.fd, VIDIOC_QBUF, buf)
        except OSError as e:
            logger.error(f"_enqueue_buffer: Error enqueuing buffer: {e}")
            return None

    # to determine the resolutions + pixel_formats which achieve the target FPS:
    # 'v4l2-ctl -d device_idx --list-formats-ext
    def set_codec(self, codec_str):
        fmt = v4l2_format()
        ctypes.memset(ctypes.byref(fmt), 0, ctypes.sizeof(fmt))
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        try:
            fcntl.ioctl(self.fd, VIDIOC_G_FMT, fmt)
        except Exception as e:
            logger.error(f"set_codec: Failed to get codec: {e}")
            raise

        fmt.fmt.pix.pixelformat = PIXEL_FORMATS.get(codec_str)

        if fmt.fmt.pix.pixelformat is None:
            raise ValueError(f"Unsupported codec string: {codec_str}")
        try:
            fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)
        except Exception as e:
            logger.error(f"set_codec: Failed to set codec: {e}")
            raise

    def get_codec(self):
        fmt = v4l2_format()
        ctypes.memset(ctypes.byref(fmt), 0, ctypes.sizeof(fmt))
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fcntl.ioctl(self.fd, VIDIOC_G_FMT, fmt)
        return fmt.fmt.pix.pixelformat

    def set_dimensions(self, width, height):
        fmt = v4l2_format()
        ctypes.memset(ctypes.byref(fmt), 0, ctypes.sizeof(fmt))
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = width
        fmt.fmt.pix.height = height
        try:
            fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)
        except Exception as e:
            logger.error(f"set_dimensions: Failed to set codec: {e}")
            raise

    def get_dimensions(self):
        fmt = v4l2_format()
        ctypes.memset(ctypes.byref(fmt), 0, ctypes.sizeof(fmt))
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fcntl.ioctl(self.fd, VIDIOC_G_FMT, fmt)
        return fmt.fmt.pix.width, fmt.fmt.pix.height

    def set_fps(self, fps):
        streamparm = v4l2_streamparm()
        ctypes.memset(ctypes.byref(streamparm), 0, ctypes.sizeof(streamparm))
        streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        streamparm.parm.capture.timeperframe.numerator = 1
        streamparm.parm.capture.timeperframe.denominator = fps
        try:
            fcntl.ioctl(self.fd, VIDIOC_S_PARM, streamparm)
        except Exception as e:
            logger.error(f"set_fps: Failed to set codec: {e}")
            raise

    def get_fps(self):
        streamparm = v4l2_streamparm()
        ctypes.memset(ctypes.byref(streamparm), 0, ctypes.sizeof(streamparm))
        streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fcntl.ioctl(self.fd, VIDIOC_G_PARM, streamparm)
        return streamparm.parm.capture.timeperframe.denominator / streamparm.parm.capture.timeperframe.numerator

    def set_buffersize(self, size):
        # currently does not support mid-stream buffer reconfiguration
        assert not self.is_streaming
        # if self.is_streaming:
        # dequeue buffers in-use
        # stop streaming
        # shutdown buffers
        # reconfigure buffers
        # re-start streaming
        self._shutdown_buffers()
        self._request_buffers(size)
        self._init_mmap_buffers()

    def get_buffersize(self):
        return len(self.buffer_pool)

    def _request_buffers(self, count):
        assert self.buffer_count == 0

        reqbufs = v4l2_requestbuffers()
        ctypes.memset(ctypes.byref(reqbufs), 0, ctypes.sizeof(reqbufs))
        reqbufs.count = count
        reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        reqbufs.memory = V4L2_MEMORY_MMAP

        try:
            fcntl.ioctl(self.fd, VIDIOC_REQBUFS, reqbufs)
        except Exception as e:
            logger.error(f"request_buffers: Failed to request buffers: {e}")
            raise

        if reqbufs.count < count:
            logger.warning("request_buffers: Returned fewer buffers than requested")

        self.buffer_count = reqbufs.count
        assert self.buffer_count != 0

    def _init_mmap_buffers(self):
        for i in range(self.buffer_count):
            buf = v4l2_buffer()
            ctypes.memset(ctypes.byref(buf), 0, ctypes.sizeof(buf))
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            buf.index = i

            try:
                fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, buf)
            except Exception as e:
                logger.error(f"init_mmap_buffers: Failed to query buffer {i}: {e}")
                continue

            buf.buffer = mmap.mmap(
                self.fd, buf.length, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=buf.m.offset
            )

            # NOTE: if we want to switch to DMA buffer export:
            # expbuf = v4l2.v4l2_exportbuffer()
            # expbuf.type = buf.type
            # expbuf.index = buf.index
            # expbuf.flags = os.O_CLOEXEC | os.O_RDWR
            # fcntl.ioctl(self.fd, v4l2.VIDIOC_EXPBUF, expbuf)
            # buf.fd = expbuf.fd

            self.buffer_pool.append(buf)
            self._enqueue_buffer(buf)

    def _shutdown_buffers(self):
        # unmap
        for buf in self.buffer_pool:
            try:
                if hasattr(buf, 'buffer') and buf.buffer is not None:
                    buf.buffer.close()  # this unmaps mmap'd memory
                    # logger.debug(f"shutdown_buffers: unmapped buffer {buf.index}")
            except Exception as e:
                logger.warning(f"shutdown_buffers: Failed to close buffer {buf.index}: {e}")

        self.buffer_pool.clear()
        self.buffer_count = 0

        # when specifying REQBUFS with count 0, it instructs the kernel
        # to release the memory
        req = v4l2_requestbuffers()
        ctypes.memset(ctypes.byref(req), 0, ctypes.sizeof(req))
        req.count = 0
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = V4L2_MEMORY_MMAP
        try:
            fcntl.ioctl(self.fd, VIDIOC_REQBUFS, req)
            # logger.debug("shutdown_buffers: kernel buffers released")
        except Exception as e:
            logger.warning(f"shutdown_buffers: Failed to release kernel buffers: {e}")
