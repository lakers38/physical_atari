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

# Excerpts from v4l2.py and https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/videodev.html

# ===================
# ioctl utils
# ===================

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2

_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS


def _IOC(dir_, type_, nr, size):
    return (
        ctypes.c_int32(dir_ << _IOC_DIRSHIFT).value
        | ctypes.c_int32(ord(type_) << _IOC_TYPESHIFT).value
        | ctypes.c_int32(nr << _IOC_NRSHIFT).value
        | ctypes.c_int32(size << _IOC_SIZESHIFT).value
    )


def _IOC_TYPECHECK(t):
    return ctypes.sizeof(t)


def _IO(type_, nr):
    return _IOC(_IOC_NONE, type_, nr, 0)


def _IOW(type_, nr, size):
    return _IOC(_IOC_WRITE, type_, nr, _IOC_TYPECHECK(size))


def _IOR(type_, nr, size):
    return _IOC(_IOC_READ, type_, nr, _IOC_TYPECHECK(size))


def _IOWR(type_, nr, size):
    return _IOC(_IOC_READ | _IOC_WRITE, type_, nr, _IOC_TYPECHECK(size))


# ===================
# v4l2 constants
# ===================

V4L2_MEMORY_MMAP = 1

V4L2_CAP_VIDEO_CAPTURE = 0x00000001


def fourcc(a, b, c, d):
    return (ord(a)) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24)


def decode_fourcc(fmt):
    return ''.join([chr((fmt >> 8 * i) & 0xFF) for i in range(4)])


PIXEL_FORMATS = {
    "YUYV": fourcc('Y', 'U', 'Y', 'V'),
    "NV12": fourcc('N', 'V', '1', '2'),
}

# ===================
# v4l2 structs
#
# NOTE: These haven't been tested for ABI compatibiity.
# ===================

# https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/vidioc-querycap.html#c.v4l2_capability


class v4l2_capability(ctypes.Structure):
    _fields_ = [
        ('driver', ctypes.c_char * 16),
        ('card', ctypes.c_char * 32),
        ('bus_info', ctypes.c_char * 32),
        ('version', ctypes.c_uint32),
        ('capabilities', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32 * 4),
    ]


# https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/pixfmt-002.html#c.v4l2_pix_format

v4l2_field = ctypes.c_uint
(
    V4L2_FIELD_ANY,
    V4L2_FIELD_NONE,
    V4L2_FIELD_TOP,
    V4L2_FIELD_BOTTOM,
    V4L2_FIELD_INTERLACED,
    V4L2_FIELD_SEQ_TB,
    V4L2_FIELD_SEQ_BT,
    V4L2_FIELD_ALTERNATE,
    V4L2_FIELD_INTERLACED_TB,
    V4L2_FIELD_INTERLACED_BT,
) = range(10)

v4l2_colorspace = ctypes.c_uint
(
    V4L2_COLORSPACE_SMPTE170M,
    V4L2_COLORSPACE_SMPTE240M,
    V4L2_COLORSPACE_REC709,
    V4L2_COLORSPACE_BT878,
    V4L2_COLORSPACE_470_SYSTEM_M,
    V4L2_COLORSPACE_470_SYSTEM_BG,
    V4L2_COLORSPACE_JPEG,
    V4L2_COLORSPACE_SRGB,
) = range(1, 9)


class v4l2_pix_format(ctypes.Structure):
    _fields_ = [
        ('width', ctypes.c_uint32),
        ('height', ctypes.c_uint32),
        ('pixelformat', ctypes.c_uint32),
        ('field', v4l2_field),
        ('bytesperline', ctypes.c_uint32),
        ('sizeimage', ctypes.c_uint32),
        ('colorspace', v4l2_colorspace),
        ('flags', ctypes.c_uint32),
        ('priv', ctypes.c_uint32),
    ]


# https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/vidioc-g-fmt.html#c.v4l2_format

v4l2_buf_type = ctypes.c_uint
(
    V4L2_BUF_TYPE_VIDEO_CAPTURE,
    V4L2_BUF_TYPE_VIDEO_OUTPUT,
    V4L2_BUF_TYPE_VIDEO_OVERLAY,
    V4L2_BUF_TYPE_VBI_CAPTURE,
    V4L2_BUF_TYPE_VBI_OUTPUT,
    V4L2_BUF_TYPE_SLICED_VBI_CAPTURE,
    V4L2_BUF_TYPE_SLICED_VBI_OUTPUT,
    V4L2_BUF_TYPE_VIDEO_OUTPUT_OVERLAY,
    V4L2_BUF_TYPE_PRIVATE,
) = list(range(1, 9)) + [0x80]


class v4l2_rect(ctypes.Structure):
    _fields_ = [
        ('left', ctypes.c_int32),
        ('top', ctypes.c_int32),
        ('width', ctypes.c_int32),
        ('height', ctypes.c_int32),
    ]


class v4l2_clip(ctypes.Structure):
    pass


v4l2_clip._fields_ = [
    ('c', v4l2_rect),
    ('next', ctypes.POINTER(v4l2_clip)),
]


class v4l2_window(ctypes.Structure):
    _fields_ = [
        ('w', v4l2_rect),
        ('field', v4l2_field),
        ('chromakey', ctypes.c_uint32),
        ('clips', ctypes.POINTER(v4l2_clip)),
        ('clipcount', ctypes.c_uint32),
        ('bitmap', ctypes.c_void_p),
        ('global_alpha', ctypes.c_uint8),
    ]


class v4l2_vbi_format(ctypes.Structure):
    _fields_ = [
        ('sampling_rate', ctypes.c_uint32),
        ('offset', ctypes.c_uint32),
        ('samples_per_line', ctypes.c_uint32),
        ('sample_format', ctypes.c_uint32),
        ('start', ctypes.c_int32 * 2),
        ('count', ctypes.c_uint32 * 2),
        ('flags', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32 * 2),
    ]


class v4l2_sliced_vbi_format(ctypes.Structure):
    _fields_ = [
        ('service_set', ctypes.c_uint16),
        ('service_lines', ctypes.c_uint16 * 2 * 24),
        ('io_size', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32 * 2),
    ]


class v4l2_format(ctypes.Structure):
    class _u(ctypes.Union):
        _fields_ = [
            ('pix', v4l2_pix_format),
            ('win', v4l2_window),
            ('vbi', v4l2_vbi_format),
            ('sliced', v4l2_sliced_vbi_format),
            ('raw_data', ctypes.c_char * 200),
        ]

    _fields_ = [
        ('type', v4l2_buf_type),
        ('fmt', _u),
    ]


# https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/vidioc-reqbufs.html#c.v4l2_requestbuffers

v4l2_memory = ctypes.c_uint
(
    V4L2_MEMORY_MMAP,
    V4L2_MEMORY_USERPTR,
    V4L2_MEMORY_OVERLAY,
) = range(1, 4)


class v4l2_requestbuffers(ctypes.Structure):
    _fields_ = [
        ('count', ctypes.c_uint32),
        ('type', v4l2_buf_type),
        ('memory', v4l2_memory),
        ('reserved', ctypes.c_uint32 * 2),
    ]
    # NOTE: For V4L2_MEMORY_DMABUF, we'd also need:
    #       - v4l2_exportbuffer for sharing buffers via fd
    #       - v4l2_plane structures for multi-plane formats


# https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/buffer.html#c.v4l2_buffer


class timeval(ctypes.Structure):
    _fields_ = [
        ('secs', ctypes.c_long),
        ('usecs', ctypes.c_long),
    ]


class v4l2_timecode(ctypes.Structure):
    _fields_ = [
        ('type', ctypes.c_uint32),
        ('flags', ctypes.c_uint32),
        ('frames', ctypes.c_uint8),
        ('seconds', ctypes.c_uint8),
        ('minutes', ctypes.c_uint8),
        ('hours', ctypes.c_uint8),
        ('userbits', ctypes.c_uint8 * 4),
    ]


class v4l2_buffer(ctypes.Structure):
    class _u(ctypes.Union):
        _fields_ = [
            ('offset', ctypes.c_uint32),  # for V4L2_MEMORY_MMAP
            ('userptr', ctypes.c_ulong),  # for V4L2_MEMORY_USERPTR
            # NOTE: For V4L2_MEMORY_DMABUF, an explicit fd field is required.
            #       If needed, expand this union to include: ('fd', ctypes.c_int)
        ]

    _fields_ = [
        ('index', ctypes.c_uint32),
        ('type', v4l2_buf_type),
        ('bytesused', ctypes.c_uint32),
        ('flags', ctypes.c_uint32),
        ('field', v4l2_field),
        ('timestamp', timeval),
        ('timecode', v4l2_timecode),
        ('sequence', ctypes.c_uint32),
        ('memory', v4l2_memory),
        ('m', _u),
        ('length', ctypes.c_uint32),
        ('input', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32),
    ]


# https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/vidioc-g-parm.html#c.v4l2_captureparm


class v4l2_fract(ctypes.Structure):
    _fields_ = [
        ('numerator', ctypes.c_uint32),
        ('denominator', ctypes.c_uint32),
    ]


class v4l2_captureparm(ctypes.Structure):
    _fields_ = [
        ('capability', ctypes.c_uint32),
        ('capturemode', ctypes.c_uint32),
        ('timeperframe', v4l2_fract),
        ('extendedmode', ctypes.c_uint32),
        ('readbuffers', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32 * 4),
    ]


# https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/vidioc-g-parm.html#c.v4l2_streamparm


class v4l2_outputparm(ctypes.Structure):
    _fields_ = [
        ('capability', ctypes.c_uint32),
        ('outputmode', ctypes.c_uint32),
        ('timeperframe', v4l2_fract),
        ('extendedmode', ctypes.c_uint32),
        ('writebuffers', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32 * 4),
    ]


class v4l2_streamparm(ctypes.Structure):
    class _u(ctypes.Union):
        _fields_ = [
            ('capture', v4l2_captureparm),
            ('output', v4l2_outputparm),
            ('raw_data', ctypes.c_char * 200),
        ]

    _fields_ = [('type', v4l2_buf_type), ('parm', _u)]


# ===================
# ioctl commands
# ===================

VIDIOC_QUERYCAP = _IOR('V', 0, v4l2_capability)
VIDIOC_G_FMT = _IOWR('V', 4, v4l2_format)
VIDIOC_S_FMT = _IOWR('V', 5, v4l2_format)
VIDIOC_REQBUFS = _IOWR('V', 8, v4l2_requestbuffers)
VIDIOC_QUERYBUF = _IOWR('V', 9, v4l2_buffer)
VIDIOC_QBUF = _IOWR('V', 15, v4l2_buffer)
VIDIOC_DQBUF = _IOWR('V', 17, v4l2_buffer)
VIDIOC_STREAMON = _IOW('V', 18, ctypes.c_int)
VIDIOC_STREAMOFF = _IOW('V', 19, ctypes.c_int)
VIDIOC_G_PARM = _IOWR('V', 21, v4l2_streamparm)
VIDIOC_S_PARM = _IOWR('V', 22, v4l2_streamparm)

# ===================
# export list
# ===================

__all__ = [
    'v4l2_capability',
    'v4l2_format',
    'v4l2_pix_format',
    'v4l2_buffer',
    'v4l2_requestbuffers',
    'v4l2_streamparm',
    'v4l2_captureparm',
    'v4l2_fract',
    'PIXEL_FORMATS',
    'VIDIOC_QUERYCAP',
    'VIDIOC_G_FMT',
    'VIDIOC_S_FMT',
    'VIDIOC_REQBUFS',
    'VIDIOC_QUERYBUF',
    'VIDIOC_QBUF',
    'VIDIOC_DQBUF',
    'VIDIOC_STREAMON',
    'VIDIOC_STREAMOFF',
    'VIDIOC_G_PARM',
    'VIDIOC_S_PARM',
    'fourcc',
    'decode_fourcc',
]
