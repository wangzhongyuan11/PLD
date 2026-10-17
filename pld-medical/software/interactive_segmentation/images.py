"""Bounded in-memory image decoding; no client-selected filesystem reads."""
import base64
import binascii
import gzip
import io
import warnings
import numpy as np
from PIL import Image,ImageOps
from .contracts import InputError

MAX_FILE_BYTES = 16*1024*1024
MAX_PIXELS = 16_000_000


def decode_image(model, data, filename='image.png'):
    if not isinstance(data,str) or len(data)>MAX_FILE_BYTES*4//3+2048:
        raise InputError('图像为空或超过16 MB')
    encoded = data.split(',',1)[1] if data.startswith('data:') and ',' in data else data
    try:
        raw = base64.b64decode(encoded,validate=True)
    except (ValueError,binascii.Error):
        raise InputError('图像 base64 格式无效') from None
    if not raw or len(raw)>MAX_FILE_BYTES:
        raise InputError('图像为空或超过16 MB')
    filename = str(filename)
    if filename.lower().endswith(('.nii','.nii.gz')):
        if model!='camus':
            raise InputError('NIfTI 单帧只支持 CAMUS 接口')
        try:
            if raw.startswith(b'\x1f\x8b'):
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
                    raw = f.read(64*1024*1024+1)
                if len(raw)>64*1024*1024:
                    raise InputError('解压后影像过大')
            import nibabel as nib
            handle = nib.Nifti1Image.from_bytes(raw)
            if np.prod(handle.shape)>MAX_PIXELS:
                raise InputError('NIfTI 超过1600万像素')
            array = np.squeeze(np.asarray(handle.dataobj))
            if array.ndim!=2:
                raise InputError('请选择单个 ED/ES 二维影像；暂不接受多帧序列')
            if not np.isfinite(array).all() or array.min()<0 or array.max()>255:
                raise InputError('CAMUS 输入应为0..255的有限灰度值')
            # Same X,Y -> displayed H,W convention as the frozen CAMUS loader.
            image = Image.fromarray(np.rint(array.T).astype(np.uint8)).convert('RGB')
        except InputError:
            raise
        except Exception:
            raise InputError('无法读取 NIfTI，请使用单帧 CAMUS .nii 或 .nii.gz') from None
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as source:
                    if source.width*source.height>MAX_PIXELS:
                        raise InputError('图像超过1600万像素')
                    if source.format not in ('PNG','JPEG','BMP','WEBP','TIFF'):
                        raise InputError('请使用 PNG/JPG/BMP/WebP/TIFF 图像')
                    image = ImageOps.exif_transpose(source).convert('RGB')
        except InputError:
            raise
        except Exception:
            raise InputError('无法解码图像') from None
    if image.width<2 or image.height<2 or image.width*image.height>MAX_PIXELS:
        raise InputError('图像尺寸应至少2×2且不超过1600万像素')
    return np.asarray(image).copy()

