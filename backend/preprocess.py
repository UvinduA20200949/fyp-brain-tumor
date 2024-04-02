import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from typing import Optional, Union, List, Tuple
import numpy as np
from typing import Iterable, Union
import pywt
import io
import numpy as np
from PIL import Image
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

def display_image(image_array):
    """
    Display an image from a NumPy array using Matplotlib.

    Parameters:
        image_array (numpy.ndarray): The image array to be displayed.
    """
    # Reshape the array to remove the singleton dimension if present
    image_array = np.squeeze(image_array)

    # Display the image using Matplotlib
    plt.imshow(image_array, cmap='gray')  # Assuming the image is grayscale
    plt.axis('off')  # Turn off axis
    plt.show()


def image_to_binary(image_path):
    """
    Convert an image to a binary object.
    
    Parameters:
    - image_path (str): The file system path to the image.
    
    Returns:
    - bytes: The binary representation of the image.
    """
    with open(image_path, 'rb') as image_file:
        binary_data = image_file.read()
    return binary_data


def load_and_convert_image_from_binary(image_binary, target_height=256, target_width=256):
    """
    Load an image from a binary file and convert it into a TensorFlow tensor.

    Args:
    - image_binary: Binary content of the image.

    Returns:
    - image: A TensorFlow tensor representing the loaded image with values in the range [0, 1].
    """

    image = tf.image.decode_image(image_binary, channels=1)

    # Resize the image to the target dimensions
    resized_image = tf.image.resize(image, [target_height, target_width])

    return resized_image


Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable
]

Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable
]

def apply_median_filter(image):
    """
    Apply a median filter to an image.

    Args:
    - image: A tensor representing an image.

    Returns:
    - filtered_image: The image after applying the median filter.
    """
    filtered_image = median_filter2d(image, filter_shape=(3, 3))

    return filtered_image

def get_ndims(image):
    return image.get_shape().ndims or tf.rank(image)


def to_4D_image(image):
    """Convert 2/3/4D image to 4D image.

    Args:
      image: 2/3/4D `Tensor`.

    Returns:
      4D `Tensor` with the same type.
    """
    with tf.control_dependencies(
        [
            tf.debugging.assert_rank_in(
                image, [2, 3, 4], message="`image` must be 2/3/4D tensor"
            )
        ]
    ):
        ndims = image.get_shape().ndims
        if ndims is None:
            return _dynamic_to_4D_image(image)
        elif ndims == 2:
            return image[None, :, :, None]
        elif ndims == 3:
            return image[None, :, :, :]
        else:
            return image

def _dynamic_to_4D_image(image):
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    # 4D image => [N, H, W, C] or [N, C, H, W]
    # 3D image => [1, H, W, C] or [1, C, H, W]
    # 2D image => [1, H, W, 1]
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.reshape(image, new_shape)

def _dynamic_from_4D_image(image, original_rank):
    shape = tf.shape(image)
    # 4D image <= [N, H, W, C] or [N, C, H, W]
    # 3D image <= [1, H, W, C] or [1, C, H, W]
    # 2D image <= [1, H, W, 1]
    begin = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)

def from_4D_image(image, ndims):
    """Convert back to an image with `ndims` rank.

    Args:
      image: 4D `Tensor`.
      ndims: The original rank of the image.

    Returns:
      `ndims`-D `Tensor` with the same type.
    """
    with tf.control_dependencies(
        [tf.debugging.assert_rank(image, 4, message="`image` must be 4D tensor")]
    ):
        if isinstance(ndims, tf.Tensor):
            return _dynamic_from_4D_image(image, ndims)
        elif ndims == 2:
            return tf.squeeze(image, [0, 3])
        elif ndims == 3:
            return tf.squeeze(image, [0])
        else:
            return image

@tf.function
def median_filter2d(
    image: TensorLike,
    filter_shape: Union[int, Iterable[int]] = (3, 3),
    padding: str = "REFLECT",
    constant_values: TensorLike = 0,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Perform median filtering on image(s).

    Args:
      image: Either a 2-D `Tensor` of shape `[height, width]`,
        a 3-D `Tensor` of shape `[height, width, channels]`,
        or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: An `integer` or `tuple`/`list` of 2 integers, specifying
        the height and width of the 2-D median filter. Can be a single integer
        to specify the same value for all spatial dimensions.
      padding: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
      name: A name for this operation (optional).
    Returns:
      2-D, 3-D or 4-D `Tensor` of the same dtype as input.
    Raises:
      ValueError: If `image` is not 2, 3 or 4-dimensional,
        if `padding` is other than "REFLECT", "CONSTANT" or "SYMMETRIC",
        or if `filter_shape` is invalid.
    """
    with tf.name_scope(name or "median_filter2d"):
        image = tf.convert_to_tensor(image, name="image")
        original_ndims = get_ndims(image)
        image = to_4D_image(image)

        filter_shape = normalize_tuple(filter_shape, 2, "filter_shape")

        image_shape = tf.shape(image)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]
        channels = image_shape[3]

        # Explicitly pad the image
        image = _pad(image, filter_shape, mode=padding, constant_values=constant_values)

        area = filter_shape[0] * filter_shape[1]

        floor = (area + 1) // 2
        ceil = area // 2 + 1

        patches = tf.image.extract_patches(
            image,
            sizes=[1, filter_shape[0], filter_shape[1], 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        patches = tf.reshape(patches, shape=[batch_size, height, width, area, channels])

        patches = tf.transpose(patches, [0, 1, 2, 4, 3])

        # Note the returned median is casted back to the original type
        # Take [5, 6, 7, 8] for example, the median is (6 + 7) / 2 = 3.5
        # It turns out to be int(6.5) = 6 if the original type is int
        top = tf.nn.top_k(patches, k=ceil).values
        if area % 2 == 1:
            median = top[:, :, :, :, floor - 1]
        else:
            median = (top[:, :, :, :, floor - 1] + top[:, :, :, :, ceil - 1]) / 2

        output = tf.cast(median, image.dtype)
        output = from_4D_image(output, original_ndims)
        return output

def normalize_tuple(value, n, name):
    """Transforms an integer or iterable of integers into an integer tuple.

    A copy of tensorflow.python.keras.util.

    Args:
      value: The value to validate and convert. Could an int, or any iterable
        of ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.

    Returns:
      A tuple of n integers.

    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise TypeError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        if len(value_tuple) != n:
            raise ValueError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    "The `"
                    + name
                    + "` argument must be a tuple of "
                    + str(n)
                    + " integers. Received: "
                    + str(value)
                    + " "
                    "including element "
                    + str(single_value)
                    + " of type"
                    + " "
                    + str(type(single_value))
                )
        return value_tuple


def _pad(
    image: TensorLike,
    filter_shape: Union[List[int], Tuple[int]],
    mode: str = "CONSTANT",
    constant_values: TensorLike = 0,
) -> tf.Tensor:
    """Explicitly pad a 4-D image.

    Equivalent to the implicit padding method offered in `tf.nn.conv2d` and
    `tf.nn.depthwise_conv2d`, but supports non-zero, reflect and symmetric
    padding mode. For the even-sized filter, it pads one more value to the
    right or the bottom side.

    Args:
      image: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: A `tuple`/`list` of 2 integers, specifying the height
        and width of the 2-D filter.
      mode: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
    """
    if mode.upper() not in {"REFLECT", "CONSTANT", "SYMMETRIC"}:
        raise ValueError(
            'padding should be one of "REFLECT", "CONSTANT", or "SYMMETRIC".'
        )
    constant_values = tf.convert_to_tensor(constant_values, image.dtype)
    filter_height, filter_width = filter_shape
    pad_top = (filter_height - 1) // 2
    pad_bottom = filter_height - 1 - pad_top
    pad_left = (filter_width - 1) // 2
    pad_right = filter_width - 1 - pad_left
    paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    return tf.pad(image, paddings, mode=mode, constant_values=constant_values)

### Segment

def segment_image(image):
    """
    Apply simple thresholding to segment the image.
    
    Args:
    - image: The input image tensor.
    
    Returns:
    - segmented_image: The segmented image tensor, with foreground as 1 and background as 0.
    """
    # Convert the image to grayscale if it's not already
    image_gray = tf.image.rgb_to_grayscale(image) if image.shape[-1] == 3 else image
    
    # Normalize the image tensor to [0, 1]
    image_normalized = image_gray /255.0 #TODO : Check normalizing in this stage
    
    # Define the threshold value
    threshold = 0.5 # This is a normalized threshold since the image is normalized
    
    # Apply thresholding
    segmented_image = tf.where(image_normalized < threshold, 0.0, 1.0)
    
    # Ensure the segmented image is returned in a suitable format for visualization
    
    return segmented_image

def apply_dwt_single(image):
    # Ensure image is 2D by removing the last dimension if it's 1
    if image.shape[-1] == 1:
        image = tf.squeeze(image, axis=-1)

    # coeffs = pywt.dwt2(image_np, 'haar')
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    return LL

def apply_dwt_thrice(image):
    for i in range(3):
        image = apply_dwt_single(image)
    # After processing, add back the channel dimension
    image = np.expand_dims(image, axis=-1)
    return image

# Image will be tensor given by a tf.dataset
def dwt_applier(image):
    # Ensure operations are done outside TensorFlow's graph to use numpy and pywt
    image_dwt = tf.py_function(func=apply_dwt_thrice, inp=[image], Tout=tf.float64)

    # Ensure the output tensor has the right shape and type
    # image_dwt.set_shape((None, None, 1))  # We know the final channel dimension, but not spatial dimensions
    return image_dwt

def prepare_image(image):
    img_array = img_to_array(image)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims

def tensor_to_base64(tensor):
    """
    Converts a TensorFlow tensor to a base64-encoded binary image representation.
    
    Args:
    - tensor (tf.Tensor): A TensorFlow tensor representing an image.
    
    Returns:
    - str: A base64-encoded string representing the binary image.
    """
    # Ensure tensor values are in the appropriate range [0, 255] and of type uint8
    tensor = tf.cast(tensor * 255, tf.uint8)
    
    # Convert the TensorFlow tensor to a NumPy array
    np_image = tensor.numpy()
    
    # Squeeze to remove single-dimension entries from the shape
    np_image = np.squeeze(np_image)
    
    # Check and adjust shape
    if np_image.ndim not in [2, 3]:
        raise ValueError("Tensor must be for grayscale or RGB image")
    
    # If single color channel (grayscale) with extra dimension, squeeze it out
    if np_image.ndim == 3 and np_image.shape[-1] == 1:
        np_image = np_image.reshape(np_image.shape[:-1])
    
    # Convert to PIL Image
    image = Image.fromarray(np_image)
    
    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # Using PNG as it's lossless
    buffer.seek(0)
    
    # Encode the binary data to base64
    base64_image = base64.b64encode(buffer.read()).decode('utf-8')
    
    return base64_image

def preprocess_image(image):
    image_binary = base64.b64decode(image)
    image_tensor = load_and_convert_image_from_binary(image_binary)  
    image_median_filter = median_filter2d(image_tensor) 
    image_segmented = segment_image(image_median_filter)
    image_dwt = dwt_applier(image_segmented)
    image_prepared = prepare_image(image_dwt) 

    image_segmented_base64 = tensor_to_base64(image_segmented)

    return image_segmented_base64, image_prepared