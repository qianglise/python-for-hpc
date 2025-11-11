# CUPY

:::{questions}
- What syntax is used to make a lesson?
- How do you structure a lesson effectively for teaching?
- `questions` are at the top of a lesson and provide a starting
  point for what you might learn.  It is usually a bulleted list.
:::

:::{objectives}
- Show a complete lesson page with all of the most common
  structures.
- ...

This is also a holdover from the carpentries-style.  It could
usually be left off.
:::



Another excellent tool for writing Python code to run on GPUs is CuPy.
CuPy is a NumPy/SciPy-compatible array library for
GPU-accelerated computing with Python,
which implements most of the NumPy/SciPy operations
and acts as a drop-in replacement to run existing
NumPy/SciPy code on both NVIDIA CUDA or AMD ROCm platforms.
By design, the CuPy interface is as close as possible to NumPy/SciPy,
making code porting much easier.

The introduction should be a high level overview of what is on the
page and why it is interesting.


The lines below (only in the source) will set the default highlighting
language for the entire page.

:::{highlight} python
:::

## Basics of CuPy

CuPy's syntax here is identical to that of NumPy. A list of
NumPy/SciPy APIs and its corresponding CuPy implementations
is summarised
[here](https://docs.cupy.dev/en/stable/reference/comparison.html#comparison-table).

In summary, CuPy provides N-dimensional array (ndarray),
sparse matrices, and the associated routines for GPU devices,
most having the same API as NumPy/SciPy.

Let us take a look at the following code snippet which calculates the L2-norm of an array.
Note how simple it is to run on a GPU device using CuPy, i.e. essentially by changing np to cp.

```
>> import numpy as np
>> import cupy as cp
>> x_cpu = np.array([1, 2, 3])
>> l2_cpu = np.linalg.norm(x_cpu)
>> x_gpu = cp.array([1 ,2 ,3])
>> l2_gpu = cp.linalg.norm(x_gpu)
```

:::{admonition} Warning
:class: warning

One recommendation is not to change the import line
in the code to something like `import cupy as np`,
which can cause problems if you need to use
NumPy code and not CuPy code.

:::


### Conversion to/from NumPy arrays

Although cupy.ndarray is the CuPy counterpart of NumPy numpy.ndarray, the main difference is that cupy.ndarray resides on `the current device`, and they are not implicitly convertible to each other.

- To convert numpy.ndarray to cupy.ndarray, use cupy.array() or cupy.asarray()
- To convert cupy.ndarray to numpy.ndarray, use cupy.asnumpy() or cupy.ndarray.get()

As in the above example, the variable l2_gpu remains on the GPU. One has to copy the variable back to the CPU explicitly e.g. if printing the result to the screen is needed.

```
>> import numpy as np
>> import cupy as cp
>> x_cpu = np.array([1, 2, 3])
>> l2_cpu = np.linalg.norm(x_cpu)
>> x_gpu = cp.array([1 ,2 ,3])
>> l2_gpu = cp.linalg.norm(x_gpu)
>> # copy l2_gpu from GPU to CPU for e.g. printing
>> l2_cpu = cp.asnumpy(l2_gpu)
```


Note that converting between cupy.ndarray and numpy.ndarray incurs data transfer between the host (CPU) device and the GPU device, which is costly in terms of performance.

> [!NOTE]
> Note that the device will be called <CUDA Device 0> even if you are on AMD GPUs.


## User-Defined Kernels

## CuPy vs Numpy/SciPy


Although the CuPy team focuses on providing a complete
NumPy/SciPy API coverage to become a full drop-in replacement,
some important differences between CuPy and NumPy should be noted,
one should keep these differences in mind when porting NumPy code to CuPy.

- Some casting behaviors from floating point to integer
are not defined in the C++ specification. The casting
from a negative floating point to an unsigned integer
and from infinity to an integer are examples.
- CuPy random methods support the dtype argument.
- Out-of-bounds indices and duplicate values in indices are handled differently.
- Reduction methods return zero-dimension arrays.



### Cast behavior from float to integer

Some casting behaviors from float to integer are not defined in C++ specification. The casting from a negative float to unsigned integer and infinity to integer is one of such examples. The behavior of NumPy depends on your CPU architecture. This is the result on an Intel CPU:

```
np.array([-1], dtype=np.float32).astype(np.uint32)
array([4294967295], dtype=uint32)

cupy.array([-1], dtype=np.float32).astype(np.uint32)
array([0], dtype=uint32)
```

```
np.array([float('inf')], dtype=np.float32).astype(np.int32)
array([-2147483648], dtype=int32)

cupy.array([float('inf')], dtype=np.float32).astype(np.int32)
array([2147483647], dtype=int32)
```

### Random methods support dtype argument

NumPy’s random value generator does not support a dtype argument and instead always returns a float64 value. We support the option in CuPy because cuRAND, which is used in CuPy, supports both float32 and float64.

```
np.random.randn(dtype=np.float32)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: randn() got an unexpected keyword argument 'dtype'

cupy.random.randn(dtype=np.float32)    
array(0.10689262300729752, dtype=float32)
```

### Out-of-bounds indices

CuPy handles out-of-bounds indices differently by default from NumPy when using integer array indexing. NumPy handles them by raising an error, but CuPy wraps around them.

```
x = np.array([0, 1, 2])

x[[1, 3]] = 10
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: index 3 is out of bounds for axis 1 with size 3

x = cupy.array([0, 1, 2])

x[[1, 3]] = 10

x
array([10, 10,  2])
```

### Duplicate values in indices

CuPy’s __setitem__ behaves differently from NumPy when integer arrays reference the same location multiple times. In that case, the value that is actually stored is undefined. Here is an example of CuPy.

```
a = cupy.zeros((2,))

i = cupy.arange(10000) % 2

v = cupy.arange(10000).astype(np.float32)

a[i] = v

a  
array([ 9150.,  9151.])
```

NumPy stores the value corresponding to the last element among elements referencing duplicate locations.

```
a_cpu = np.zeros((2,))

i_cpu = np.arange(10000) % 2

v_cpu = np.arange(10000).astype(np.float32)

a_cpu[i_cpu] = v_cpu

a_cpu
array([9998., 9999.])
```

### Zero-dimensional array

Reduction methods

NumPy’s reduction functions (e.g. numpy.sum()) return scalar values (e.g. numpy.float32). However CuPy counterparts return zero-dimensional cupy.ndarray s. That is because CuPy scalar values (e.g. cupy.float32) are aliases of NumPy scalar values and are allocated in CPU memory. If these types were returned, it would be required to synchronize between GPU and CPU. If you want to use scalar values, cast the returned arrays explicitly.

```
type(np.sum(np.arange(3))) == np.int64
True

type(cupy.sum(cupy.arange(3))) == cupy.ndarray
True
```

Type promotion

CuPy automatically promotes dtypes of cupy.ndarray s in a function with two or more operands, the result dtype is determined by the dtypes of the inputs. This is different from NumPy’s rule on type promotion, when operands contain zero-dimensional arrays. Zero-dimensional numpy.ndarray s are treated as if they were scalar values if they appear in operands of NumPy’s function, This may affect the dtype of its output, depending on the values of the “scalar” inputs.

```
(np.array(3, dtype=np.int32) * np.array([1., 2.], dtype=np.float32)).dtype
dtype('float32')

(np.array(300000, dtype=np.int32) * np.array([1., 2.], dtype=np.float32)).dtype
dtype('float64')

(cupy.array(3, dtype=np.int32) * cupy.array([1., 2.], dtype=np.float32)).dtype
dtype('float64')
```

### Matrix type (numpy.matrix)

SciPy returns numpy.matrix (a subclass of numpy.ndarray) when dense matrices are computed from sparse matrices (e.g., coo_matrix + ndarray). However, CuPy returns cupy.ndarray for such operations.

There is no plan to provide numpy.matrix equivalent in CuPy. This is because the use of numpy.matrix is no longer recommended since NumPy 1.15.

### Data types

Data type of CuPy arrays cannot be non-numeric like strings or objects. See Overview for details.

### Universal Functions only work with CuPy array or scalar

Unlike NumPy, Universal Functions in CuPy only work with CuPy array or scalar. They do not accept other objects (e.g., lists or numpy.ndarray).

```
np.power([np.arange(5)], 2)
array([[ 0,  1,  4,  9, 16]])

cupy.power([cupy.arange(5)], 2)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Unsupported type <class 'list'>
```

### Random seed arrays are hashed to scalars

Like Numpy, CuPy’s RandomState objects accept seeds either as numbers or as full numpy arrays.

```
seed = np.array([1, 2, 3, 4, 5])

rs = cupy.random.RandomState(seed=seed)
```

However, unlike Numpy, array seeds will be hashed down to a single number and so may not communicate as much entropy to the underlying random number generator.

### NaN (not-a-number) handling

By default CuPy’s reduction functions (e.g., cupy.sum()) handle NaNs in complex numbers differently from NumPy’s counterparts:

```
a = [0.5 + 3.7j, complex(0.7, np.nan), complex(np.nan, -3.9), complex(np.nan, np.nan)]
a_np = np.asarray(a)
print(a_np.max(), a_np.min())
(0.7+nanj) (0.7+nanj)

a_cp = cp.asarray(a_np)
print(a_cp.max(), a_cp.min())
(nan-3.9j) (nan-3.9j)
```

The reason is that internally the reduction is performed in a strided fashion, thus it does not ensure a proper comparison order and cannot follow NumPy’s rule to always propagate the first-encountered NaN. Note that this difference does not apply when CUB is enabled (which is the default for CuPy v11 or later.)

### Contiguity / Strides

To provide the best performance, the contiguity of a resulting ndarray is not guaranteed to match with that of NumPy’s output.

```
a = np.array([[1, 2], [3, 4]], order='F')
print((a + a).flags.f_contiguous)
True
```

```
a = cp.array([[1, 2], [3, 4]], order='F')
print((a + a).flags.f_contiguous)
False
```

## Interoperability

CuPy implements standard APIs for data exchange and interoperability, such as  __array_ufunc__ interface (see NEP 13 — A Mechanism for Overriding Ufuncs for details), __array_function__ interface (see NEP 18 — A dispatch mechanism for NumPy’s high level array functions for details), and other [Python Array API Standard](https://data-apis.org/array-api/latest). This enables e.g. NumPy to be directly operated on CuPy arrays.

Note that the return type of these operations is still consistent with the initial type

> [!NOTE]
> __array_ufunc__ feature requires NumPy 1.13 or later.
> __array_function__ feature requires NumPy 1.16 or later; As of NumPy 1.17, __array_function__ is enabled by default.
:::{discussion}
Discuss the following.

- A discussion section
- Another discussion topic
:::



## Section

```
print("hello world")
# This uses the default highlighting language
```

```python
print("hello world)
```



## Exercises: description

:::{exercise} Exercise Topic-1: imperative description of exercise
Exercise text here.
:::

:::{solution}
Solution text here
:::



## Summary

A Summary of what you learned and why it might be useful.  Maybe a
hint of what comes next.



## See also

- Other relevant links
- Other link



:::{keypoints}
- What the learner should take away
- point 2
- ...

This is another holdover from the carpentries style.  This perhaps
is better done in a "summary" section.
:::
