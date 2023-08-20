#define PY_SSIZE_T_CLEAN
#include <Python.h>

// define NO_IMPORT_ARRAY and PYARRAY_UNIQUE_SYMBOL required for import_array to be
// called in another compilation unit
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL c_ndarray_slice_ARRAY_API

#include <numpy/arrayobject.h>
#include <stdio.h>

#include "slice.h"

PyObject *slice(PyObject *dummy, PyObject *args, PyObject *kwds)
{
  // declare variables
  PyObject *array;
  PyArray_Descr *array_descr;
  int start_index, end_index, axis;
  int num_array_dims;
  char *keyword_names[] = {"", "", "", "axis", NULL};
  npy_intp *array_dims;
  npy_intp *s_array_dims;
  npy_intp *array_strides;

  void *s_data;

  // parse the input args and kwargs
  if(!PyArg_ParseTupleAndKeywords(
    args,
    kwds,
    "Oii|$i",
    keyword_names,
    &array,
    &start_index,
    &end_index,
    &axis)) {
    return NULL;
  }

  // get the number of dimensions of the input array
  num_array_dims = PyArray_NDIM(array);

  // get the dimensions of the input array
  array_dims = PyArray_DIMS(array);

  // get the strides of the array
  array_strides = PyArray_STRIDES(array);

  // get a reference to the array description
  array_descr = PyArray_DESCR(array);

  // enforce certain values for axis
  if(axis >= num_array_dims || axis < -num_array_dims) {
    PyErr_SetString(
      PyExc_ValueError, 
      "'axis' value specified too large (positive value) or too small (negative value) for the provided array\n"
    );
    return NULL;
  }

  // modify axis so that it is the positive form
  if(axis < 0) {
    axis += num_array_dims;
  }

  // malloc space for the new array dims
  s_array_dims = (npy_intp *)malloc(sizeof(npy_intp)*num_array_dims);

  // set the dimensions of the new ndarray to that of input ndarray
  for(int i=0; i<num_array_dims; ++i){
    s_array_dims[i] = array_dims[i];
  }
  // update the dimension value for the axis the slice will occur on
  s_array_dims[axis] = end_index - start_index;

  // calculate the size of the array
  int s_array_size = 1;
  for (int i = 0; i < num_array_dims; ++i) {
    s_array_size *= s_array_dims[i];
  }
  // allocate space for the new array
  s_data = malloc(s_array_size * PyArray_ITEMSIZE(array));

  // memcpy data from the original array to fill s_data
  if(axis == 0) {
    memcpy(
      s_data,
      PyArray_DATA(array) + start_index*array_strides[0],
      (end_index-start_index)*array_strides[0]
    );
  } else {
    // calculate the number of blocks to copy
    int num_blocks = 1;
    for(int i=0; i<axis; ++i) {
      num_blocks *= array_dims[i];
    }

    // loop over the data copying the required elements
    for(int i=0; i<num_blocks; ++i) {
      memcpy(
        s_data + i*(end_index-start_index)*array_strides[axis],
        PyArray_DATA(array) + i*array_strides[axis-1] + start_index*array_strides[axis],
        (end_index-start_index)*array_strides[axis]
      );
    }
  }
  
  // create the new array
  return PyArray_SimpleNewFromData(num_array_dims, s_array_dims, PyArray_TYPE(array), s_data);
}
