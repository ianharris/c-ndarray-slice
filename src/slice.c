#define PY_SSIZE_T_CLEAN
#include <Python.h>

// define NO_IMPORT_ARRAY and PYARRAY_UNIQUE_SYMBOL required for import_array to be
// called in another compilation unit
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL c_ndarray_slice_ARRAY_API

#include <numpy/arrayobject.h>
#include <stdio.h>

#include "slice.h"

static void limit_index(int *index, int max)
{
  if(*index < 0) {
    *index = 0;
  } else if (*index > max) {
    *index = max;
  }
}

PyObject *slice(PyObject *dummy, PyObject *args)
{
  // declare variables
  PyObject *array;
  PyArrayObject *n_array;
  int start_index, end_index;
  int num_array_dims;
  npy_intp *array_dims;
  npy_intp *array_strides;
  npy_intp s_array_dims[NPY_MAXDIMS];

  // parse the input args and kwargs
  if(!PyArg_ParseTuple(
    args,
    "Oii|$i",
    &array,
    &start_index,
    &end_index)) {
    return NULL;
  }

  // get the number of dimensions of the input array
  num_array_dims = PyArray_NDIM(array);

  // get the dimensions of the input array
  array_dims = PyArray_DIMS(array);

  // limit start_index and end_index
  limit_index(&start_index, array_dims[0]);
  limit_index(&end_index, array_dims[0]);

  // get the strides of the array
  array_strides = PyArray_STRIDES(array);

  // set the dimensions of the new ndarray to that of input ndarray
  for(int i=1; i<num_array_dims; ++i){
    s_array_dims[i] = array_dims[i];
  }
  // update the dimension value for the axis the slice will occur on
  s_array_dims[0] = end_index - start_index;

  // incref the PyArray_Descr
  Py_INCREF(PyArray_DESCR(array));

  // create the new slice array
  n_array = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type,
            PyArray_DESCR(array),
            num_array_dims,
            s_array_dims,
            array_strides,
            PyArray_DATA(array) + start_index*array_strides[0],
            PyArray_FLAGS(array),
            (PyObject *)array);

  // set the new array's base array
  n_array->base = ((PyArrayObject *)array)->base?((PyArrayObject *)array)->base:array;

  // incref the base array
  Py_INCREF(n_array->base);

  // return the new array
  return n_array;
}
