#define PY_SSIZE_T_CLEAN
#include <Python.h>

// define the PY_ARRAY_UNIQUE_SYMBOL required for import_array
#define PY_ARRAY_UNIQUE_SYMBOL c_ndarray_slice_ARRAY_API

#include <numpy/arrayobject.h>

// import the slice header file
#include "slice.h"

// initialize the PyMethodDef for the module
static PyMethodDef CNdarraySliceMethods[] = {
  {"c_ndarray_slice", slice, METH_VARARGS, "Slice an array in C"},
  {NULL, NULL, 0, NULL}  // Sentinel
};

// initialize the module
static struct PyModuleDef CNdarraySliceModule = {
  PyModuleDef_HEAD_INIT,
  "c_ndarray_slice",
  NULL,
  -1,
  CNdarraySliceMethods
};

// define the module initialization function
PyMODINIT_FUNC
PyInit_c_ndarray_slice(void)
{
  // import array functions from numpy
  import_array();

  return PyModule_Create(&CNdarraySliceModule);
};
