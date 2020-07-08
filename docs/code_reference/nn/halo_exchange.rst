===================
Halo Exchange Layer
===================

Overview
========

.. automodule:: distdl.nn.halo_exchange

The Halo Exchange distributed data movement primitive pads a tensor distributed
across a cartesian topology with data from neighboring workers.

Motivation
==========

In distributed deep learning (especially on large volumetric data), we often want
to use layers such a convolutional layer or a pooling layer that use a sliding
kernel to perform computation on the input tensor. Doing this in a distributed
setting requires the use of a halo exchange, which pads the current worker in a
cartesian topology with data from neighboring workers, to ensure that computation
is performed correctly.

.. note::
    Halo Exchange diagram

Current implementations of this algorithm for things such as finite difference
methods make assumptions such as a symmetric, centered kernel, balanced halo regions,
or that the entire data is used after the exchange that do not hold in in distributed 
deep learning, specfically when the output tensor is assumed to be distributed
optimally. For example, in distributed :ref:`code_reference/nn/pooling:Pooling Layers`
kernels are usually right looking, leading to an imbalance in the exchange.

.. note::
    Pooling exchange diagram

Thus, it is essential that the Halo Exchange primitive be fully general, i.e. 
that it is dimension independent and can deal with imbalances in the size of halo
regions across and within dimensions.

Implementation
==============

Halo Mixin
----------

The Halo Exchange primitive is designed to be completely independent of the layer
which it is incorporated into, thus the user must provide functionality which computes the
sizes of various pieces of the algorithm, such as halo regions or buffers. This 
is done by creating a mixin that computes the minimum and maximum input range, from
which DistDL :class:`~distdl.nn.mixins.HaloMixin` will compute the information needed
for the exchange. Mixins for convolutional layers and pooling layers are provided
via DistDL :class:`~distdl.nn.mixins.ConvMixin` and :class:`~distdl.nn.mixins.PoolingMixin`.
These classes are used simply for computing values passed to the backend, and do
not need to be reimplemented for a new backend.

Assumptions
-----------

* The halo exchange operation is in-place. This is mainly done to save the cost
  of unecessary re-allocation of the entire tensor. Note that torch is very
  touchy about in-place functions and autograd, so it is crucial that 
  ``ctx.mark_dirty()`` is called on the input to ``forward()``.
* In a cartesian topology where one of the dimesions is 1, no exchange is performed
  in that dimension.

Forward
-------

The forward function copies data from the bulk region of data on neighboring workers
in a cartesian partion into the halo region on the current partition.

THOROUGH EXPLANATION OF NESTED EXCHANGE PATTERN HERE

This can be done in any order of dimension as long as the adjoint function operates on dimensions
in reverse order. Users should take care to make sure that the operators describing
the halo exchange are fully implemented, even if they are implemented implicitly.

Adjoint
-------

The adjoint function adds data from the halo region of data on the current worker into
the 

.. automodule:: distdl.nn.mixins.halo_mixin

API
===

.. currentmodule:: distdl.nn

.. autoclass:: HaloExchange
    :members:
    :exclude-members: forward


.. currentmodule:: distdl.nn.mixins

.. autoclass:: HaloMixin
    :members:
