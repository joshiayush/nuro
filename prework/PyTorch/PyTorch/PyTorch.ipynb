{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "c3c4451c-46d1-4379-8d81-ee0e693eb323",
   "metadata": {},
   "source": [
    "# PyTorch\n",
    "\n",
    "PyTorch is an open-source deep learning library, originally developed by Meta Platforms and currently developed with support from the Linux Foundation. The successor to Torch, PyTorch provides a high-level API that builds upon optimised, low-level implementations of deep learning algorithms and architectures, such as the **Transformer**, or **SGD**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "58c4616f-85d2-4c50-81f6-0a16158de5da",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "12b749b3-9c2a-4d65-bf19-d77c183b9a88",
   "metadata": {},
   "source": [
    "## Tensors\n",
    "\n",
    "At its core, PyTorch is a library for processing tensors. A tensor is a number, vector, matrix, or any n-dimensional array. Let's create a tensor with a single number."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "3974b752-2bea-47df-9fec-4d96427db48f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor(4.)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t1 = torch.tensor(4.)\n",
    "t1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "69fe539e-66bb-4ea8-94b1-bb6e5908563e",
   "metadata": {},
   "source": [
    "`4.` is a shorthand for `4.0`. It is used to indicate to Python (and PyTorch) that you want to create a floating-point number. We can verify this by checking the `dtype` attribute of our tensor."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "ab3b38a6-5fec-492f-98ac-7599614369ca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.float32"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t1.dtype"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dce25fb0-ee75-48b1-82b8-fec6bcbea629",
   "metadata": {},
   "source": [
    "Let's try creating more complex tensors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "330964c4-b516-41a5-a184-b3980c6b989e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([1., 2., 3., 4.])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Vector\n",
    "t2 = torch.tensor([1., 2, 3, 4])\n",
    "t2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "7584b008-78cf-413b-bce7-a13193b746cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[ 5.,  6.],\n",
       "        [ 7.,  8.],\n",
       "        [ 9., 10.]])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Matrix\n",
    "t3 = torch.tensor([[5., 6], [7, 8], [9, 10]])\n",
    "t3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "5de84e8a-bf38-4cf1-927e-b22a03e4dff9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[[11, 12, 13],\n",
       "         [14, 15, 16]],\n",
       "\n",
       "        [[17, 18, 19],\n",
       "         [20, 21, 22]]])"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 3-dimensional array\n",
    "t4 = torch.tensor([[[11, 12, 13], [14, 15, 16]], [[17, 18, 19], [20, 21, 22]]])\n",
    "t4"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9dff6006-efdb-442b-beef-8746962cefbc",
   "metadata": {},
   "source": [
    "Tensors can have any number of dimensions and different length along each dimension. We can inspect the length along each dimension by using the `.shape` property of the tensor."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "96adaf9b-051f-4890-91a5-e0ebf2beb375",
   "metadata": {},
   "source": [
    "As a rule of thumb, to count the `dimesion` of a tensor start by picking up the *bracket* where the **tensor begins** and then **count the elements** in those brackets that will become the **length** of your first dimension to determine other dimensions and their respective lengths repeat the same process for inner brackets.\n",
    "\n",
    "For example to count for `t4`:\n",
    "- Inside the first bracket there are `2` elements so the first dimension becomes `2`\n",
    "- Inside the second bracket there are `2` elements so the second dimension becomes `2`\n",
    "- Inside the third bracket there are `3` elements so the third dimension becomes `3`\n",
    "\n",
    "So, `t4.shape` should be `(2, 2, 3)`, let's check."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "adee4d75-1f30-4ad7-a1e2-387ccad09a69",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([2, 2, 3])"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t4.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f7d29666-b4a0-47a8-823d-bfd359361302",
   "metadata": {},
   "source": [
    "### Operations and Gradients\n",
    "\n",
    "We can combine tensors with the usual arithmetic operations. Let's look at an example:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9c646bb8-65b3-4e81-bc4a-1d74c614878d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(tensor(3.), tensor(4., requires_grad=True), tensor(5., requires_grad=True))"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = torch.tensor(3.)\n",
    "w = torch.tensor(4., requires_grad=True)\n",
    "b = torch.tensor(5., requires_grad=True)\n",
    "x, w, b"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1726dba-b418-4d23-a88a-3e3e7ff7f434",
   "metadata": {},
   "source": [
    "We have created three tensors: `x`, `w`, `b`, all numbers. `w` and `b` have additional parameter `requires_grad` which is set to `True`. We'll se what that does in a moment.\n",
    "\n",
    "Let's create a new tensor `y` by combining these tensors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "5d2ca308-0571-4207-9e7f-432073b0916e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor(17., grad_fn=<AddBackward0>)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = w * x + b\n",
    "y"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ff777653-4831-4da7-8712-aa619a9986b9",
   "metadata": {},
   "source": [
    "As expected, `y` is a tensor with the value `3 * 4 + 5 = 17`. What makes PyTorch unique is the ability to automatically compute the derivate of `y` w.r.t the tensors that have `requires_grad` set to `True`. This feature of PyTorch is called *autograd* (automatic gradients).\n",
    "\n",
    "To compute the derivates, we can invoke `.backward()` method on our `y`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "5a088e6e-4463-4eef-8789-2db06a8d6b8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "y.backward()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "adf6feac-5226-4872-8e7f-627f53619891",
   "metadata": {},
   "source": [
    "The derivative of `y` w.r.t the input tensors are stored in the `.grad` property of the respective input tensors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "04720edc-17a2-4092-8375-3fd4251b0557",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dy/dx None\n",
      "dy/dw tensor(3.)\n",
      "dy/db tensor(1.)\n"
     ]
    }
   ],
   "source": [
    "print(\"dy/dx\", x.grad)\n",
    "print(\"dy/dw\", w.grad)\n",
    "print(\"dy/db\", b.grad)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "876e2815-9751-41eb-903f-a27c87563a0c",
   "metadata": {},
   "source": [
    "As expected, $\\frac{dy}{dx}$ has the same value as `x` i.e., `3`, and $\\frac{dy}{db}$ has the value `1`. Note that $\\frac{dy}{dx}$ is `None` because `x` doesn't have `requires_grad` set to `True`.\n",
    "\n",
    "The \"grad\" in here is short for *gradient*, which is another term for derivative. The term *gradient* is primarily used when dealing with **vectors** and **metrices**."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6de57827-b21b-4529-b5c9-00c44e30823d",
   "metadata": {},
   "source": [
    "### Functions\n",
    "\n",
    "Apart from arithmetic operations, the `torch` module also contains many functions for creating and manipulating tensors. Let's look at some examples."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b849ea5a-7ccd-41cf-a483-2aa227f598ed",
   "metadata": {},
   "source": [
    "#### torch.full()\n",
    "\n",
    "Create a tensor with a fixed value for every element."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "257bface-f2f9-4711-8dba-5c7471bbdd83",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[42, 42],\n",
       "        [42, 42],\n",
       "        [42, 42]])"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t6 = torch.full((3, 2), 42)\n",
    "t6"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f0fbc5e3-3347-4945-a99a-b607d70f1f66",
   "metadata": {},
   "source": [
    "#### torch.cat()\n",
    "\n",
    "Concatenate two tensors with compatible shapes using [`torch.cat()`](https://docs.pytorch.org/docs/2.13/generated/torch.cat.html#torch-cat). By default it concatenates along the `0` dimension."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "9f9b8081-d08e-45ad-8f5e-882c8df209ce",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[ 5.,  6.],\n",
       "        [ 7.,  8.],\n",
       "        [ 9., 10.],\n",
       "        [42., 42.],\n",
       "        [42., 42.],\n",
       "        [42., 42.]])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t7 = torch.cat((t3, t6))\n",
    "t7"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "88584196-9606-43a6-8cd6-86497fcd26d3",
   "metadata": {},
   "source": [
    "#### torch.sin()\n",
    "\n",
    "Compute trignometric sine of each element of a tensor."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "6b7e2782-99b0-4a36-9383-12c9e679d88c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[-0.9589, -0.2794],\n",
       "        [ 0.6570,  0.9894],\n",
       "        [ 0.4121, -0.5440],\n",
       "        [-0.9165, -0.9165],\n",
       "        [-0.9165, -0.9165],\n",
       "        [-0.9165, -0.9165]])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t8 = torch.sin(t7)\n",
    "t8"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2d51608-d80c-480b-850a-5d82e4c0ddd5",
   "metadata": {},
   "source": [
    "#### torch.reshape()\n",
    "\n",
    "Change the view of the underlying tensor using [`torch.reshape()`](https://docs.pytorch.org/docs/2.13/generated/torch.reshape.html#torch-reshape)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "d6c6cc08-eff8-44ca-8eea-797cf435a3a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[[-0.9589, -0.2794],\n",
       "         [ 0.6570,  0.9894]],\n",
       "\n",
       "        [[ 0.4121, -0.5440],\n",
       "         [-0.9165, -0.9165]],\n",
       "\n",
       "        [[-0.9165, -0.9165],\n",
       "         [-0.9165, -0.9165]]])"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t9 = torch.reshape(t8, (3, 2, 2))\n",
    "t9"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5ddea120-a398-4046-982c-802cfa789b74",
   "metadata": {},
   "source": [
    "#### torch.squeeze() vs torch.unsqueeze()\n",
    "\n",
    "`unsqueeze()` \"adds\" a superficial `1` dimension to tensor (at the specified dimension), while `squeeze()` removes all superficial `1` dimensions from tensor.\n",
    "\n",
    "A `1` dimension is superficial in the sense that it does not add any more elements to the tensor than would be there without it.\n",
    "\n",
    "<div align=\"center\">\n",
    "<img src=\"../../../static/images/squeeze_vs_unsqueeze_2d.png\">\n",
    "</div>\n",
    "\n",
    "`unsqueeze()` is useful for providing single sample to the network (which requires first dimension to be batch), for images it would be:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "9bdac2b4-6c52-418f-8046-bee04d50bf9a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([3, 32, 32])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t10 = torch.randn(3, 32, 32)\n",
    "t10.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "cf89bd49-168a-4f7c-9484-f3763d14afbe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([1, 3, 32, 32])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t11 = t10.unsqueeze(dim=0)\n",
    "t11.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2da328e6-5f7a-4343-992e-7806f42944f4",
   "metadata": {},
   "source": [
    "And you can use `squeeze()` to squeeze out the extra superficial dimension from the unsqueezed tensor.\n",
    "\n",
    "> **Note:** `squeeze()` only removes the dimension that has a size of `1` using `squeeze()` on the original image tensor will not change the shape of the tensor."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "29f4eea5-0e62-41c0-97d6-6d5caad6702c",
   "metadata": {},
   "source": [
    "`squeeze` will squeeze out the superficial dimension in `t11`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "92c95204-f30e-4470-b169-37a4190688ae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([3, 32, 32])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t12 = t11.squeeze(dim=0)\n",
    "t12.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c82277a9-c66b-4e90-87da-04b3a2a6aa5d",
   "metadata": {},
   "source": [
    "`squeeze` will not do anything since there's no superficial dimension in `t10`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "dc3bd8e3-8d96-42af-a5bc-1263f1b7525f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([3, 32, 32])"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t13 = t10.squeeze(dim=0)\n",
    "t13.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bfe9c3da-7fde-4b75-8026-6bacde430c1b",
   "metadata": {},
   "source": [
    "#### torch.from_numpy\n",
    "\n",
    "Create a [`Tensor`](https://docs.pytorch.org/docs/2.13/tensors.html#torch.Tensor) from a [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "3fe2c083-2008-4d81-9774-8207a1d6dc78",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "2bda6711-e67e-4996-b7dc-1345f1855f3b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([1., 2., 3.], dtype=torch.float64)"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.array([1., 2., 3.])\n",
    "t14 = torch.from_numpy(arr)\n",
    "t14"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ca43deaa-ada0-42ea-a5c2-8ed6c9663902",
   "metadata": {},
   "source": [
    "And then after performing GPU-bound operations on `t14` you can also get your `numpy.array` back by calling `.numpy()` on a `Tensor`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "764b8a19-6d8d-4513-8522-3fd7957bbfed",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 2., 3.])"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t14.numpy()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d61cd823-dc91-4f2d-95e1-3096f558a8b3",
   "metadata": {},
   "source": [
    "You can see the type is `array` whereas previously the type was `tensor`."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
