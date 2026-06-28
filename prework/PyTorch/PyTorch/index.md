{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "c3c4451c-46d1-4379-8d81-ee0e693eb323",
   "metadata": {},
   "source": [
    "# PyTorch\n",
    "\n",
    "PyTorch is an open-source deep learning library, originally developed by Meta Platforms and currently developed with support from the Linux Foundation. The successor to Torch, PyTorch provides a high-level API that builds upon optimised, low-level implementations of deep learning algorithms and architectures, such as the Transformer, or SGD. Notably, this API simplifies model training and inference to a few lines of code. PyTorch allows for automatic parallelization of training and, internally, implements CUDA bindings that speed training further by leveraging GPU resources."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e08d096-b8c3-4b2a-aa90-0660b962405a",
   "metadata": {},
   "source": [
    "## `squeeze()` vs `unsqueeze()`"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5ddea120-a398-4046-982c-802cfa789b74",
   "metadata": {},
   "source": [
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
   "execution_count": 8,
   "id": "cf89bd49-168a-4f7c-9484-f3763d14afbe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original image shape torch.Size([3, 32, 32])\n",
      "Unsqueezed image shape torch.Size([1, 3, 32, 32])\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "\n",
    "image_32x32 = torch.randn(3, 32, 32)\n",
    "unsqueezed_image_32x32 = image_32x32.unsqueeze(dim=0)\n",
    "\n",
    "print(\"Original image shape\", image_32x32.shape)\n",
    "print(\"Unsqueezed image shape\", unsqueezed_image_32x32.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2da328e6-5f7a-4343-992e-7806f42944f4",
   "metadata": {},
   "source": [
    "And you can use `squeeze()` to squeeze out the extra superficial dimension from the unsqueezed tensor.\n",
    "\n",
    "**NOTE:** `squeeze()` only removes the dimension that has a size of `1` using `squeeze()` on the original `image_32x32` image tensor will not change the shape of the tensor."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "dc3bd8e3-8d96-42af-a5bc-1263f1b7525f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Squeezed image from unsqueezed image shape torch.Size([3, 32, 32])\n",
      "Squeezed image from original image shape torch.Size([3, 32, 32])\n"
     ]
    }
   ],
   "source": [
    "squeezed_image_32x32 = unsqueezed_image_32x32.squeeze(dim=0)\n",
    "squeezed_image_original = image_32x32.squeeze(dim=0)\n",
    "\n",
    "print(\"Squeezed image from unsqueezed image shape\", squeezed_image_32x32.shape)\n",
    "print(\"Squeezed image from original image shape\", squeezed_image_original.shape)"
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
