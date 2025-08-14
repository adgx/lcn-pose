import os
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

print(os.getcwd()) 
print(os.path.exists('./experiment/test10/checkpoints/best/model-178002'))

print_tensors_in_checkpoint_file(file_name='./experiment/test4/checkpoints/final/model-191400', 
                                 tensor_name='', all_tensors=False, all_tensor_names=False)