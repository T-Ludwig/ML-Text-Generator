import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
path_to_file = 'shakespear.txt'
text = open(path_to_file, 'r').read()
print(text[:500])