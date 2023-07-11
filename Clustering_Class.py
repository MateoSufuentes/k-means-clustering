# imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Clustering:
  def __init__(self, num_data, num_clusters, x_and_y_data, x_and_y_reference):
    self.num_data_points = num_data # int with num of data points from excel
    self.num_clusters = num_clusters # int with num of clusters
    self.data_points = x_and_y_data # np 2D array or dataframe with x and y coordinates of datapoints
    self.reference_points = x_and_y_reference # np 2D array or dataframe with x and y coordinates of cluster mean points
    self.average_x_y_values = None # np 2D array or dataframe with average x and y coordinates for each cluster (next coordinates for the center points)
    self.total_distance = int(0) # total distance from the data_points to the reference_points
    self.linked_reference_index = None # np array or 1D dataframe with row index of corresponding reference coordinates in reference_points class variable

  # Calculates average_x_y_values, linked_reference_index, and total_distance
  def calculations(self):

    # Initialize linked_reference_index as an np array
    self.linked_reference_index = np.zeros((self.num_data_points,1))

    # Match each data point to cluster
    for val in range (0, self.num_data_points):
        
        # Initialize closest index and corresponding distance from reference point to data point
        curr_closest_index = 0
        curr_closest_dist = 1
        
        # For each data point, iterate through all reference points and chose favorite
        for ref in range (0, self.num_clusters):
            curr_dist = math.sqrt(((self.data_points.iloc[val,0] - self.reference_points.loc[ref,0]) ** 2)+((self.data_points.iloc[val,1] - self.reference_points.loc[ref,1]) ** 2))
            if (curr_dist < curr_closest_dist):
                curr_closest_dist = curr_dist
                curr_closest_index = ref
        
        # Update total_distance and linked_reference_index with newly matched data point
        self.total_distance = self.total_distance + curr_closest_dist
        self.linked_reference_index[val] = curr_closest_index

    # Fill out average_x_y_values
    self.average_x_y_values = pd.DataFrame(index=range(self.num_clusters),columns=range(2))

    # Finds average x and y value for each cluster
    for cluster in range (0, self.num_clusters):

      # Creates boolean array of whether that specific data point is in the given cluster
      has_ref_cluster = np.array(self.linked_reference_index == cluster).T
      
      # Transfers to 1D array
      new_has_ref_cluster = has_ref_cluster[0,:]

      # Divies x and y values into two separate arrays
      x = np.array(self.data_points.iloc[:,0])
      y = np.array(self.data_points.iloc[:,1])

      # Cuts x and y arrays to strictly datapoints belonging to the given cluster
      cut_x = x[new_has_ref_cluster]
      cut_y = y[new_has_ref_cluster]

      # Finds x mean value and y mean value
      mean_x = cut_x.mean()
      mean_y = cut_y.mean()

      # Updates mean values for x and y
      self.average_x_y_values.loc[cluster,0] = mean_x
      self.average_x_y_values.loc[cluster,1] = mean_y

