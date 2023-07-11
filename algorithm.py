# imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import Clustering_Class

# FUNCTIONS

# Cutting NaN values
def cutting_nan_vals(x, y):
    x_nan = np.isnan(x)
    y_nan = np.isnan(y)
    both_nan = x_nan | y_nan
    both_numbers = ~both_nan
    cut_x = x[both_numbers]
    cut_y = y[both_numbers]
    combo = np.vstack((cut_x, cut_y)).T
    return combo

# Normalize
def normalize(x, y):
    x_min = min(new_x)
    x_max = max(new_x)
    y_min = min(new_y)
    y_max = max(new_y)
    x_norm = (new_x - x_min) / (x_max - x_min)
    y_norm = (new_y - y_min) / (y_max - y_min)
    combo = np.vstack((x_norm, y_norm)).T
    return combo

# Gives option to graph cooridnates
def option_to_print_graph(x, y):
    print("Would you like to see the graph? Yes or No. Note: if you proceed to make the graph, you will need to restart the program to keep going with the k-means clustering.")
    choice = input()
    if (choice == "Yes"):
        plt.scatter(x, y)
        plt.savefig('see_graph.png')
        plt.show()

def randomized(num_clusters):
    # Preps initial randomized estimates into a dataframe
    pairs, dim = num_clusters, 2
    estimates_x = np.random.rand(pairs,1).T
    estimates_y = np.random.rand(pairs,1).T
    estimates = np.vstack((estimates_x, estimates_y)).T
    df_estimates = pd.DataFrame(estimates)
    print("THIS ROUND'S ESTIMATES: ")
    print(df_estimates)
    return df_estimates

# Prints data based on clustering
def final_graph(x, y, num_clusters, estimates, corresponding):
    num_coordinates = len(x)
    clusters = [[0 for x in range(2 * num_clusters)] for y in range(num_coordinates)] 
    plt.xlim(0, 1.25)
    plt.ylim(0, 1.25)
    plt.grid()
    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]

    for val in range (0, num_coordinates):
        face = int(corresponding[val])
        plt.plot(x[val], y[val], marker="o", markersize=5, markeredgecolor="black", markerfacecolor=colors[face])
    
    print("The graph separated by colors should now print to a file called clustered_geothermal_spots.png")
    plt.savefig('clustered_geothermal_spots.png')
    plt.show()


if __name__ == "__main__":
    # Read useful data into dataframe
    geothermal_spots = pd.read_excel(io='ALL DATA Argonne Geothermal Geochemical Database v2_00.xlsx')

    # Read in variable names from user
    print("Here are your options for variable names:")
    print(geothermal_spots.columns)

    print("Please insert your x variable")
    x_var =  input()
    print("Please insert your y variable")
    y_var =  input()

    # Extract corresponding columns
    x_points = np.array(geothermal_spots[x_var])
    y_points = np.array(geothermal_spots[y_var])

    # Cutting NaN values
    cut_combo = cutting_nan_vals(x_points, y_points)
    new_x = cut_combo[:,0]
    new_y = cut_combo[:,1]

    # Normalize
    norm_combo = normalize(new_x,new_y)
    x_norm = norm_combo[:,0]
    y_norm = norm_combo[:,1]
    num_data_points = len(x_norm)
    x_and_y = np.vstack((x_norm, y_norm)).T
    df_data = pd.DataFrame(x_and_y)

    # Print coordinates
    option_to_print_graph(x_norm, y_norm)


    # CLUSTERING

    # Tracks the changes from this cluster to the next one
    cross_cluster_num_change = int(1)

    # Initializes Initial Class with randomized estimates for a given number of clusters
    CurrentCluster = Clustering_Class.Clustering(0, 0, 0, 0)
    PreviousCluster = Clustering_Class.Clustering(0, 0, 0, 0)

    # Initializes Counter 
    counter = int(1)

    while (cross_cluster_num_change > 0.01):
        # Creates randomized estimates to start off round of optimization with that number of clusters
        pairs, dim = counter, 2
        estimates_x = np.random.rand(pairs,1).T
        estimates_y = np.random.rand(pairs,1).T
        estimates = np.vstack((estimates_x, estimates_y)).T
        df_estimates = pd.DataFrame(estimates)

        # Assigns CurrentCluster (which is the best one yet, to Previous)
        PreviousCluster = CurrentCluster

        # CurrentCluster becomes the first, randomized estimate with counter-many clusters
        CurrentCluster = Clustering_Class.Clustering(num_data_points, counter, df_data, df_estimates)

        # Create temporary cluster that will change each optimization round)
        TempCluster = Clustering_Class.Clustering(num_data_points, counter, df_data, df_estimates)

        # Initialized variable tracking optimization round differences to 1 so that the first optimization happens
        optimization_change_percent = int(1)

        # Counter keeping track of optimization rounds
        rounds_of_optimization = 1

        # Loops through various rounds of optimization
        while (optimization_change_percent > 0.01):
            
            # Calls calculations method from Clustering class to find total distance, average x and y values, etc.
            TempCluster.calculations()

            # Assigns difference between this and last total distance 
            if (rounds_of_optimization == 1):
                optimization_change_percent = 1
            else:
                optimization_change_percent = (CurrentCluster.total_distance - TempCluster.total_distance) / CurrentCluster.total_distance
            
            # CurrentCluster is made to hold this round's results for the next round
            CurrentCluster.reference_points = TempCluster.reference_points
            CurrentCluster.average_x_y_values = TempCluster.average_x_y_values
            CurrentCluster.linked_reference_index = TempCluster.linked_reference_index
            CurrentCluster.total_distance = TempCluster.total_distance

            # We clear TempCluster and it's reference points are now the previous average x and y values
            TempCluster.reference_points = TempCluster.average_x_y_values
            TempCluster.total_distance = int(0)
            TempCluster.average_x_y_values = None
            TempCluster.linked_reference_index = None

            # Increments rounds counter
            rounds_of_optimization = rounds_of_optimization + 1
        
        # Assigns difference between this and last total distance 
        if (counter != 1):
            cross_cluster_num_change = (PreviousCluster.total_distance - CurrentCluster.total_distance) / PreviousCluster.total_distance
        else:
            cross_cluster_num_change = 1
        
        # Increments counter
        counter = counter + 1

    final_graph(x_norm, y_norm, CurrentCluster.num_clusters, CurrentCluster.reference_points, CurrentCluster.linked_reference_index)


    print("TESTING")