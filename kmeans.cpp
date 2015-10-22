
#include <iostream>
#include <vector>
#include "cluster.h"
#include "data.h"

using namespace std;

// K-Means Clustering 
void SimCluster::KMeansClustering() {
	
	float min_distance;                        // Temporary variable holding minimum distance between a data point and a mean
    vector<vector<float> > new_means(centers); // 2D vector with the means to update the <centers> 
    bool not_done = true;                      // When set to false, the clustering has converged and is finished
    bool no_change = false;                    // Boolean to hold whether the means have moved since the last iteration
    float tmp_dist;                            // Temporary distance variable

    while (not_done) {
        // (1) Clear the counts from the last run
        for (unsigned int i = 0; i < now_k; i++) {
            counts[ i ] = 0;
        }

        // (2) Assign each data point to a cluster
        for (unsigned int i = 0; i < DataManager->num_data_pts; i++) {
            // Initially assume the first cluster has the minimum distance to the point
            min_distance = ComputeDistance(DataManager->data[ i ], centers[0]);
            labels[ i ] = 0;
            // Now find the cluster with the true minimum distance to the point
            for (unsigned int j = 1; j < now_k; j++) {
                tmp_dist = ComputeDistance(DataManager->data[ i ], centers[j]);
                if (tmp_dist < min_distance) {
                    min_distance = tmp_dist;
                    labels[ i ] = j; // Assign the data point to belong to cluster j
                }
            }

            // Increase the number of data points with it's closest mean as cluster "labels[ i ]"
            counts[labels[ i ]] += 1;
        }

        // (3) Update means based on the labeling
        for (unsigned int i = 0; i < DataManager->num_data_pts; i++) {
            for (unsigned int j = 0; j < DataManager->dimensions; j++) {
                new_means[labels[ i ]][j] += DataManager->data[ i ][j];
            }
        }
        for (unsigned int i = 0; i < now_k; i++) {
            if (counts[ i ] != 0) {
                for (unsigned int j = 0; j < DataManager->dimensions; j++) {
                    new_means[ i ][j] = new_means[ i ][j] / static_cast<float>(counts[ i ]);
                }
            }
            else {
                if (WARNINGS) cerr << "WARNING: no points for cluster #" << i << endl;
            }
        }

        // (4) Check for convergence (means didn't move)
        no_change = true;
        for (unsigned int i = 0; i < now_k; i++) {
            for (unsigned int j= 0; j < DataManager->dimensions; j++) {
                if (new_means[ i ][j] != centers[ i ][j]) {
                    no_change = false;
                    break;
                }
            }
            if (!no_change) {
                break;
            }
        }

        if (no_change) { // Austin, we have convergence
            not_done = false;
        }
        else {
            centers = new_means;
        }
    } // while (not_done)

} 