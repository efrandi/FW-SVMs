#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <vector>

#include "partitioner.h"
#include "random.h"


using namespace std;


//PUBLIC METHODS
//Constructor
    //1. RANDOM OR KMEANS PARTITION
Partition::Partition(int n_frags, int mode, const svm_problem *prob, bool group_classes,bool do_minimal_balancing, int minimum){
		
		set_members(mode,n_frags,do_minimal_balancing, minimum);			
		do_partition(mode,prob,NULL,NULL,NULL,group_classes);
	
					
}

    //2.PREDEFINED PARTITION
Partition::Partition(int n_frags,const svm_problem *prob, int* perm, int* ns){
		
		int mode = PREPART;		
		set_members(mode,n_frags,false,0);				
		do_partition(mode,prob,perm,ns,NULL,false);	
		
}

	//3.WEIGHTED PARTITION
Partition::Partition(int n_frags,const svm_problem *prob, float* weights,bool do_minimal_balancing, int minimum){
		
		int mode = RANDOM_WEIGHTED;		
		set_members(mode,n_frags,do_minimal_balancing,minimum);				
		do_partition(pmode,prob,NULL,NULL,weights,false);
}

    //2.GENERAL
Partition::Partition(int n_frags, int mode, const svm_problem *prob, int* perm, int* ns, float* weights, bool group_classes,bool do_minimal_balancing, int minimum){
		
		set_members(mode,n_frags,do_minimal_balancing,minimum);				
		do_partition(pmode,prob,perm,ns,weights,group_classes);	
		
}

//BASIC SETTING OF STRUCTURES, ALLOCATION, DEFAULT PARAMETERS 
void Partition::set_members(int mode, int n_frags, bool do_minimal_balancing, int minimum){
	
		n = n_frags;
		indexes = new int*[n];
		sizes = new int[n];
		currents = new int[n];
		pmode = mode;
		set_default_params();
		for(int i=0; i < n; i++){
			currents[i]=0;
			sizes[i]=0;
		}
		//set the seed for random numbers generation
		ensure_minimum_by_class = do_minimal_balancing;
		
		if(minimum >= 0){
			minimum_per_class = minimum;
		} else{
			minimum_per_class = 1;
		}
			
		srandom(time(NULL));
	    cout<<"initializated ... begining partitioning\n";	
}

//Destructor
Partition::~Partition(){
		
		for(int i=0; i < n; i++){
			delete[] indexes[i];
		}
		
		delete [] sizes;
		delete [] indexes;
		delete [] currents;
		
} 

//for a given part, returns the index of the item following the last item used   	
//if the index is out of scope or the complete list has been already accessed returns -1
int Partition::get_next(int part){

int curr_index;

	if((0 <= part) && (part < n)){
		curr_index = currents[part]; 
		if(curr_index < sizes[part]){
			currents[part]++;
			return indexes[part][curr_index];
		}
		
	}
	
	return -1;			

}

//for a given part, resets the "last-used-pointer" 
void Partition::rewind(){
	
	for(int i=0; i < n; i++){
			currents[i]=0;
	} 
	
}

//reset all the "last-used-pointers"
int Partition::rewind(int part){
	
	if((0 <= part) && (part < n)){
		currents[part] = 0;
		return 0;
	}	
	
	return -1;
}

//returns the size of a given part
int Partition::get_size(int part){
	
	if((0 <= part) && (part < n)) 
		return sizes[part];
	else
		return -1;			
}

//returns the number of parts
int Partition::get_nparts(){
	return n;	
}

//returns a pointer to the indexes of a given part
int* Partition::get_indexes(int part){
	
	if((0 <= part) && (part < n)) 
		return indexes[part];
	else
		return NULL;		
}



void Partition::set_kmeans_tolerance(double value){

	kmeans_tolerance = value;
}

/////////////////////////////////////////////////////////////////////
//PRIVATE METHODS

//interface to compute the partition with different options
bool Partition::do_partition(int mode, const svm_problem *prob, int* perm, int* ns, float* weights, bool group_classes_in_kmeans){

 int* tempcounts = new int[n]; //temp, to advance through the parts
 int* assignment = new int[prob->l];//for each item, the part at which is assigned 
 grouped_indexes *gitems = NULL;

	  
  if(ensure_minimum_by_class & group_classes_in_kmeans){
  	
  	gitems = new grouped_indexes[1];
  	group_items(prob,gitems);
  	
  }
  
 //For all the partitioning methods 
 //Assignment stores the results of the partition and sizes the sizes
 //these datastructures are already allocated 

 if (mode == RANDOM){
    //random partition
    cout << "partitioning mode is RANDOM\n";
	do_random_assignment(prob->l,assignment,sizes);
 }
 else if(mode == RANDOM_WEIGHTED){
 	//weighted random partition
 	    cout << "partitioning mode is WEIGHTED-RANDOM\n";
	do_weighted_rand_assignment(prob->l,assignment,sizes,weights);
 }
 else if (mode == KMEANS){
 	//partition based on k-means clustering	
 	    cout << "partitioning mode is KMEANS\n";
	do_kmeans_assignment(prob,assignment,sizes,group_classes_in_kmeans,gitems);
 }
 else if (mode == PREPART){
 	//partition is done using a predefined permutation array perm
 	cout << "partitioning mode is PRE-COMPUTED\n";
 	for(int i=0; i < n; i++)
 		 sizes[i] = ns[i];
 	for(int i=0; i < prob->l ; i++)	 
 		 assignment[i] = perm[i]; 
 }

 for(int j=0; j < n ; j++){
		cout << "Part #"<<j<<" has "<<sizes[j]<<" elements.\n";
  }

  vector<int> *balancing_items = new vector<int>[n];	
  
  if(ensure_minimum_by_class){//Minimal-Balancing
  //Ensures at least 'minimum_per_class points' of each class at each part of the partition
   
  	if(!group_classes_in_kmeans){//if the indexes have not been grouped, group them 
  	
  		gitems = new grouped_indexes[1];
  		group_items(prob,gitems);
  		
  	}
  
   //count the number of points of each class in each part
   int **counts_part_classes = new int*[n];
 
   for(int j=0; j < n ; j++){//
		counts_part_classes[j] = new int[gitems->nr_classes];
		for(int i=0; i < gitems->nr_classes; i++)
			counts_part_classes[j][i] = 0; 
   }
  	
   for(int class_it=0; class_it < gitems->nr_classes; class_it++){
 	
    	int current_l = gitems->class_counts[class_it];    
    	int current_start = gitems->starts[class_it];
    			
		for(int i = 0; i < current_l; i++){
			int current_item_idx = gitems->idxs[current_start+i]; 
			int assigned_part = assignment[current_item_idx]; 
			counts_part_classes[assigned_part][class_it]+=1;
		}
	}
	
   cout<<"\nChecking and Correcting Minimum of Items per Class ...\n";
	
   //guaranteeing minimum per class: if at some part and for some class
   //the number of items is under the minimum choose randomly the rest of the points from other parts
   //the resulting partition has now an overlap
   
  for(int j=0; j < n ; j++){
		cout<<"Center N° "<<j<<" :";
		for(int i=0; i < gitems->nr_classes; i++){
			cout<<"class "<<i<<" : "<<counts_part_classes[j][i]<<" ";
			if(counts_part_classes[j][i] < minimum_per_class){ 
				int items_assigned = 0;	
				while(counts_part_classes[j][i] + items_assigned < minimum_per_class){
 						int random_selector = (gitems->starts[i])+(random()%gitems->class_counts[i]);
 						balancing_items[j].push_back(gitems->idxs[random_selector]);
 						items_assigned++; 		
				}	
			}
		}	 
		cout<<"\n";	 
  	}
  	
 cout<<"\n after minimal-balancing ...\n";
 for(int cur_part=0; cur_part < n ; cur_part++){
		cout << "Part #"<<cur_part<<" has "<<sizes[cur_part]+balancing_items[cur_part].size()<<" elements.\n";
 }
 
 }//end minimal-balancing
 
 for(int cur_part=0; cur_part < n ; cur_part++){
		sizes[cur_part] = sizes[cur_part]+balancing_items[cur_part].size();
		indexes[cur_part] = new int[sizes[cur_part]+balancing_items[cur_part].size()]; 
		tempcounts[cur_part] = 0;	
 	}
	
 for(int m=0; m < prob->l ; m++){
		int cur_part = assignment[m]; //sub-array(->part) at which the actual item will be assigned 
		int cur_pos = tempcounts[cur_part];//position of the array in which the item will be stored
		//int cur_pos = part.starts[cur_part] + tempcounts[cur_part];
		tempcounts[cur_part]+=1;
		indexes[cur_part][cur_pos] = m;//the index of the item is stored
    }
    
  for(int cur_part=0; cur_part < n ; cur_part++){
  		for(unsigned int k=0; k < balancing_items[cur_part].size(); k++){
  			int cur_pos = tempcounts[cur_part]; 
  			indexes[cur_part][cur_pos] = balancing_items[cur_part][k];
  			tempcounts[cur_part]+=1;
  		}		
 	}	
 
 
 delete [] balancing_items; 	
 delete [] assignment;
 delete [] tempcounts;	 	

 if(gitems != NULL)
 	destroy_grouped_items(gitems);
 
 cout<<"fin\n";
 
 return true; 
}

void Partition::destroy_grouped_items(grouped_indexes *git){

free(git->class_counts);
free(git->starts);
free(git->idxs);

}
//Group data items according to the labels specified in the prob.y field
//label_ret: label names, start_ret: begin of each class, count_ret: #data of each class, perm: indices to the original data
//perm (length = #data_items) must be allocated before calling this function
void Partition::group_items(const svm_problem *prob, grouped_indexes *items)
{	
	//cout<<"grouping items by class\n";
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	items->class_counts = Malloc(int,max_nr_class);
	items->idxs = Malloc(int,l);
	int *data_label = Malloc(int,l);	
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++items->class_counts[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				items->class_counts = (int *)realloc(items->class_counts,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			items->class_counts[nr_class] = 1;
			++nr_class;
		}
	}
	
	items->starts = Malloc(int,nr_class);
	items->starts[0] = 0;
	for(i=1;i<nr_class;i++)
		items->starts[i] = items->starts[i-1]+items->class_counts[i-1];
	for(i=0;i<l;i++)
	{
		items->idxs[items->starts[data_label[i]]] = i;
		++items->starts[data_label[i]];
	}
	items->starts[0] = 0;
	for(i=1;i<nr_class;i++)
		items->starts[i] = items->starts[i-1]+items->class_counts[i-1];

	free(data_label);
	free(label);
 	items->nr_classes = nr_class;
 	items->total = l;
 	
 	cout<<"Se encontraron "<<nr_class<<" classes\n";
 	
 	for (int s=0; s < nr_class; s++)
 		cout<<"Classe "<<s<<" con "<<items->class_counts[s]<<" elementos\n";
 	
}


//random partition
bool Partition::do_random_assignment(int l, int* assignments, int* counts){

    for(int data_it=0; data_it < l; data_it++){
    	
    		int randomNumber = random();
			int randomPart = randomNumber%n;
			counts[randomPart]++; 
			assignments[data_it] = randomPart; 
    } 	
    
 return true;
 	
}


	
//weighted random partition
bool Partition::do_weighted_rand_assignment(int l, int *assignments, int *counts, float* weights){

 float* cum_weights = new float[n]; 
 cum_weights[0] = weights[0];
 //compute slots sizes in the rulette for each aprt
 for(int parts_it = 1; parts_it < n; parts_it++)
 	cum_weights[parts_it] = cum_weights[parts_it-1] + weights[parts_it];  
 
 
 for(int data_it = 0; data_it < l; data_it++){
 	
 	int randomNumber = random();
 	float random01 =  ((float) (randomNumber%range_rand01) ) / ((float) range_rand01);
	
	//rulette: take a random number and determines in which area it falls
	
	int i;
	for(i = 0; random01 > cum_weights[i] && i < n; i++);
 	
 	//now is all is ok we assign the item to the owner of the slot
 	if(i < n) {
 		assignments[data_it] = i; 
 		counts[i]+=1;
 	
 	} else {//this should not happen
 		
 		assignments[data_it] = 0;
 		counts[0]+=1;
 	}
 }
 
 delete [] cum_weights;	
 return true;
 	
}

bool Partition::do_kmeans_assignment(const svm_problem *prob, int* assignments, int* counts, bool separate_classes, grouped_indexes* items){

 if (items == NULL){
 if (!separate_classes){
 	
 	//k-means on the data without separating by class
 	items = new grouped_indexes[1];		
 	items->total = prob->l;
 	items->nr_classes=1;
 	items->class_counts = new int[1];
 	items->class_counts[0] = prob->l;
 	items->starts = new int[1]; 
 	items->starts[0] = 0;
 	items->idxs = new int[prob->l];
 
 	for(int i=0;i<prob->l;i++){
 	items->idxs[i] = i;  
 	}
 
 } else {
 
   items = new grouped_indexes[1];		
   group_items(prob,items); 
 } 
 }
   
for(int class_it=0; class_it < items->nr_classes; class_it++){
 	
    int current_l = items->class_counts[class_it];    
    int current_start = items->starts[class_it];
    int *current_assignments = new int[current_l]; 
    int *current_counts = new int[n];
    
    svm_node **x = new svm_node*[current_l];
				
	for(int i = 0; i < current_l; i++){
			int current_item_idx = items->idxs[current_start+i]; 
			x[i] = prob->x[current_item_idx];
	}
	
	cout<<"kmeans for class "<<class_it<<" n° items: "<<current_l<<"\n";
	kmeans(x,current_l,n,current_assignments,current_counts,prob->input_dim);
	
	cout<<"kmeans for class "<<class_it<<" finished\n";
		
	for(int i = 0; i < current_l; i++){
			int current_item_idx = items->idxs[current_start+i]; 
			assignments[current_item_idx] = current_assignments[i]; 
	}
	
	for(int j=0; j < n; j++){
	counts[j] += current_counts[j]; 
	}
	delete [] x;
    
 } //end iteration through data items
     
		
 return true;
 
}

//clustering via k-means
//svm_node stores the data, l the number of points
//dim the input dimensionality and k the number of desired clusters
bool Partition::kmeans(svm_node** const x, int l,  int k, int* assignments, int* counts, int dim){
	
	double **centers;
	double *norm_centers;
	double *norm_items;
	centers = new double*[k];
	norm_centers = new double[k];
	norm_items = new double[l];
	
	int max_kmeans_it = 10000;
	
	int center_it; //iterator through the means
	int dim_it; //iterator through the dimensions
	int data_it; //iterator through the data
	
	//compute only once the norms of the items
	//cout<<"computing the norms for the items\n";
	for (data_it = 0; data_it < l; data_it++)
		  norm_items[data_it] = compute_norm_item(x[data_it],dim);
     
	//initialize the centers
	//select a random sample of size k_
	bool *check_selected = new bool[l]; 
	
	for(data_it=0; data_it < l; data_it++)
		check_selected[data_it] = false;  
	
	int current_center = 0;
	
	while(current_center < k){ 	
		
		cout<<"random initialization of means " <<current_center<<"\n";
		
		int random_item = random()%l;
		
		if(check_selected[random_item]==false){
		
			check_selected[random_item]=true;
			
			centers[current_center] = new double[dim];
			
			for(dim_it=0; dim_it < dim; dim_it++)
				centers[current_center][dim_it] = 0; 
			
			norm_centers[current_center] = norm_items[random_item];
			svm_node* selected_item = x[random_item]; 
			
			int dim_count = 0; 
			while(selected_item->index != -1 && dim_count < dim){	
				centers[current_center][selected_item->index-1] = selected_item->value;  
				selected_item++;
				dim_count++;
			}
			current_center++;
		}
	}//end while not all the centers initialized
			 
	double min_distance, tmp_dist, tmp_dot;
    bool not_done = true; //if true we need to continue iterating
    bool no_change = false;  //if true the means didn't change the last step
   	
   	int counter_kmeans_it = 0;
   	
    while (not_done && (counter_kmeans_it < max_kmeans_it)){
    			
		 counter_kmeans_it++;
		 if (counter_kmeans_it%20 == 0 && counter_kmeans_it > 0)
   		  cout<<"Kmeans loop (20-it): "<<counter_kmeans_it<<", clearing the counts\n";
		
        // (1) Clear the counts from the last run
		
        for (center_it = 0; center_it < k; center_it++) {
            counts[center_it] = 0;
         }
   		
   		if (counter_kmeans_it%20 == 0 && counter_kmeans_it > 0)
   			 cout<<"Kmeans loop (20-it): "<<counter_kmeans_it<<", assigning items to centers\n"; 
        
        // (2) Assign each data point to a cluster
        for (data_it = 0; data_it < l; data_it++) {
            // Initially assume the first cluster has the minimum distance to the point
            int default_center = 0;
            tmp_dot = compute_dot(x[data_it], centers[default_center], dim);
            min_distance = norm_centers[default_center] + norm_items[data_it] - 2*tmp_dot;
            assignments[data_it] = default_center;
            
            // Now find the cluster with the true minimum distance to the point
            for (int center_it = 0; center_it < k; center_it++) {
                
                tmp_dot = compute_dot(x[data_it], centers[center_it], dim);
                tmp_dist = norm_centers[center_it] + norm_items[data_it] - 2*tmp_dot;
                
                if (tmp_dist < min_distance) {
                    min_distance = tmp_dist;
                    assignments[data_it] = center_it; // Assign the data point to belong to cluster j
                }
            }
            // Increase the number of data points with it's closest mean
            counts[assignments[data_it]] += 1;
        }

        // (3) Update means based on the labeling and
        // (4) Check for convergence (means didn't move)
        if (counter_kmeans_it%20 == 0 && counter_kmeans_it > 0){
        cout<<"assignments summary:\n";
        for(int j=0;j<k;j++){
         	cout<<"Center "<<j<<" : "<<counts[j]<<" items\n";				
         	}
        cout<<"Kmeans loop (20-it): "<<counter_kmeans_it<<", update centers and check convergence\n"; 
        }
         		

        no_change = update_means(centers,x,assignments,counts,l,k,dim);
        	
        if (no_change) { //we have convergence
            not_done = false;
            //cout<<"Convergence achieved!\n"; 
        } else{
            // for efficiency compute the norm of the means only once
            //cout<<"Convergence NOT achieved. Computing norms\n"; 
        	for (center_it = 0; center_it < k; center_it++) {
        		norm_centers[center_it] = 0;
              for(dim_it = 0; dim_it < dim; dim_it++)
            	norm_centers[center_it] += centers[center_it][dim_it]*centers[center_it][dim_it]; 
         	}   
         	
         	//show assignments 
         	//cout<<"assignments:\n";
         	//for(int i=0; i < l; i++){
         	//	cout<<assignments[i]<<" ";				
         	//	if(i%10==0 && i > 0){
         	//		cout<<"\n";
         	//	}
         	//}
         	cout<<"\n";
        }
        
    } // while (not_done)

for(int j=0; j < k; j++)
	delete [] centers[j]; 

	
delete [] centers;
return counts;

} 

double Partition::compute_distance(const svm_node *point, const double *center, int dim){

 double sum = 0;
 for(int m=0; m < dim; m++){
 		if(point->index == m+1){
 			sum += ((double)point->value - center[m])*((double)point->value - center[m]);
 			point++;
 		}
 		else
	 		sum += center[m]*center[m];
 }

 return sum;

}

double Partition::compute_dot(const svm_node *point, const double *center, int dim){

 double sum = 0;
 int counter = 0;
 
 while(point->index != -1 && counter < dim){	
 	sum +=  (double)((double)point->value)*(center[point->index-1]);
 	point++;
 	counter++;
 }	

 return sum;

}

double Partition::compute_norm_item(const svm_node *point,int dim){

 double sum = 0;
 int counter = 0;
 while(point->index != -1 && counter < dim){	
 	sum +=  (double)((double)point->value)*((double)point->value);
 	point++;
 	counter++;
 }	

 return sum;

}

bool Partition::update_means(double **centers, struct svm_node **x, int *assignments, int *counts, int l, int k, int dim){

 double **previous_centers;
 previous_centers = new double*[k];
 
 for(int j=0; j < k; j++){
		previous_centers[j] = new double[dim];
		for(int m=0; m < dim; m++){
			previous_centers[j][m] = centers[j][m];
			centers[j][m] = 0;  
		}
 }
//cout <<"old means saved\n";
	 
for (int i = 0; i < l; i++){
	svm_node *point = x[i];
	int counter = 0;	 
	while (point->index != -1 && counter < dim){
			centers[assignments[i]][point->index-1] += point->value/counts[assignments[i]];
			point++;
			counter++;
		}
 }
	

for(int j=0; j < k; j++){
		for(int m=0; m < dim; m++){
			if((previous_centers[j][m]+kmeans_tolerance < centers[j][m]) || (previous_centers[j][m]-kmeans_tolerance> centers[j][m])){
			//cout<<"difference old-new for center "<<j<<" (tol="<<kmeans_tolerance<<")\n";
			//cout<<"dimension "<<m<<" old is "<<previous_centers[j][m]<<" new is "<<centers[j][m]<<"\n";
			return false;
			}
		}
 }
		
return true;
}

//set default parameters
void Partition::set_default_params(){
	
	range_rand01 = 30000;
	kmeans_tolerance = pow(10,-10);
	ensure_minimum_by_class = true;
	minimum_per_class = 1;

}