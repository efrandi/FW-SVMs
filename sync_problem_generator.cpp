#include "sync_problem_generator.h"
#define PI 3.14159265

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath> 
#include <time.h>
// include TRNG header files
#include <trng/yarn2.hpp>
#include <trng/uniform_dist.hpp>
#include <trng/normal_dist.hpp>
#include <trng/correlated_normal_dist.hpp>
#include <trng/lcg64.hpp>


sync_data::sync_data(int problem_id_){

	problem_id = problem_id_;
	is_done = false;
	partition = NULL;
	x_space = NULL;
	prob = NULL;
	
}

sync_data::~sync_data(){

	if(is_done){
		delete partition;
		delete [] prob->x;
		delete [] prob->y; 
		delete [] x_space;
	}
}

Partition* sync_data::get_partition(){
	return partition;
}

svm_problem* sync_data::get_problem(){
	
	if(!is_done){
		generate(1800);
	}
	
	return prob;
}

svm_problem* sync_data::generate(const int no_items){
	
	int ncomps; //number of pieces in the problem being simulated
	int noibc; //number of items to generate at each component
	
	if(is_done){
		
		//clean previous simulation	
		delete partition;
		delete [] prob->x;
		delete [] prob->y; 
		delete [] x_space;
		is_done = !is_done;
		partition = NULL;
		x_space = NULL;
		prob = NULL;
	}
		
	printf("SIMULATOR: generating data for id %d...\n",problem_id);
	
	switch(problem_id){
		
		case RECTANGLES:
			ncomps = 2*3; //2 classes, 3 rectangles for each class
			noibc = (int)floor(no_items/ncomps);
			do_rectangles(noibc,noibc,noibc,noibc,noibc,noibc);
			break;
			
		case GAUSSIANS_MIX:
			ncomps = 2*4; //2 classes, 4 gaussians for each class
			noibc = (int)floor(no_items/ncomps);
			do_mix_gaussians(noibc,noibc,noibc,noibc,noibc,noibc,noibc,noibc);
			break;
			
	} 
	
	if (prob == NULL || partition == NULL){
		std::cout<<"error: imposible to create synthetic problem, prob is NULL or partition failed"<<std::endl;
		exit(1);
	}
	
	is_done = true;
	return prob; 
}

void sync_data::plot(){
	
	if(!is_done){
		generate(1800);
	}
	
	switch(problem_id){
		
		case RECTANGLES:
			plot_rectangles();
			break;	
		case GAUSSIANS_MIX:
			plot_gaussians();
			break;
	}
	
}


void sync_data::do_mix_gaussians(int ng1,int ng2,int ng3,int ng4,int ng5,int ng6,int ng7,int ng8){

	const int d= 2;
	
	//Gaussian B1 Center Class #2 
	double mean1[d] = {0.1, 0.1};
	double angle1 = 0.0;                     		
	double cov_matrix1[d][d] = {{ 0.025,  0.000 },
	                     		{ 0.000,  0.015 }};                   
	
	//Gaussian B2 Upper-Left Class #2 
	double mean2[d] = {0.4, 0.4};
	double angle2 = 0.0;                     		
	double cov_matrix2[d][d] = {{ 0.025,  0.000 },
	                     		{ 0.000,  0.015 }};                   
	
	//Gaussian B3 Center-Left Class #2 
	double mean3[d] = {0.6, 0.1};
	double angle3 = 0.0;                     		
	double cov_matrix3[d][d] = {{ 0.005,  0.000 },
	                     		{ 0.000,  0.005 }};                   
	
	//Gaussian B4 Lower-Left Class #2 
	double mean4[d] = {0.7,-0.5};
	double angle4 = PI/4.0;                     		
	double cov_matrix4[d][d] = {{ 0.025,  0.000 },
	                     		{ 0.000,  0.005 }};                   
	
	//Gaussian A1 Upper-Most-Left Class #1 
	double mean5[d] = {0.8, 0.5};
	double angle5 = -PI/3.0;                     		
	double cov_matrix5[d][d] = {{ 0.030,  0.000 },
	                     		{ 0.000,  0.010 }};                   
	
	//Gaussian A2 Upper-Right Class #1
	double mean6[d] = {-0.2, 0.5};
	double angle6 = 0.0;                     		
	double cov_matrix6[d][d] = {{ 0.030,  0.000 },
	                     		{ 0.000,  0.020 }};                   
	
	//Gaussian A3 Center-Right Class #1 
	double mean7[d] = {-0.4, 0.0};
	double angle7 = 0.0;                     		
	double cov_matrix7[d][d] = {{ 0.030,  0.000 },
	                     		{ 0.000,  0.025 }};                   
	
	//Gaussian A4 Lower-A-Bit-Left Class #1 
	double mean8[d] = {0.1,-0.4};
	double angle8 = -PI/4;                     		
	double cov_matrix8[d][d] = {{ 0.020,  0.000 },
	                     		{ 0.000,  0.020 }};                   
	
	//Summary of Gaussians
	//1: Class #2 Center, in picture B1
	//2: Class #2 Upper-Left, in picture B2
	//3: Class #2 Center-Left, in picture B3
	//4: Class #2 Lower-Left, in picture B4
	//5: Class #1 Upper-Most-Left, in picture A1
	//6: Class #1 Upper-Right, in picture A2
	//7: Class #1 Center-Right in picture A3
	//8: Class #1 Lower-A-Bit-Left, in picture A4
	 
	
	//Distribution: 4 nodes
	//Node #1: A1-B1
	//Node #2: A2-B2
	//Node #3: A3-B3
	//Node #4: A4-B4
	
	int label_g1 = 2; int node_g1 = 0; //B1
	int label_g2 = 2; int node_g2 = 1; //B2
	int label_g3 = 2; int node_g3 = 2; //B3
	int label_g4 = 2; int node_g4 = 3; //B4
	int label_g5 = 1; int node_g5 = 0; //A1
	int label_g6 = 1; int node_g6 = 1; //A2
	int label_g7 = 1; int node_g7 = 2; //A3
	int label_g8 = 1; int node_g8 = 3; //A4

	const int no_fragments = 4;
	int ns[no_fragments];
	ns[0] = ng1 + ng5;
	ns[1] = ng2 + ng6;
	ns[2] = ng3 + ng7;	
	ns[3] = ng4 + ng8;
	
	printf("GAUSSIANS_SIMULATOR: creating distributions ...\n");
	
	trng::correlated_normal_dist<> gaussian1(&cov_matrix1[0][0], &cov_matrix1[d-1][d-1]+1);
	trng::correlated_normal_dist<> gaussian2(&cov_matrix2[0][0], &cov_matrix2[d-1][d-1]+1);
	trng::correlated_normal_dist<> gaussian3(&cov_matrix3[0][0], &cov_matrix3[d-1][d-1]+1);
	trng::correlated_normal_dist<> gaussian4(&cov_matrix4[0][0], &cov_matrix4[d-1][d-1]+1);
	trng::correlated_normal_dist<> gaussian5(&cov_matrix5[0][0], &cov_matrix5[d-1][d-1]+1);
	trng::correlated_normal_dist<> gaussian6(&cov_matrix6[0][0], &cov_matrix6[d-1][d-1]+1);
	trng::correlated_normal_dist<> gaussian7(&cov_matrix7[0][0], &cov_matrix7[d-1][d-1]+1);
	trng::correlated_normal_dist<> gaussian8(&cov_matrix8[0][0], &cov_matrix8[d-1][d-1]+1);

	using trng::lcg64;
	lcg64 Rgen; Rgen.seed((long unsigned int) time(NULL));
	
	prob = new svm_problem [1];
	prob->l = ng1 + ng2 + ng3 + ng4 + ng5 + ng6 + ng7 + ng8;
	prob->x = new svm_node* [prob->l];
	prob->y = new double [prob->l];
	x_space = new svm_node [(d+1)*prob->l];
	
	int k;
	int count = 0;
	int count_items = 0; 
	int* assignments = new int [prob->l];
	double cx_0, cy_0, cx_p, cy_p;
	
	for(k=0; k < ng1; k++){
		
		//generate ~U coordinates
		cx_0 = gaussian1(Rgen);
		cy_0 = gaussian1(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle1) - cy_0*sin(angle1) + mean1[0];
		cy_p = cx_0*sin(angle1) + cy_0*cos(angle1) + mean1[1];
		
		assignments[count_items] = node_g1;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_g1;
		
	}
	
	
	printf("GAUSSIANS-GENERATOR: gaussian B1 created ... \n");
	
	for(k=0; k < ng2; k++){
		
		//generate ~U coordinates
		cx_0 = gaussian2(Rgen);
		cy_0 = gaussian2(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle2) - cy_0*sin(angle2) + mean2[0];
		cy_p = cx_0*sin(angle2) + cy_0*cos(angle2) + mean2[1];
		
		assignments[count_items] = node_g2;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_g2;
		
	}
	
	printf("GAUSSIANS-GENERATOR: gaussian B2 created ... \n");

	for(k=0; k < ng3; k++){
		
		//generate ~U coordinates
		cx_0 = gaussian3(Rgen);
		cy_0 = gaussian3(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle3) - cy_0*sin(angle3) + mean3[0];
		cy_p = cx_0*sin(angle3) + cy_0*cos(angle3) + mean3[1];
		
		assignments[count_items] = node_g3;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_g3;
		
	}
	
	printf("GAUSSIANS-GENERATOR: gaussian B3 created ... \n");
	
	for(k=0; k < ng4; k++){
		
		//generate ~U coordinates
		cx_0 = gaussian4(Rgen);
		cy_0 = gaussian4(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle4) - cy_0*sin(angle4) + mean4[0];
		cy_p = cx_0*sin(angle4) + cy_0*cos(angle4) + mean4[1];
		
		assignments[count_items] = node_g4;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_g4;
		
	}
	
	printf("GAUSSIANS_SIMULATOR: gaussian B4 created ... \n");
	
	for(k=0; k < ng5; k++){
		
		//generate ~U coordinates
		cx_0 = gaussian5(Rgen);
		cy_0 = gaussian5(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle5) - cy_0*sin(angle5) + mean5[0];
		cy_p = cx_0*sin(angle5) + cy_0*cos(angle5) + mean5[1];
		
		assignments[count_items] = node_g5;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_g5;
		
	}
	
	printf("GAUSSIANS-GENERATOR: gaussian A1 created ... \n");
	
	for(k=0; k < ng6; k++){
		
		//generate ~U coordinates
		cx_0 = gaussian6(Rgen);
		cy_0 = gaussian6(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle6) - cy_0*sin(angle6) + mean6[0];
		cy_p = cx_0*sin(angle6) + cy_0*cos(angle6) + mean6[1];
		
		assignments[count_items] = node_g6;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_g6;
		
	}
	
	printf("GAUSSIANS-GENERATOR: gaussian A2 created ... \n");
	
	for(k=0; k < ng7; k++){
		
		//generate ~U coordinates
		cx_0 = gaussian7(Rgen);
		cy_0 = gaussian7(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle7) - cy_0*sin(angle7) + mean7[0];
		cy_p = cx_0*sin(angle7) + cy_0*cos(angle7) + mean7[1];
		
		assignments[count_items] = node_g7;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_g7;
		
	}
	
	printf("GAUSSIANS-GENERATOR: gaussian A3 created ... \n");
	
	for(k=0; k < ng8; k++){
		
		//generate ~U coordinates
		cx_0 = gaussian8(Rgen);
		cy_0 = gaussian8(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle8) - cy_0*sin(angle8) + mean8[0];
		cy_p = cx_0*sin(angle8) + cy_0*cos(angle8) + mean8[1];
		
		assignments[count_items] = node_g8;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_g8;
		
	}
	
	printf("GAUSSIANS-GENERATOR: gaussian A4 created ... \n");
	
	partition = new Partition(no_fragments,prob,assignments,ns);
	
	delete [] assignments;
	
}


void sync_data::do_rectangles(int nr1,int nr2,int nr3,int nr4,int nr5,int nr6){
	
	//parameters rectangle 1 Upper-Left
	double c1_x = -0.5; double c1_y = 1.5;
	double l1_x = 2.5; double l1_y = 1.0;
	double angle1 = PI/4; 

	
	//parameters rectangle 2 Center-Left
	double c2_x = -1.5; double c2_y = 0.0;
	double l2_x = 2.0; double l2_y = 1.0;
	double angle2 = 0.0; 
	
	//parameters rectangle 3 Lower-Left
	double c3_x = -0.5; double c3_y = -1.5;
	double l3_x = 2.5; double l3_y = 1.0;
	double angle3 = 7*PI/4; 
	
	//parameters rectangle 4 Upper-Right
	double c4_x = 1.0; double c4_y = 1.25;
	double l4_x = 2.5; double l4_y = 1.0;
	double angle4 = PI/4; 
	
	//parameters rectangle 5 Center-Right
	double c5_x = 1.0; double c5_y = 0.0;
	double l5_x = 2.0; double l5_y = 1.0;
	double angle5 = 0.0; 
	
	//parameters rectangle 6
	double c6_x = 1.0; double c6_y = -1.25;
	double l6_x = 2.5; double l6_y = 1.0;
	double angle6 = -PI/4; 
	
	
	printf("RECTANGLES_SIMULATOR: creating distributions ...\n");
	
	trng::uniform_dist<double> rectangle1_x(-1.0*l1_x/2.0,l1_x/2.0);
	trng::uniform_dist<double> rectangle1_y(-1.0*l1_y/2.0,l1_y/2.0);	
	trng::uniform_dist<double> rectangle2_x(-1.0*l2_x/2.0,l2_x/2.0);
	trng::uniform_dist<double> rectangle2_y(-1.0*l2_y/2.0,l2_y/2.0);
	trng::uniform_dist<double> rectangle3_x(-1.0*l3_x/2.0,l3_x/2.0);
	trng::uniform_dist<double> rectangle3_y(-1.0*l3_y/2.0,l3_y/2.0);
	trng::uniform_dist<double> rectangle4_x(-1.0*l4_x/2.0,l4_x/2.0);
	trng::uniform_dist<double> rectangle4_y(-1.0*l4_y/2.0,l4_y/2.0);
	trng::uniform_dist<double> rectangle5_x(-1.0*l5_x/2.0,l5_x/2.0);
	trng::uniform_dist<double> rectangle5_y(-1.0*l5_y/2.0,l5_y/2.0);
	trng::uniform_dist<double> rectangle6_x(-1.0*l6_x/2.0,l6_x/2.0);
	trng::uniform_dist<double> rectangle6_y(-1.0*l6_y/2.0,l6_y/2.0);
	
	using trng::yarn2;
	yarn2 Rgen; Rgen.seed(time(NULL));
	
	int label_r1 = 1; int node_r1 = 0;//upper-left  - >node 0
	int label_r2 = 1; int node_r2 = 1;//center-left  ->node 1
	int label_r3 = 1; int node_r3 = 2;//lower-left   ->node 2
	int label_r4 = 2; int node_r4 = 2;//upper-right  ->node 2
	int label_r5 = 2; int node_r5 = 1;//center-right ->node 1
	int label_r6 = 2; int node_r6 = 0;//lower-right  ->node 0
	
	const int no_fragments = 3;
	int ns[no_fragments];
	ns[0] = nr1 + nr6;
	ns[1] = nr2 + nr5;
	ns[2] = nr3 + nr4;	
	
	prob = new svm_problem [1];
	prob->l = nr1 + nr2 + nr3 + nr4 + nr5 + nr6;
	prob->x = new svm_node* [prob->l];
	prob->y = new double [prob->l];
	int dim = 2;
	x_space = new svm_node [(dim+1)*prob->l];
	
	int k;
	int count = 0;
	int count_items = 0; 
	int* assignments = new int [prob->l];
	double cx_0, cy_0, cx_p, cy_p;
	
	for(k=0; k < nr1; k++){
		
		//generate ~U coordinates
		cx_0 = rectangle1_x(Rgen);
		cy_0 = rectangle1_y(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle1) - cy_0*sin(angle1) + c1_x;
		cy_p = cx_0*sin(angle1) + cy_0*cos(angle1) + c1_y;
		
		assignments[count_items] = node_r1;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_r1;
		
	}
	
	printf("RECTANGLES_SIMULATOR: rectangle 1 created ... \n");
	
	for(k=0; k < nr2; k++){
		
		//generate ~U coordinates
		cx_0 = rectangle2_x(Rgen);
		cy_0 = rectangle2_y(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle2) - cy_0*sin(angle2) + c2_x;
		cy_p = cx_0*sin(angle2) + cy_0*cos(angle2) + c2_y;
		
		assignments[count_items] = node_r2;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_r2;
		
	}
	
	printf("RECTANGLES_SIMULATOR: rectangle 2 created ... \n");
		
	for(k=0; k < nr3; k++){
		
		//generate ~U coordinates
		cx_0 = rectangle3_x(Rgen);
		cy_0 = rectangle3_y(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle3) - cy_0*sin(angle3) + c3_x;
		cy_p = cx_0*sin(angle3) + cy_0*cos(angle3) + c3_y;
		
		assignments[count_items] = node_r3;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_r3;
		
	}
	
	printf("RECTANGLES_SIMULATOR: rectangle 3 created ... \n");
	
	for(k=0; k < nr4; k++){
		
		//generate ~U coordinates
		cx_0 = rectangle4_x(Rgen);
		cy_0 = rectangle4_y(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle4) - cy_0*sin(angle4) + c4_x;
		cy_p = cx_0*sin(angle4) + cy_0*cos(angle4) + c4_y;
		
		assignments[count_items] = node_r4;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_r4;
		
	}
	
	printf("RECTANGLES_SIMULATOR: rectangle 4 created ... \n");
	
	for(k=0; k < nr5; k++){
		
		//generate ~U coordinates
		cx_0 = rectangle5_x(Rgen);
		cy_0 = rectangle5_y(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle5) - cy_0*sin(angle5) + c5_x;
		cy_p = cx_0*sin(angle5) + cy_0*cos(angle5) + c5_y;
		
		assignments[count_items] = node_r5;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_r5;
		
	}
	
	printf("RECTANGLES_SIMULATOR: rectangle 5 created ... \n");
	
	for(k=0; k < nr6; k++){
		
		//generate ~U coordinates
		cx_0 = rectangle6_x(Rgen);
		cy_0 = rectangle6_y(Rgen);
		//rotate by angle
		cx_p = cx_0*cos(angle6) - cy_0*sin(angle6) + c6_x;
		cy_p = cx_0*sin(angle6) + cy_0*cos(angle6) + c6_y;
		
		assignments[count_items] = node_r6;
		
		x_space[count].index = 1;
		x_space[count].value = cx_p;
		
		prob->x[count_items] = &x_space[count++]; 
		
		x_space[count].index = 2;
		x_space[count++].value = cy_p;
		
		x_space[count].index = -1;
		x_space[count++].value = 0;
		
		prob->y[count_items++] = label_r6;
		
	}
	
	printf("RECTANGLES_SIMULATOR: rectangle 6 created ... \n");
	

	 
	partition = new Partition(no_fragments,prob,assignments,ns);
	
	delete [] assignments;
}


void sync_data::plot_rectangles(){

		int stamp;
		srand(time(NULL));
		stamp = (int) rand()%1000000;
	
		char tempfile_node0[FILENAME_LEN];
		char tempfile_node1[FILENAME_LEN];
		char tempfile_node2[FILENAME_LEN];
		
		sprintf(tempfile_node0,"RECTANGLES_DATA_NODE0.%d.txt",stamp);
		sprintf(tempfile_node1,"RECTANGLES_DATA_NODE1.%d.txt",stamp);
		sprintf(tempfile_node2,"RECTANGLES_DATA_NODE2.%d.txt",stamp);
		
		std::ofstream data_node0(tempfile_node0,std::ios_base::out);
		std::ofstream data_node1(tempfile_node1,std::ios_base::out);
		std::ofstream data_node2(tempfile_node2,std::ios_base::out);
		
		partition->rewind();
		svm_node* current_x;
		int current_idx;
		using std::endl;
		
		//write data rectangles node 0
		for(int i=0; i < partition->get_size(0); i++){
			current_idx = partition->get_next(0);		
			current_x = prob->x[current_idx];
			while(current_x->index != -1){
				data_node0<<current_x->value<<" ";
				current_x++;
			}
			data_node0<<prob->y[current_idx]<<endl;
		}
		
		data_node0.close();
		
		//write data rectangles node 1
		for(int i=0; i < partition->get_size(1); i++){
			current_idx = partition->get_next(1);		
			current_x = prob->x[current_idx];
			while(current_x->index != -1){
				data_node1<<current_x->value<<" ";
				current_x++;
			}
			data_node1<<prob->y[current_idx]<<endl;
		}
		
		data_node1.close();
		
		//write data rectangles node 2
		for(int i=0; i < partition->get_size(2); i++){
			current_idx = partition->get_next(2);		
			current_x = prob->x[current_idx];
			while(current_x->index != -1){
				data_node2<<current_x->value<<" ";
				current_x++;
			}
			data_node2<<prob->y[current_idx]<<endl;
		}
		
		data_node2.close();
		
		int marker_class2 = 4; 
		int marker_class1 = 7; 
		int label_class2 = 2;
		int label_class1 = 1;
		//1=+, 2=X, 3=*, 4=square, 5=filled square, 6=circle,
#       //7=filled circle, 8=triangle, 9=filled triangle, etc.

		
		int color_node0 =  1;
		int color_node1 = -1;
		int color_node2 =  2;
		//-1=black 1=red 2=grn 3=blue 4=purple 5=aqua 6=brn 7=orange 8=light-brn
		
		FILE* gnuplotpipe=popen("gnuplot -persist","w");
		char command[2048];
		fprintf(gnuplotpipe,"unset key\n");
		fprintf(gnuplotpipe,"set multiplot\n");
		fprintf(gnuplotpipe,"set xrange [-2:2]\n");
		fprintf(gnuplotpipe,"set yrange [-2:2]\n");
			
		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node0, label_class2, marker_class2, color_node0);
		fprintf(gnuplotpipe,"%s\n",command);	
		
		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node0, label_class1, marker_class1, color_node0);
		fprintf(gnuplotpipe,"%s\n",command);	
		
		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node1, label_class2, marker_class2, color_node1);
		fprintf(gnuplotpipe,"%s\n",command);	
		
		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node1, label_class1, marker_class1, color_node1);
		fprintf(gnuplotpipe,"%s\n",command);	
		
		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node2, label_class2, marker_class2, color_node2);
		fprintf(gnuplotpipe,"%s\n",command);	
		
		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node2, label_class1, marker_class1, color_node2);
		fprintf(gnuplotpipe,"%s\n",command);	
		
		fprintf(gnuplotpipe,"unset multiplot\n");
		
		pclose(gnuplotpipe);
		
		
		int r0 = remove(tempfile_node0);
		if(r0 != 0)
			printf("RECTANGLES_PLOT: It was not possible to remove temp file: %s",tempfile_node0);
		
		int r1 = remove(tempfile_node1);
		if(r1 != 0)
			printf("RECTANGLES_PLOT: It was not possible to remove temp file: %s",tempfile_node1);
		
		int r2 = remove(tempfile_node2);
		if(r2 != 0)
			printf("RECTANGLES_PLOT: It was not possible to remove temp file: %s",tempfile_node2);
	
		partition->rewind();
		
}

void sync_data::plot_gaussians(){

		int stamp;
		srand(time(NULL));
		stamp = (int) rand()%1000000;
		
		char tempfile_node0[FILENAME_LEN];
		char tempfile_node1[FILENAME_LEN];
		char tempfile_node2[FILENAME_LEN];
		char tempfile_node3[FILENAME_LEN];
		
		sprintf(tempfile_node0,"GAUSSIANS_DATA_NODE0.%d.txt",stamp);
		sprintf(tempfile_node1,"GAUSSIANS_DATA_NODE1.%d.txt",stamp);
		sprintf(tempfile_node2,"GAUSSIANS_DATA_NODE2.%d.txt",stamp);
		sprintf(tempfile_node3,"GAUSSIANS_DATA_NODE3.%d.txt",stamp);
		
		
		const int no_nodes = partition->get_nparts();

		std::ofstream data_node0(tempfile_node0,std::ios_base::out);
		std::ofstream data_node1(tempfile_node1,std::ios_base::out);
		std::ofstream data_node2(tempfile_node2,std::ios_base::out);
		std::ofstream data_node3(tempfile_node3,std::ios_base::out);
		
				
		partition->rewind();
		svm_node* current_x;
		int current_idx;
		using std::endl;
		
		for(int i=0; i < partition->get_size(0); i++){
			current_idx = partition->get_next(0);		
			current_x = prob->x[current_idx];
			while(current_x->index != -1){
				data_node0<<current_x->value<<" ";
				current_x++;
			}
			data_node0<<prob->y[current_idx]<<endl;
		}
		
		data_node0.close();
		
		for(int i=0; i < partition->get_size(1); i++){
			current_idx = partition->get_next(1);		
			current_x = prob->x[current_idx];
			while(current_x->index != -1){
				data_node1<<current_x->value<<" ";
				current_x++;
			}
			data_node1<<prob->y[current_idx]<<endl;
		}
		
		data_node1.close();
		
		for(int i=0; i < partition->get_size(2); i++){
			current_idx = partition->get_next(2);		
			current_x = prob->x[current_idx];
			while(current_x->index != -1){
				data_node2<<current_x->value<<" ";
				current_x++;
			}
			data_node2<<prob->y[current_idx]<<endl;
		}
		
		data_node2.close();
		
		for(int i=0; i < partition->get_size(3); i++){
			current_idx = partition->get_next(3);		
			current_x = prob->x[current_idx];
			while(current_x->index != -1){
				data_node3<<current_x->value<<" ";
				current_x++;
			}
			data_node3<<prob->y[current_idx]<<endl;
		}
		
		data_node3.close();
				
		const int no_classes = 2;
		int markers[no_classes] = {4, 7};
		int labels[no_classes] = {2, 1};
		//1=+, 2=X, 3=*, 4=square, 5=filled square, 6=circle,
        //7=filled circle, 8=triangle, 9=filled triangle, etc.

		
		int* colors =  new int [no_nodes];
		colors[0] =   1;
		colors[1] =  -1;
		colors[2] =   2;
		colors[3] =   6;
		
		//-1=black 1=red 2=grn 3=blue 4=purple 5=aqua 6=brn 7=orange 8=light-brn
		
		FILE* gnuplotpipe=popen("gnuplot -persist","w");
		char command[2048];
		fprintf(gnuplotpipe,"unset key\n");
		fprintf(gnuplotpipe,"set multiplot\n");
		fprintf(gnuplotpipe,"set xrange [-1:1]\n");
		fprintf(gnuplotpipe,"set yrange [-1:1]\n");
				
		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node0, labels[0],markers[0], colors[0]);
		fprintf(gnuplotpipe,"%s\n",command);	
		
		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node0, labels[1],markers[1], colors[0]);
		fprintf(gnuplotpipe,"%s\n",command);	
		
		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node1, labels[0],markers[0], colors[1]);
		fprintf(gnuplotpipe,"%s\n",command);	
		
		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node1, labels[1],markers[1], colors[1]);
		fprintf(gnuplotpipe,"%s\n",command);	
		
		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node2, labels[0],markers[0], colors[2]);
		fprintf(gnuplotpipe,"%s\n",command);	
		
		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node2, labels[1],markers[1], colors[2]);
		fprintf(gnuplotpipe,"%s\n",command);	

		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node3, labels[0],markers[0], colors[3]);
		fprintf(gnuplotpipe,"%s\n",command);	
		
		sprintf(command,"plot \"%s\" using ($1):($3==%d ? $2 : 1/0) pt %d lt %d",tempfile_node3, labels[1],markers[1], colors[3]);
		fprintf(gnuplotpipe,"%s\n",command);
				
		fprintf(gnuplotpipe,"unset multiplot\n");
		
		pclose(gnuplotpipe);
		
		
		int r0 = remove(tempfile_node0);
		if(r0 != 0)
			printf("GAUSSIANS_PLOT: It was not possible to remove temp file: %s",tempfile_node0);
		
		int r1 = remove(tempfile_node1);
		if(r1 != 0)
			printf("GAUSSIANS_PLOT: It was not possible to remove temp file: %s",tempfile_node1);
		
		int r2 = remove(tempfile_node2);
		if(r2 != 0)
			printf("GAUSSIANS_PLOT: It was not possible to remove temp file: %s",tempfile_node2);
	
		int r3 = remove(tempfile_node3);
		if(r3 != 0)
			printf("GAUSSIANS_PLOT: It was not possible to remove temp file: %s",tempfile_node3);
	
		partition->rewind();
		
}