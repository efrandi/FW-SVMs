#ifndef STREAM_HANDLER_H_
#define STREAM_HANDLER_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>

#include "SVM-commons.h"
#include "Kernel.h"


#define SIZE_FILE_NAMES 1024
#define INITIAL_ALLOC_ATT 5
#define NEW_ALLOC_ATT 5
#define SHOW_INFO_COUTING false

using namespace std;


//Class Stream_Handler
//handles an stream of data providing interfaces to get new examples 

struct stream_info{
int nr_examples;//total number of examples
int nr_classes;//>=2
bool is_sparse;//sparse(true) or dense(false)
int* class_counters;//number of examples per class
int* labels;
bool set; //true if the info has been already determined
};

void print_stream_info(stream_info* si);


class Stream_Handler{
		
public: 
		
		Stream_Handler();
		Stream_Handler(char* filename);
		~Stream_Handler();
		bool is_open(){	return ifs.is_open(); }		
		bool is_fail(){ return ifs.fail(); }
		bool eof(){ return ifs.eof(); }
		bool open(char* filename);
		bool restart(){ ifs.clear(); ifs.seekg(0, ios::beg); numex_read_=0; return !ifs.fail(); } 
		bool close();
		stream_info* info(){ set_info(); return &sinfo; }
		int numex_read(){ return numex_read_; } 
		svm_problem*  get_next();
		svm_problem*  get_next(int nexamples);
		svm_problem*  get_random(int nexamples);
		svm_problem*  get_balanced(int nex_pclass);
		svm_problem*  load_set(char* filename){
			int dummy = -1;
			return load_set(filename,dummy);
		}
		svm_problem*  load_set(char* filename,int & maxNum);
		
		void destroy_problem(svm_problem* problem);
		bool is_sparse();
		bool permute(); 
		double estimate_gamma(char* filename);
	
		svm_node* temp_x;//for reading
		
protected:

	ifstream ifs;
	char name[SIZE_FILE_NAMES];
	stream_info sinfo;
	int numex_read_;

	int actual_alloc_att;//for reading
	
	void set_info();//read the complete file to count 
	bool determine_if_its_sparse();//true if it's
	void count_dense();//writes on sinfo
	void count_sparse();//writes on sinfo
};

#endif /*STREAM_HANDLER_H_*/
