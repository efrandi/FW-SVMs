#include "Stream_Handler.h"

Stream_Handler::Stream_Handler(): ifs() { 

	actual_alloc_att = INITIAL_ALLOC_ATT;
	temp_x =  Malloc(svm_node,actual_alloc_att*sizeof(struct svm_node));
	sinfo.set = false;
	numex_read_=0;
}

Stream_Handler::Stream_Handler(char* filename): ifs(filename) { 
	
	strcpy(name,filename); 
	actual_alloc_att = INITIAL_ALLOC_ATT;
	temp_x = Malloc(svm_node,actual_alloc_att*sizeof(struct svm_node));
	sinfo.set = false;
	numex_read_=0;

}
		
Stream_Handler::~Stream_Handler(){ 
	
	//free(temp_x);
	
//	if(sinfo.set){
//	
//		delete [] sinfo.class_counters;
//		delete [] sinfo.labels;
//	
//	}
	

}
		
		
svm_problem*  Stream_Handler::get_next(){
	
	if(!ifs.is_open() || ifs.eof())
		return NULL;
	
	svm_problem* new_pair = new svm_problem [1];
	new_pair->x = new svm_node* [1];
	new_pair->y = new double [1];
	new_pair->l = 1;
	
	string line;
	getline(ifs,line);	
	string::size_type pos_found, previous_pos;
	
	//int actual_alloc_att = INITIAL_ALLOC_X;
	//new_pair->x[0] = (struct svm_node *) malloc(svm_node,INITIAL_ALLOC_X*sizeof(struct svm_node);
	
	if(is_sparse()){
		
		int attributes = 0;
		int index;
		double label,value;
		
		pos_found = line.find_first_of(" ",0);
		label = atof(line.substr(0,pos_found).c_str()); 
		previous_pos = pos_found;
		pos_found = line.find_first_of(":",previous_pos);
    		
		while((pos_found < line.size()) && (previous_pos < line.size())) {
			
			index = atoi(line.substr(previous_pos+1,pos_found-previous_pos-1).c_str()); 
    		previous_pos = pos_found;
    		pos_found = line.find_first_of(" ",previous_pos+1);
    		if(pos_found >= line.size())
    			pos_found = line.size()-1;
    		value = atof(line.substr(previous_pos+1,pos_found-previous_pos-1).c_str()); 
    		previous_pos = pos_found;
    		pos_found = line.find_first_of(":",previous_pos);
    		
    		
    		if(attributes >= actual_alloc_att){
    			actual_alloc_att = actual_alloc_att + NEW_ALLOC_ATT;
    			temp_x = (struct svm_node *) realloc(temp_x,actual_alloc_att*sizeof(struct svm_node));
    			
    		}
    		
    		temp_x[attributes].index = index;
    		temp_x[attributes].value = value; 	
    		attributes++;
    		
  		}
  		
  		new_pair->x[0] = new svm_node [attributes+1];
  		
  		for(int i=0;i<attributes;i++){
  			new_pair->x[0][i].value = temp_x[i].value;
  			new_pair->x[0][i].index = temp_x[i].index;
  			//cout<<"written: "<<new_pair->x[0][i].index<<" : "<<new_pair->x[0][i].value<<endl;
    	
  		}
  		new_pair->x[0][attributes].value = 0;
  		new_pair->x[0][attributes].index = -1;		
		new_pair->y[0] = (double)label;
		
	} else {//dense
	
	string::size_type last_pos;
	int attributes = 0;
	int label;
	double value;
	
	last_pos = line.find_last_of(",");
	label = atoi(line.substr(last_pos+1,line.size()-last_pos-1).c_str()); 
	previous_pos = 0;
	pos_found = line.find_first_of(",",0);
		
	while(pos_found != last_pos) {
    	
    	if(attributes >= actual_alloc_att){
    			actual_alloc_att = actual_alloc_att + NEW_ALLOC_ATT;
    			temp_x = (struct svm_node *) realloc(temp_x,actual_alloc_att*sizeof(struct svm_node));
    		
    	}
    			
		value = atof(line.substr(previous_pos,pos_found-previous_pos).c_str()); 
		previous_pos = pos_found+1;	
		pos_found = line.find_first_of(",",previous_pos);
    	temp_x[attributes].value = value;
    	temp_x[attributes].index = attributes+1;
  		attributes++;
	}
	
	new_pair->x[0] = new svm_node [attributes+1];
  		
  	for(int i=0;i<attributes;i++){
  		
  			new_pair->x[0][i].value = temp_x[i].value;
  			new_pair->x[0][i].index = temp_x[i].index;
  			//cout<<"written: "<<new_pair->x[0][i].index<<" : "<<new_pair->x[0][i].value<<endl;
    	
  	}
  	
  	new_pair->x[0][attributes].value = 0;
  	new_pair->x[0][attributes].index = -1;		
	new_pair->y[0] = (double)label;
		
	}	
	numex_read_++;
	return new_pair;
		
}

svm_problem* Stream_Handler::get_next(int nexamples){
	
	if(!ifs.is_open() || ifs.eof()){
		return NULL;	
	}
	
	int nread = 0;
	svm_problem* new_examples = new svm_problem [1];
	new_examples->x = new svm_node* [nexamples];
	new_examples->y = new double [nexamples];
	
	while(ifs.is_open() && !ifs.eof() && (nread < nexamples)){
	
	string line;
	getline(ifs,line);	
	string::size_type pos_found, previous_pos;
	
	if(is_sparse()){
			
		int attributes = 0;
		int index;
		double label,value;
		
		pos_found = line.find_first_of(" ",0);
		label = atof(line.substr(0,pos_found).c_str()); 
		previous_pos = pos_found;
		pos_found = line.find_first_of(":",previous_pos);
    		
		while((pos_found < line.size()) && (previous_pos < line.size())) {
			
		index = atoi(line.substr(previous_pos+1,pos_found-previous_pos-1).c_str()); 
    		previous_pos = pos_found;
    		pos_found = line.find_first_of(" ",previous_pos+1);
    		if(pos_found >= line.size())
    			pos_found = line.size()-1;
    		value = atof(line.substr(previous_pos+1,pos_found-previous_pos-1).c_str()); 
    		previous_pos = pos_found;
    		pos_found = line.find_first_of(":",previous_pos);
    		
    		
    		if(attributes >= actual_alloc_att){
    			actual_alloc_att = actual_alloc_att + NEW_ALLOC_ATT;
    			temp_x = (struct svm_node *) realloc(temp_x,actual_alloc_att*sizeof(struct svm_node));
    			
    		}
    		
    		temp_x[attributes].index = index;
    		temp_x[attributes].value = value; 	
    		attributes++;
    		
  		}
  		
  		new_examples->x[nread] = new svm_node [attributes+1];
  		
  		for(int i=0;i<attributes;i++){
  			new_examples->x[nread][i].value = temp_x[i].value;
  			new_examples->x[nread][i].index = temp_x[i].index;
  			//cout<<"written: "<<new_examples->x[nread][i].index<<" : "<<new_examples->x[0][i].value<<endl;
    	
  		}
  		new_examples->x[nread][attributes].value = 0;
  		new_examples->x[nread][attributes].index = -1;		
		new_examples->y[nread] = (double)label;
		
	} else {//dense
	
	
	string::size_type last_pos;
	int attributes = 0;
	int label;
	double value;
	
	last_pos = line.find_last_of(",");
	label = atoi(line.substr(last_pos+1,line.size()-last_pos-1).c_str()); 
	previous_pos = 0;
	pos_found = line.find_first_of(",",0);
		
	while(pos_found != last_pos) {
    	
    	if(attributes >= actual_alloc_att){
    		
    			actual_alloc_att = actual_alloc_att + NEW_ALLOC_ATT;
    			temp_x = (struct svm_node *) realloc(temp_x,actual_alloc_att*sizeof(struct svm_node));
    		
    		
    	}
    			
		value = atof(line.substr(previous_pos,pos_found-previous_pos).c_str()); 
		previous_pos = pos_found+1;	
		pos_found = line.find_first_of(",",previous_pos);
		temp_x[attributes].value = value;
    		temp_x[attributes].index = attributes+1;
  		attributes++;
	}

	new_examples->x[nread] = new svm_node [attributes+1];
  		
  	for(int i=0;i<attributes;i++){
  		
  			new_examples->x[nread][i].value = temp_x[i].value;
  			new_examples->x[nread][i].index = temp_x[i].index;
  			//cout<<"written: "<<new_examples->x[nread][i].index<<" : "<<new_examples->x[0][i].value<<endl;
    	
  	}
  	
  	new_examples->x[nread][attributes].value = 0;
  	new_examples->x[nread][attributes].index = -1;		
	new_examples->y[nread] = (double)label;
		
	}	
		
	nread++;
	
	}//end while reading examples

	new_examples->l = nread;
	numex_read_+=nread;
	return new_examples;

}

svm_problem* Stream_Handler::load_set(char* filename, int &maxNum){
	
	ifstream init_file(filename);
	int type = 1;
	string first_line;
	getline(init_file,first_line);
	
	string delimiters = ":";
	string::size_type pos = first_line.find_first_of(delimiters,0);
	if(pos < first_line.size())
		type=0;
	
	init_file.clear();
	init_file.seekg(0, ios::beg);
	
	int nexamples=0;
	
	string line;
	while(init_file.is_open() && !init_file.eof()){
		
		getline(init_file,line);	
		nexamples++;
	}
	
	if (maxNum <= 0)
		maxNum = nexamples;
	
	init_file.clear();
	init_file.seekg(0, ios::beg);
	
	
	int nread = 0;
	svm_problem* new_examples = new svm_problem [1];
	new_examples->x = new svm_node* [nexamples];
	new_examples->y = new double [nexamples];
	
	printf("loading file\n");	
	while(init_file.is_open() && !init_file.eof() && nread<maxNum){
	
	string line;
	getline(init_file,line);	
	string::size_type pos_found, previous_pos;
	
	if(type==0){//sparse
		//printf("loading line of sparse file\n");	
		int attributes = 0;
		int index;
		double label,value;
		
		pos_found = line.find_first_of(" ",0);
		//cout<<line<<endl;
		label = atof(line.substr(0,pos_found).c_str()); 
		previous_pos = pos_found;
		pos_found = line.find_first_of(":",previous_pos);
    	
			
		while((pos_found < line.size()) && (previous_pos < line.size())) {
			
			//cout<<"posfound: "<<pos_found<<" label: "<<label<<" previous_pos: "<<previous_pos<<endl;
			
		index = atoi(line.substr(previous_pos+1,pos_found-previous_pos-1).c_str()); 
    		previous_pos = pos_found;
    		pos_found = line.find_first_of(" ",previous_pos+1);
    		if(pos_found >= line.size())
    			pos_found = line.size()-1;
    		value = atof(line.substr(previous_pos+1,pos_found-previous_pos-1).c_str()); 
    		previous_pos = pos_found;
    		pos_found = line.find_first_of(":",previous_pos);
    		
    		
    		if(attributes >= actual_alloc_att){
    			actual_alloc_att = actual_alloc_att + NEW_ALLOC_ATT;
    			temp_x = (struct svm_node *) realloc(temp_x,actual_alloc_att*sizeof(struct svm_node));
    			
    		}
    		
    		temp_x[attributes].index = index;
    		temp_x[attributes].value = value; 	
    		attributes++;
    		
  		}
  		
  		new_examples->x[nread] = new svm_node [attributes+1];
  		
  		for(int i=0;i<attributes;i++){
  			new_examples->x[nread][i].value = temp_x[i].value;
  			new_examples->x[nread][i].index = temp_x[i].index;
  			//cout<<"written: "<<new_examples->x[nread][i].index<<" : "<<new_examples->x[0][i].value<<endl;
    	
  		}
  		new_examples->x[nread][attributes].value = 0;
  		new_examples->x[nread][attributes].index = -1;		
		new_examples->y[nread] = (double)label;
		
	} else {//dense
	
	
	string::size_type last_pos;
	int attributes = 0;
	int label;
	double value;
	
	last_pos = line.find_last_of(",");
	label = atoi(line.substr(last_pos+1,line.size()-last_pos-1).c_str()); 
	previous_pos = 0;
	pos_found = line.find_first_of(",",0);
		
	while(pos_found != last_pos) {
    	
    	if(attributes >= actual_alloc_att){
    		
    			actual_alloc_att = actual_alloc_att + NEW_ALLOC_ATT;
    			temp_x = (struct svm_node *) realloc(temp_x,actual_alloc_att*sizeof(struct svm_node));
    		
    		
    	}
    			
		value = atof(line.substr(previous_pos,pos_found-previous_pos).c_str()); 
		previous_pos = pos_found+1;	
		pos_found = line.find_first_of(",",previous_pos);
    	temp_x[attributes].value = value;
    	temp_x[attributes].index = attributes+1;
  		attributes++;
	}

	new_examples->x[nread] = new svm_node [attributes+1];
  		
  	for(int i=0;i<attributes;i++){
  		
  			new_examples->x[nread][i].value = temp_x[i].value;
  			new_examples->x[nread][i].index = temp_x[i].index;
  			//cout<<"written: "<<new_examples->x[nread][i].index<<" : "<<new_examples->x[0][i].value<<endl;
    	
  	}
  	
  	new_examples->x[nread][attributes].value = 0;
  	new_examples->x[nread][attributes].index = -1;		
	new_examples->y[nread] = (double)label;
		
	}	
		
	nread++;
	
	}//end while reading examples

	new_examples->l = nread;
	maxNum = nread;
	
	return new_examples;
}

svm_problem* Stream_Handler::get_random(int nexamples){

return NULL;

}

svm_problem* Stream_Handler::get_balanced(int nex_pclass){


return NULL;
}	
	

double Stream_Handler::estimate_gamma(char* filename){

	int numData = 0;
	svm_problem* prob = load_set(filename,numData);
	
	double sumDiagonal    = 0.0;
	double sumWholeKernel = 0.0;
	int count = 0;
	
	int inc = 1;
	numData = prob->l;
	
	if (numData > 5000)
	{
		inc = (int)ceil(numData/5000.0);
	}

	for(int i=0; i<numData; i+=inc)
	{
		count++;

		for (int j=i; j<numData; j+=inc)
		{
			double dot = Kernel::dot(prob->x[i], prob->x[j]);
			if (j == i)
			{
				sumDiagonal    += dot;
				sumWholeKernel += (dot/2.0);
			}
			else sumWholeKernel += dot;
		}
	}
	double gamma = (sumDiagonal - (sumWholeKernel*2)/count)*(2.0/(count-1));
	printf("gamma : %g\n",gamma);
	destroy_problem(prob);
	return gamma;
	
}

void Stream_Handler::destroy_problem(svm_problem* problem){

	for(int i=0; i<problem->l; i++){	
		delete [] problem->x[i];
	}
	delete [] problem->x;
	delete [] problem->y;	
}

bool Stream_Handler::open(char* filename){
	
	if(ifs.is_open()) ifs.close();
	
	ifs.open(filename);
	strcpy(name,filename);

	return !ifs.fail();
}

bool Stream_Handler::close(){

	if(ifs.is_open()) 
		ifs.close();
	
	if(sinfo.set){
		delete [] sinfo.class_counters;
		delete [] sinfo.labels;
	}
	
	sinfo.set = false;
	numex_read_=0;
	 
	return true;
}


bool Stream_Handler::is_sparse(){
	if(sinfo.set){
	
		return sinfo.is_sparse;
	} else {
		
		sinfo.is_sparse = determine_if_its_sparse();
		return sinfo.is_sparse;
	}
}

void Stream_Handler::set_info(){
	
	//cout<<"entering set info"<<endl;
	
	if(!sinfo.set){
		
		//cout<<"setting info"<<endl;
		sinfo.is_sparse = determine_if_its_sparse();
		if (sinfo.is_sparse)
			count_sparse();
		else
			count_dense();
	}
	
	sinfo.set = true;
	
}


bool Stream_Handler::determine_if_its_sparse(){
	
	
	istream::pos_type current_stream_position = ifs.tellg();
	ifs.clear();
	ifs.seekg(0, ios::beg);

	string line;
	getline(ifs,line);
	
	ifs.clear();
	ifs.seekg(current_stream_position); 
	
	string delimiters = ":";
	string::size_type pos = line.find_first_of(delimiters,0);
	if(pos < line.size())
		return true;
		
	return false;
}

void Stream_Handler::count_sparse(){
	
	
	istream::pos_type current_stream_position = ifs.tellg();
	ifs.clear();
	ifs.seekg(0, ios::beg);
		
	string line;
	int ntotal = 0, nr_classes=0;
	int label;
	vector<int> labels;
	string::size_type pos_found;
	
	
	while (!ifs.eof()){

		getline(ifs,line);

		pos_found = line.find_first_of(":",0);
		
		if(pos_found != string::npos){
			
			pos_found = line.find_first_of(" ",0);
			label = atoi(line.substr(0,pos_found).c_str()); 
			
			bool newlabel = true;

			for(int k=0; k < labels.size(); k++){
				if (label == labels[k]){
					newlabel = false;	
				}
			}
			
			if(newlabel){
				labels.push_back(label);		
			}
 
		ntotal++;
		
		}

	}
	nr_classes = labels.size();
	int* class_counters = new int[nr_classes];
	int* labels_array = new int[nr_classes];
	for(int k=0; k < labels.size(); k++){
		class_counters[k]=0;
		labels_array[k]=labels[k];	
	}
			

	ifs.clear(); 
	ifs.seekg(0, ios::beg);
	int ndims_not_zero=0;
	int max_index=0; 
	int last_feature;

	while (!ifs.eof()){

		getline(ifs,line);

		pos_found = line.find_first_of(":",0);
		
		if(pos_found != string::npos){
			
			pos_found = line.find_first_of(" ",0);
			label = atoi(line.substr(0,pos_found).c_str()); 
			
			for(int k=0; k < labels.size(); k++){
				if (label == labels[k]){
					class_counters[k]++;	
				}
			}
			
		}

	}
	if(SHOW_INFO_COUTING){
	printf("dataset in format sparse\n");	
	//printf("max index %d\n",max_index);	

	printf("dataset counts:\n");	
	printf("total of examples: %d\n",ntotal);
	//printf("nr. features: %d\n",max_index);	
	printf("nr. classes: %d\n",nr_classes);
	printf("distribution by classes:\n");
	for(int k=0; k < labels.size(); k++){
			printf("\t class %3d : %7d \t (%.2f%%)\n",labels[k], class_counters[k], ((double)class_counters[k]/ntotal)*100);	
	}
	}
	sinfo.nr_examples = ntotal;
	sinfo.nr_classes = nr_classes;
	sinfo.labels = labels_array;
	sinfo.class_counters = class_counters;
	sinfo.is_sparse = true;
	
	ifs.clear();
	ifs.seekg(current_stream_position); 
	
}


void Stream_Handler::count_dense(){

	istream::pos_type current_stream_position = ifs.tellg();
	ifs.seekg(0, ios::beg);
	
	string line;
	int ntotal = 0, nr_classes;
	int label;
	vector<int> labels;
	string::size_type first_pos, last_pos;
	
	while (!ifs.eof()){

		getline(ifs,line);
		last_pos = line.find_last_of(",");
		first_pos = line.find_first_of(",",0);
		
		if(first_pos != last_pos){
			
			label = atoi(line.substr(last_pos+1,line.size()-last_pos-1).c_str()); 
			bool newlabel = true;

			for(int k=0; k < labels.size(); k++){
				if (label == labels[k]){
					newlabel = false;	
				}
			}
			
			if(newlabel){
				labels.push_back(label);		
			}
 
		ntotal++;
		
		}

	}
	
	nr_classes = labels.size();
	int* class_counters = new int[nr_classes];
	int* labels_array = new int[nr_classes];
	for(int k=0; k < labels.size(); k++){
		class_counters[k]=0;
		labels_array[k]=labels[k];	
	}
			

	ifs.clear(); 
	ifs.seekg(0, ios::beg);
	
	while (!ifs.eof()){

		getline(ifs,line);

		last_pos = line.find_last_of(",");
		first_pos = line.find_first_of(",",0);
		
		if(first_pos != last_pos){
			
			label = atoi(line.substr(last_pos+1,line.size()-last_pos-1).c_str()); 
		
			for(int k=0; k < labels.size(); k++){
				if (label == labels[k]){
					class_counters[k]++;	
				}
			}
			
		}

	}
	if(SHOW_INFO_COUTING){
	printf("dataset in format dense\n");	
	printf("dataset counts:\n");	
	printf("total of examples: %d\n",ntotal);
	//printf("nr. features: %d\n",ndims);	
	printf("nr. classes: %d\n",nr_classes);
	printf("distribution by classes:\n");
	for(int k=0; k < labels.size(); k++){
			printf("\t class %3d : %7d \t (%.2f%%)\n",labels[k], class_counters[k], ((double)class_counters[k]/ntotal)*100);	
	}
	}
	sinfo.nr_examples = ntotal;
	sinfo.nr_classes = nr_classes;
	sinfo.labels = labels_array;
	sinfo.class_counters = class_counters;
	sinfo.is_sparse = false;
	
	ifs.clear();
	ifs.seekg(current_stream_position); 
	
}


void print_stream_info(stream_info* si){
if(si->set){
	if(si->is_sparse)
		cout<<"dataset in format sparse"<<endl;
	else
		cout<<"dataset in format dense"<<endl;
	
		
	cout<<"dataset counts"<<endl;	
	cout<<"total of examples: "<<si->nr_examples<<endl;
	cout<<"nr. classes: "<<si->nr_classes<<endl;
	cout<<"distribution by classes: "<<endl;
	for(int k=0; k < si->nr_classes; k++){
			printf("\t class %3d : %7d \t (%.2f%%)\n",si->labels[k], si->class_counters[k], ((double)si->class_counters[k]/si->nr_examples)*100);	
	}
	
} else {
	cout<<"structure is not set up"<<endl;
}
}
